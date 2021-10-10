from __future__ import print_function

import sys
import os
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn

from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy, robust_acc
from util import set_optimizer
from networks.resnet_big import SupConResNet, LinearClassifier

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'waterbirds'], help='dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/SupConLinear/{}_models'.format(opt.dataset)
    opt.log_path = './save/SupConLinear/{}_logs'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.log_folder = os.path.join(opt.log_path, opt.model_name)
    if not os.path.isdir(opt.log_folder):
        os.makedirs(opt.log_folder)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'waterbirds':
        opt.n_cls = 2
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """
    one epoch training
    return:
    avg loss (float)
    top1 accuracy (AverageMeter)
    group accuracy (array of each group -- 4 in the case of waterbirds, none in the case of cifar)
    """
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()

    if opt.dataset == 'waterbirds':
        groups = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
        for idx, (images, labels, group) in enumerate(train_loader):
            data_time.update(time.time() - end)

            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # warm-up learning rate
            warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

            # compute loss
            with torch.no_grad():
                features = model.encoder(images)
            output = classifier(features.detach())
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1 = accuracy(output, labels, topk=(1, 1))
            acc = robust_acc(output, group)
            for i in range(4):
                groups[i].update(acc[i], bsz)
            top1.update(acc1[0], bsz)

            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (idx + 1) % opt.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'.format(
                    epoch, idx + 1, len(train_loader)))
                print('BT {batch_time.val:.3f} (Avg: {batch_time.avg:.3f}, Total: {batch_time.sum:.3f})\t'.format(batch_time=batch_time))
                print('DT {data_time.val:.3f} (Avg: {data_time.avg:.3f}, Total: {data_time.sum:.3f})\t'.format(data_time=data_time))
                print('loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(loss=losses))
                print(top1)
                print(top1.val)
                print(top1.avg)
                print('Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(top1=top1))
                print('Group Acc@1 {acc0:.3f} {acc1:.3f} {acc2:.3f} {acc3:.3f}'.format(acc0=groups[0], acc1=groups[1], acc2=groups[2], acc3=groups[3]))
                print('Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} (Avg: {batch_time.avg:.3f}, Total: {batch_time.sum:.3f})\t'
                    'DT {data_time.val:.3f} (Avg: {data_time.avg:.3f}, Total: {data_time.sum:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Group Acc@1 {acc0:.3f} {acc1:.3f} {acc2:.3f} {acc3:.3f}'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, 
                    acc0=groups[0], acc1=groups[1], acc2=groups[2], acc3=groups[3]))
                sys.stdout.flush()
        print("Done for loop")
    else:
        group = []
        for idx, (images, labels) in enumerate(train_loader):
            data_time.update(time.time() - end)

            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # warm-up learning rate
            warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

            # compute loss
            with torch.no_grad():
                features = model.encoder(images)
            output = classifier(features.detach())
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (idx + 1) % opt.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} (Avg: {batch_time.avg:.3f}, Total: {batch_time.sum:.3f})\t'
                    'DT {data_time.val:.3f} (Avg: {data_time.avg:.3f}, Total: {data_time.sum:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))
                sys.stdout.flush()

    return losses.avg, top1, group


def validate(val_loader, model, classifier, criterion, opt):
    """
    validation
    return:
    avg loss (list of floats [val, test(opt)])
    top1 accuracy (list of AverageMeter [val, test(opt)])
    group accuracy (list of array of each group [val, test(opt)])
    """
    model.eval()
    classifier.eval()

    losses = [AverageMeter()]
    top1 = [AverageMeter()]

    
    if opt.dataset == 'waterbirds':
        losses.append(AverageMeter())
        top1.append(AverageMeter())
        group = [[AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()], 
                 [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]]
        with torch.no_grad():
            end = time.time()
            for dtype, validate_loader in enumerate(val_loader):
                batch_time = AverageMeter()
                for idx, (images, labels) in enumerate(validate_loader):
                    images = images.float().cuda()
                    labels = labels.cuda()
                    bsz = labels.shape[0]

                    # forward
                    output = classifier(model.encoder(images))
                    loss = criterion(output, labels)

                    # update metric
                    losses[dtype].update(loss.item(), bsz)
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    acc = robust_acc(output, group)
                    for i in range(4):
                        group[dtype][i].update(acc[i], bsz)
                    top1[dtype].update(acc1[0], bsz)

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if idx % opt.print_freq == 0:
                        print('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} (Avg: {batch_time.avg:.3f}, Total: {batch_time.sum:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Group Acc@1 {acc0:.3f} {acc1:.3f} {acc2:.3f} {acc3:.3f}'.format(
                            idx, len(validate_loader), batch_time=batch_time,
                            loss=losses, top1=top1,
                            acc0=groups[0], acc1=groups[1], acc2=groups[2], acc3=groups[3]))
 
                    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    else:
        batch_time = AverageMeter()
        group = [[]]
        with torch.no_grad():
            end = time.time()
            for idx, (images, labels) in enumerate(val_loader):
                images = images.float().cuda()
                labels = labels.cuda()
                bsz = labels.shape[0]

                # forward
                output = classifier(model.encoder(images))
                loss = criterion(output, labels)

                # update metric
                losses[0].update(loss.item(), bsz)
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                top1[0].update(acc1[0], bsz)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if idx % opt.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} (Avg: {batch_time.avg:.3f}, Total: {batch_time.sum:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        idx, len(val_loader), batch_time=batch_time,
                        loss=losses, top1=top1))

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return [loss.avg for loss in losses], top1, group


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier) #SGD
    
    # logs
    log_file = open(opt.log_folder+"/log.csv", "a")
    header = "epoch,avg_train_acc,avg_train_count,avg_val_acc,avg_val_count"
    if opt.dataset == "waterbirds":
        header += ",avg_test_acc,avg_test_count"
        for group in range(4):
            for dtype in ["train", "val", "test"]:
                header += ",avg_{dtype}_acc:group_{group},avg_{dtype}count:group_{group}".\
                            format(dtype=dtype, group=group)
    log_file.write(header+"\n")

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc, group = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc.avg))

        # eval for one epoch
        loss, val_acc, val_group = validate(val_loader, model, classifier, criterion, opt)

        # save train, val loss, acc, group(s) to csv
        row = "{epoch},{avg_train_acc},{avg_train_count}".\
                format(epoch=epoch, avg_train_acc=acc.avg, avg_train_count=acc.count)
        for acc in val_acc:
            row += ',{},{}'.format(acc.avg, acc.count)
        for i in range(len(group)):
                         # train, val, test
            for dtype in [group, val_group[0], val_group[1]]:
                row += ',{},{}'.format(dtype[i].avg, dtype[i].count)
        
        log_file.write(row+"\n")
   
        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    log_file.close()

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
