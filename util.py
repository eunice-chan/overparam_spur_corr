from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        # print("Target", type(target))
        # print(target)
        # print("Batch size", batch_size)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print("Pred")
        # print(pred)
        # print("Correct")
        # print(correct)

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            # print(correct_k)
            # print(correct_k.mul_(100.0 / batch_size))
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def robust_acc(output, target):
    with torch.no_grad():
        print("Target", type(target))
        target = target.to(torch.device("cuda"))
        print(target)
        batch_size = target.size(0)
        print("Batch size", batch_size)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        res = []
        for group in range(4):
            target_group = target.eq(group)
            target_count = sum(target_group)
            print("Target count", target_count)
            print("Target group", group, type(target_group))
            print(target_group)
            print(target_group.view(1, -1).expand_as(pred))
            print("Pred")
            print(pred)
            print("Correct")
            correct = pred.eq(target_group.view(1, -1).expand_as(pred))
            print(correct)
            correct = correct[:1].view(-1).float().sum(0, keepdim=True)
            print(correct)
            print(correct.mul_(100.0 / batch_size))
            res.append(correct.mul_(100.0 / batch_size))
        print(output, pred, target, group, res, target_group)
        return res

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
