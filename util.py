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

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def robust_acc(output, target, group):
    with torch.no_grad():
        target = target.to(torch.device("cuda"))
        group = group.to(torch.device("cuda"))
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))[:1].view(-1)

        # pred=pred.cpu().numpy()[0]
        # print("Batch size", batch_size)
        # print("Group | Target | Pred | Correct")
        # for groupi, targeti, predi, correcti in zip(group, target, pred, correct):
        #     print(groupi.item(), "|", targeti.item(), "|", predi, "|", correcti.item())

        res = []
        for i in range(4):
            this_group = group.eq(i)
            group_count = this_group.sum(0, keepdim=True).item()
            group_acc = correct[this_group]
            # print(i, "group size", group_count)
            # print("This Group? | Group | Target | Pred | Correct")
            # for tg, groupi, targeti, predi, correcti in zip(this_group, group, target, pred, correct):
            #     print(tg.item(), "|", groupi.item(), "|", targeti.item(), "|", predi, "|", correcti.item())
            # print(group_acc, sum(group_acc))
            if group_count:
                 # print("ACC", group_acc.float().sum(0, keepdim=True).mul_(100.0 / group_count))
                res.append([group_acc.float().sum(0, keepdim=True).mul_(100.0 / group_count), group_count])
            else:
                print("Group", i, "has no examples in this batch")
                res.append([torch.Tensor([100]), 0])
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
