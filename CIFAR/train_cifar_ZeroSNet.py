# from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import _LRScheduler
import time

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from tensorboardX import SummaryWriter

from models import *
from datetime import datetime
import errno
import shutil



parser = argparse.ArgumentParser(description='ZeorSNet CIFAR')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=160, help='training epoch')
parser.add_argument('--warm', type=int, default=0, help='warm up training phase')
parser.add_argument('--data', default='./data', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--arch', '-a', default='ZeroSNet20_Opt', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--bs', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--test-batch', default=32, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--weight-decay', '--wd', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 4e-5 for mobile models)')
parser.add_argument('--opt', default='SGD', type=str)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--ex", default=0, type=int)
parser.add_argument("--notes", default='', type=str)
parser.add_argument('--PL', type=float, default=1.0)
parser.add_argument('--sche', default='step', type=str)
parser.add_argument('--coe_ini', type=float, default=1)
parser.add_argument('--share_coe', type=bool, default=False)
# parser.add_argument('--given_coe', default=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9], nargs='+', type=float)
parser.add_argument('--given_coe', default=None, nargs='+', type=float)

args = parser.parse_args()
if args.given_coe is not None:
    given_coe_text = 'a_0_' + str(args.given_coe[0]) + 'a_1_' + str(args.given_coe[1]) + 'a_2_' + str(
    args.given_coe[2]) + 'b_0_' + str(args.given_coe[3])
else:
    given_coe_text = ''
if args.share_coe:
    share_coe_text = 'share_coe_True'
else:
    share_coe_text = 'share_coe_False'
if args.dataset == "cifar10":
    args.num_classes = 10
    args.epoch = 160
if args.dataset == "cifar100":
    args.num_classes = 100
    args.epoch = 300

args.save_path = 'runs/' + args.dataset + '/ZeroSNet/' + args.arch + '/PL' + str(args.PL) + 'coe_ini_' + str(
    args.coe_ini)  + given_coe_text + share_coe_text+'_sche_' + args.sche + str(args.opt) + \
                 '_BS' + str(args.bs) + '_LR' + \
                 str(args.lr) + 'epoch' + \
                 str(args.epoch) + 'warmup' + str(args.warm) + \
                 args.notes + \
                 "{0:%Y-%m-%dT%H-%M/}".format(datetime.now())

# checkpoint
if args.checkpoint is None:
    args.checkpoint = args.save_path+'checkpoint.pth.tar'
    print('args.checkpoint', args.checkpoint)

# hyper_params = {
#     'epoch': args.epoch,
#     "learning_rate": args.lr,
#     'warmup': args.warm,
#     'dataset': args.dataset,
#     'arch': args.arch,
#     "batch_size": args.bs, 
#     'momentum': args.momentum,
#     'wd': args.weight_decay,
#     'opt': args.opt,
#     'PL': args.PL,
#     'sche': args.sche,
#     'coe_ini': args.coe_ini,
#     'share_coe': args.share_coe,
#     'given_coe': args.given_coe,
#     'notes': args.notes
#     }
# experiment.log_parameters(hyper_params)


def train(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    end = time.time()
    net.train()

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if 'zerosnet' in args.arch or 'ZeroSNet' in args.arch:
            outputs, coes, stepsize = net(inputs)
        elif 'MResNet' in args.arch:
            outputs, coes = net(inputs)
        else:
            outputs = net(inputs)

        loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        loss.backward()
        optimizer.step()

        if epoch > args.warm:
            train_scheduler.step(epoch)
        if epoch <= args.warm:
            warmup_scheduler.step()
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: {:.1f}, Train set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(epoch, losses.avg, top1.avg))
    writer.add_scalar('Train/Average loss', losses.avg, epoch)
    writer.add_scalar('Train/Accuracy-top1', top1.avg, epoch)
    writer.add_scalar('Train/Accuracy-top5', top5.avg, epoch)
    writer.add_scalar('Train/Time', batch_time.sum, epoch)

    if 'zerosnet' in args.arch or 'ZeroSNet' in args.arch:
        if not isinstance(stepsize, int):
            stepsize = stepsize.data.cpu().numpy()
        writer.add_scalar('stepsize', float(stepsize), epoch)
        if coes != -1:
            if isinstance(coes, float):
                writer.add_scalar('coes', coes, epoch)
            else:
                for i in range(len(coes)):

                    if not isinstance(coes[i], float):
                        coes[i] = float(coes[i].data.cpu().numpy())
                    writer.add_scalar('coes_' + str(i), coes[i], epoch)

    return top1.avg, losses.avg, batch_time.sum

def test(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    end = time.time()
    net.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if 'zerosnet' in args.arch or 'ZeroSNet' in args.arch:
                outputs, coes, stepsize = net(inputs)
            elif 'MResNet' in args.arch:
                outputs, coes = net(inputs)
            else:
                outputs = net(inputs)
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(losses.avg, top1.avg))

    writer.add_scalar('Test/Average loss', losses.avg, epoch)
    writer.add_scalar('Test/Accuracy-top1', top1.avg, epoch)
    writer.add_scalar('Test/Accuracy-top5', top5.avg, epoch)
    writer.add_scalar('Test/Time', batch_time.sum, epoch)

    return top1.avg, losses.avg, batch_time.sum


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


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
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


if __name__ == '__main__':

    # if 1:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    start_epoch = 0
    if not os.path.isdir(args.save_path) and args.local_rank == 0:
        mkdir_p(args.save_path)

    if args.dataset == 'cifar10':
        print('==> Preparing cifar10 data..')

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data/cifar10', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, args.bs, shuffle=True, num_workers=args.workers)

        testset = torchvision.datasets.CIFAR10(
            root='./data/cifar10', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, args.bs, shuffle=False, num_workers=args.workers)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

    elif args.dataset == 'cifar100':
        print('==> Preparing cifar100 data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        trainset = torchvision.datasets.CIFAR100(
            root='./data/cifar100', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, args.bs, shuffle=True, num_workers=args.workers)

        testset = torchvision.datasets.CIFAR100(
            root='./data/cifar100', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, args.bs, shuffle=False, num_workers=args.workers)

    #        classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #                   'dog', 'frog', 'horse', 'ship', 'truck')

    print('==> Building model..')
    net_name = args.arch
    model_name = args.arch

    if args.arch == "ZeroSNet20_Tra":
        net = ZeroSNet20_Tra(PL=args.PL, coe_ini=args.coe_ini, num_classes= args.num_classes)
    elif args.arch == "ZeroSNet32_Tra":
        net = ZeroSNet32_Tra(PL=args.PL, coe_ini=args.coe_ini, num_classes= args.num_classes)
    elif args.arch == "ZeroSNet44_Tra":
        net = ZeroSNet44_Tra(PL=args.PL, coe_ini=args.coe_ini, num_classes= args.num_classes)
    elif args.arch == "ZeroSNet56_Tra":
        net = ZeroSNet56_Tra(PL=args.PL, coe_ini=args.coe_ini, num_classes= args.num_classes)
    elif args.arch == "ZeroSNet110_Tra":
        net = ZeroSNet110_Tra(PL=args.PL, coe_ini=args.coe_ini, num_classes= args.num_classes)
    elif args.arch == "ZeroSNet164_Tra":
        net = ZeroSNet164_Tra(PL=args.PL, coe_ini=args.coe_ini, num_classes= args.num_classes)
    elif args.arch == "ZeroSNet326_Tra":
        net = ZeroSNet326_Tra(PL=args.PL, coe_ini=args.coe_ini, num_classes= args.num_classes)
    elif args.arch == "ZeroSNet650_Tra":
        net = ZeroSNet650_Tra(PL=args.PL, coe_ini=args.coe_ini, num_classes= args.num_classes)
    elif args.arch == "ZeroSNet1298_Tra":
        net = ZeroSNet1298_Tra(PL=args.PL, coe_ini=args.coe_ini, num_classes= args.num_classes)


    elif args.arch == "ZeroSNet20_Opt":
        net = ZeroSNet20_Opt(PL=args.PL, num_classes= args.num_classes)
    elif args.arch == "ZeroSNet32_Opt":
        net = ZeroSNet32_Opt(PL=args.PL, num_classes= args.num_classes)
    elif args.arch == "ZeroSNet44_Opt":
        net = ZeroSNet44_Opt(PL=args.PL, num_classes= args.num_classes)
    elif args.arch == "ZeroSNet56_Opt":
        net = ZeroSNet56_Opt(PL=args.PL, num_classes= args.num_classes)
    elif args.arch == "ZeroSNet110_Opt":
        net = ZeroSNet110_Opt(PL=args.PL, num_classes= args.num_classes)
    elif args.arch == "ZeroSNet164_Opt":
        net = ZeroSNet164_Opt(PL=args.PL, num_classes= args.num_classes)
    elif args.arch == "ZeroSNet326_Opt":
        net = ZeroSNet326_Opt(PL=args.PL, num_classes= args.num_classes)
    elif args.arch == "ZeroSNet650_Opt":
        net = ZeroSNet650_Opt(PL=args.PL, num_classes= args.num_classes)
    elif args.arch == "ZeroSNet1298_Opt":
        net = ZeroSNet1298_Opt(PL=args.PL, num_classes= args.num_classes)

    elif args.arch == "MResNet20":
        net = MResNet20(PL=args.PL)
    elif args.arch == "MResNet32":
        net = MResNet32(PL=args.PL)
    elif args.arch == "MResNet44":
        net = MResNet44(PL=args.PL)
    elif args.arch == "MResNet56":
        net = MResNet56(PL=args.PL)
    elif args.arch == "MResNet110":
        net = MResNet110(PL=args.PL)
    elif args.arch == "MResNet164":
        net = MResNet164(PL=args.PL)
    elif args.arch == "MResNet326":
        net = MResNet326(PL=args.PL)
    elif args.arch == "MResNet650":
        net = MResNet650(PL=args.PL)
    elif args.arch == "MResNet1298":
        net = MResNet1298(PL=args.PL)

    elif args.arch == "MResNetSD20":
        net = MResNetSD20()
    elif args.arch == "MResNetSD110":
        net = MResNetSD110()
    elif args.arch == "MResNetC20":
        net = MResNetC20()
    elif args.arch == "MResNetC32":
        net = MResNetC32()
    elif args.arch == "MResNetC44":
        net = MResNetC44()
    elif args.arch == "MResNetC56":
        net = MResNetC56()

    elif args.arch == "DenseResNet20":
        net = DenseResNet20()
    elif args.arch == "DenseResNet110":
        net = DenseResNet110()

    elif args.arch == "ResNet_20":
        net = ResNet_20()
    elif args.arch == "ResNet_32":
        net = ResNet_32()
    elif args.arch == "ResNet_44":
        net = ResNet_44()
    elif args.arch == "ResNet_56":
        net = ResNet_56()
    elif args.arch == "ResNet_110":
        net = ResNet_110()
    elif args.arch == "ResNet_164":
        net = ResNet_164()
    elif args.arch == "ResNet_326":
        net = ResNet_326()
    elif args.arch == "ResNet_650":
        net = ResNet_650()
    elif args.arch == "ResNet_1298":
        net = ResNet_1298()

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if args.sche == 'cos':
        train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
    elif args.sche == 'step':
        if args.dataset == 'cifar100':
            train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)
        elif args.dataset == 'cifar10':
            train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    iter_per_epoch = len(trainloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    # optionally resume from a checkpoint
    title = 'CIFAR-' + args.arch
    args.lastepoch = -1
    if args.resume:
        if os.path.isfile(args.checkpoint):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.checkpoint)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            args.lastepoch = checkpoint['epoch']

    train_time = 0.0
    test_time = 0.0
    train_top1_acc = 0.0
    train_min_loss = 100
    test_top1_acc = 0.0
    test_min_loss = 100
    best_prec1 = -1
    # lr_list = []

    writer = SummaryWriter(log_dir=args.save_path)

    for epoch in range(1, args.epoch):

        # print('start time: ', "{0:%Y-%m-%dT%H-%M/}".format(datetime.now()))
        train_acc_epoch, train_loss_epoch, train_epoch_time = train(epoch)
        # print('end time: ', "{0:%Y-%m-%dT%H-%M/}".format(datetime.now()))
        train_top1_acc = max(train_top1_acc, train_acc_epoch)
        train_min_loss = min(train_min_loss, train_loss_epoch)
        train_time += train_epoch_time
        acc, test_loss_epoch, test_epoch_time = test(epoch)
        test_top1_acc = max(test_top1_acc, acc)
        test_min_loss = min(test_min_loss, test_loss_epoch)
        test_time += test_epoch_time

        if args.local_rank == 0:

            is_best = test_top1_acc > best_prec1
            best_prec1 = max(test_top1_acc, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=args.save_path)
            print('checkpoint saved.')

    writer.close()
    end_train = train_time // 60
    end_test = test_time // 60

    print(model_name)
    print("train time: {}D {}H {}M".format(end_train // 1440, (end_train % 1440) // 60, end_train % 60))
    print("test time: {}D {}H {}M".format(end_test // 1440, (end_test % 1440) // 60, end_test % 60))
    print(
        "train_acc_top1:{}, train_min_loss:{}, train_time:{}, test_top1_acc:{}, test_min_loss:{}, test_time:{}".format(
            train_top1_acc, train_min_loss, train_time, test_top1_acc, test_min_loss, test_time))
    print("args.save_path:", args.save_path)


# if __name__ == '__main__':
#     d = torch.rand(2, 3, 32, 32)
#     # net = ZeroSNet20_rec()
#     # net = ZeroSNet164_Tra()
#     # net = ZeroSNet650_Tra()
#     net=ResNet_20()
#     # net = MResNet164()
#     # net = ResNet_164()
#     # net = ResNet_650()
#     o = net(d)
#     probs = F.softmax(o).detach().numpy()[0]
#     pred = np.argmax(probs)
#     print('pred, probs', pred, probs)
#
#     total_params = sum(p.numel() for p in net.parameters())
#     print(f'{total_params:,} total parameters.')
#     total_trainable_params = sum(
#         p.numel() for p in net.parameters() if p.requires_grad)
#     print(f'{total_trainable_params:,} training parameters.')
#     for name, parameters in net.named_parameters():
#         print(name, ':', parameters.size())
# #     onnx_path = "onnx_model_name.onnx"
# #     torch.onnx.export(net, d, onnx_path)
# #     #
# #     netron.start(onnx_path)
# # #
