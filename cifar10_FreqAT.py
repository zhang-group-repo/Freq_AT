import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
import datetime
from models import *
from earlystop import earlystop
import numpy as np
from utils import *
import attack_generator as attack
from attacks import *


parser = argparse.ArgumentParser(description='FreqAT train')
parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
parser.add_argument('--num_steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step_size', type=float, default=0.007, help='step size')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="resnet18",help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--classes', type=int, default=10, help='models classes')
parser.add_argument('--tau', type=int, default=0, help='step tau')
parser.add_argument('--beta',type=float,default=6.0,help='regularization parameter')
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn,cifar100,Tiny_imagenet")
parser.add_argument('--rand_init', type=bool, default=True, help="whether to initialize adversarial sample with random noise")
parser.add_argument('--omega', type=float, default=0.0, help="random sample parameter")
parser.add_argument('--dynamictau', type=bool, default=True, help='whether to use dynamic tau')
parser.add_argument('--depth', type=int, default=32, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10, help='WRN width factor')
parser.add_argument('--drop_rate', type=float, default=0.0, help='WRN drop rate')
parser.add_argument('--out_dir',type=str,default='./cifar10/results',help='dir of output')
parser.add_argument('--resume', type=str, default='', help='whether to resume training, default: None')

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def MART_loss(adv_logits, natural_logits, target, beta):
    # Based on the repo MART https://github.com/YisenWang/MART
    kl = nn.KLDivLoss(reduction='none')
    batch_size = len(target)
    adv_probs = F.softmax(adv_logits, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == target, tmp1[:, -2], tmp1[:, -1])
    loss_adv = F.cross_entropy(adv_logits, target) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    nat_probs = F.softmax(natural_logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (target.unsqueeze(1)).long()).squeeze()
    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust
    return loss


def TRADES_loss(adv_logits, natural_logits, target, beta):
    # Based on the repo TREADES: https://github.com/yaodongyu/TRADES
    batch_size = len(target)
    criterion_kl = nn.KLDivLoss(size_average=False).cuda()
    loss_natural = nn.CrossEntropyLoss(reduction='mean')(natural_logits, target)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits, dim=1),
                                                         F.softmax(natural_logits, dim=1))
    loss = loss_natural + beta * loss_robust
    return loss

def train(model, train_loader, optimizer, tau, fft_h=10):
    starttime = datetime.datetime.now()
    loss_sum = 0
    bp_count = 0
    atk=PGD(model)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        
        # 图像处理方法 fft_reconstruct, wavelet_transform_multi_level, jpeg_compress
        # fft_data = fft_reconstruct(data , num_l=fft_h)
        
        output_adv = atk(data, target)
        output_adv = fft_reconstruct(output_adv , num_l=fft_h)
        # output_adv = wavelet_transform_multi_level(output_adv)
        # output_adv = jpeg_compress(output_adv, quality=70)
        output_target = target
        output_natural = data
        bp_count += 0
        model.train()
        optimizer.zero_grad()
        output = model(output_adv)

        # calculate standard adversarial training loss
        loss = nn.CrossEntropyLoss(reduction='mean')(output, output_target)

        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

        # calculate MART adversarial training loss
        # loss = MART_loss(adv_logits, natural_logits, output_target, args.beta)

        # loss_sum += loss.item()
        # loss.backward()
        # optimizer.step()

        # calculate TRADES adversarial training loss
        # loss = TRADES_loss(adv_logits,natural_logits,output_target,args.beta)

        # loss_sum += loss.item()
        # loss.backward()
        # optimizer.step()


    bp_count_avg = bp_count / len(train_loader.dataset)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds




    return time, loss_sum, bp_count_avg

def adjust_tau(epoch, dynamictau):
    tau = args.tau
    if dynamictau:
        if epoch <= 50:
            tau = 0
        elif epoch <= 90:
            tau = 1
        else:
            tau = 2
    return tau


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 60:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 110:
        lr = args.lr * 0.005
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, checkpoint=out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

print('==> Load Test Data')
if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
if args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
if args.dataset == "svhn":
    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

print('==> Load Model')
if args.net == "smallcnn":
    model = SmallCNN().cuda()
    net = "smallcnn"
if args.net == "resnet18":
    model = load_model("ResNet18",args.classes)
    net = "resnet18"
if args.net == "resnet34":
    model = load_model("ResNet34",args.classes)
    net = "resnet34"
if args.net == "WRN34-10":
    model = load_model("WRN34-10",args.classes)
    net = "WRN34-10"
model = torch.nn.DataParallel(model)
print(net)

model = torch.nn.DataParallel(model)
model = model.to("cuda")
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

start_epoch = 0
# Resume
title = 'FreqAT train'
if args.resume:
    # resume directly point to checkpoint.pth.tar e.g., --resume='./out-dir/checkpoint.pth.tar'
    print('==> Friendly Adversarial Training Resuming from checkpoint ..')
    print(args.resume)
    assert os.path.isfile(args.resume)
    out_dir = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title, resume=True)
else:
    print('==> Friendly Adversarial Training')
    logger_test = Logger(os.path.join(args.out_dir, 'log_results.txt'), title=title)
    # logger_test.set_names(['Epoch', 'Natural Test Acc', 'FGSM Acc', 'PGD20 Acc', 'CW Acc'])
    logger_test.set_names(['Epoch', 'Natural Test Acc',  'PGD20 Acc'])


import math

def sched_no(start, end, pos):
    """无变化调度."""
    return start

def sched_lin(start, end, pos):
    """线性调度."""
    return start + pos * (end - start)

def sched_cos(start, end, pos):
    """余弦调度."""
    return start + (1 + math.cos(math.pi * (1 - pos))) * (end - start) / 2

def sched_exp(start, end, pos):
    """指数调度."""
    return start * (end / start) ** pos

def adjust_fft_h(epoch, total_epochs, start_fft_h, end_fft_h, schedule_func):
    """
    根据给定的调度函数调整fft_h的值。
    
    :param epoch: 当前训练周期
    :param total_epochs: 总训练周期数
    :param start_fft_h: fft_h的初始值
    :param end_fft_h: fft_h的最终值
    :param schedule_func: 调度函数
    :return: 调整后的fft_h值
    """
    pos = epoch / total_epochs
    return schedule_func(start_fft_h, end_fft_h, pos)


class SchedCos:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __call__(self, pos):
        return self.start + (1 + math.cos(math.pi * (1 - pos))) * (self.end - self.start) / 2

def combine_scheds(pcts, scheds):
    """
    组合调度函数。
    
    :param pcts: 每个调度占总训练周期的比例
    :param scheds: 调度函数列表
    :return: 组合后的调度函数
    """
    assert sum(pcts) == 1.0, "比例的总和必须为1"
    pcts = [p/sum(pcts) for p in pcts]  # 确保比例之和为1
    def _inner(pos):
        idx = 0
        for i, pct in enumerate(pcts):
            if pos < sum(pcts[:i+1]):
                idx = i
                break
        actual_pos = (pos - sum(pcts[:idx])) / pcts[idx]
        return scheds[idx](actual_pos)
    return _inner


test_nat_acc = 0
fgsm_acc = 0
test_pgd20_acc = 0
cw_acc = 0
best_epoch = 0
start_fft_h = 7
end_fft_h = 12

# 定义两个不同阶段的余弦调度
sched1 = SchedCos(end_fft_h, start_fft_h)
sched2 = SchedCos(start_fft_h, end_fft_h)

# 组合调度，前30%使用sched1，后70%使用sched2
combined_sched = combine_scheds([0.3, 0.7], [sched1, sched2])


for epoch in range(start_epoch, args.epochs):

    adjust_learning_rate(optimizer, epoch + 1)
    pos = epoch / args.epochs
    # 选用不同调度算法 combined,no,lin,cos,exp
    fft_h = combined_sched(pos)
    print(f"Epoch {epoch+1}, Param Value: {fft_h}")
    # 在这里进行训练，使用更新后的参数值


    train_time, train_loss, bp_count_avg = train(model, train_loader, optimizer, adjust_tau(epoch + 1, args.dynamictau), fft_h=fft_h)

    # 
    loss, test_nat_acc = attack.eval_clean(model, test_loader, fft_h=fft_h)

    # 
    loss, test_pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=10, epsilon=0.031, step_size=0.031 / 4,loss_fn="cent", category="Madry", rand_init=True, fft_h=fft_h)

    print(
        # 'Epoch: [%d | %d] | Train Time: %.2f s | BP Average: %.2f | Natural Test Acc %.2f | FGSM Test Acc %.2f | PGD20 Test Acc %.2f | CW Test Acc %.2f |\n' % (
        'Epoch: [%d | %d] | Train Time: %.2f s | BP Average: %.2f | Natural Test Acc %.2f | PGD10 Test Acc %.2f |\n' % (
        epoch + 1,
        args.epochs,
        train_time,
        bp_count_avg,
        test_nat_acc,
        test_pgd20_acc,
        )
        )

    # logger_test.append([epoch + 1, test_nat_acc, fgsm_acc, test_pgd20_acc, cw_acc])
    logger_test.append([epoch + 1, test_nat_acc, test_pgd20_acc])

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'bp_avg': bp_count_avg,
        'test_nat_acc': test_nat_acc,
        'test_pgd20_acc': test_pgd20_acc,
        'optimizer': optimizer.state_dict(),
    })


