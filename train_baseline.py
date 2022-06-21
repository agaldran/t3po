DEBUG = False

import argparse
import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.utils import seed_torch, str2bool, init_experiment
from data.open_set_datasets import get_class_splits, get_datasets
from utils.schedulers import get_scheduler

########################################################################################################################
import os, os.path as osp
from utils.get_model import get_arch

from tqdm import tqdm


########################################################################################################################

parser = argparse.ArgumentParser("Training")
# Dataset
parser.add_argument('--dataset', type=str, default='kather2016', help="")
parser.add_argument('--image_size', type=int, default=150)
# optimization
parser.add_argument('--optim', type=str, default='adam', help="Which optimizer to use {adam, sgd, adam_sam}")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01, help="learning rate for model")
parser.add_argument('--weight_decay', type=float, default=0.0, help="LR regularisation on weights")
parser.add_argument('--momentum', type=float, default=0.0, help="momentum for SGD")
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--stop_epoch', type=int, default=-1)
parser.add_argument('--scheduler', type=str, default='cosine_warm_restarts_warmup')
parser.add_argument('--num_restarts', type=int, default=0, help='How many restarts for cosine_warm_restarts schedule')
# model
parser.add_argument('--model', type=str, default='mobilenet')
parser.add_argument('--dropout_p', type=float, default=0.0, help="dropout for classifier")
# aug
parser.add_argument('--transform', type=str, default='trivial-augment_wide')
# misc
parser.add_argument('--verbose', default=False, type=str2bool, help='print stats to screen during training', metavar='BOOL')
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--device', default='cuda:0', type=str, help='device (cuda or cpu, default: cuda:0)')
parser.add_argument('--eval_freq', type=int, default=20)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--split_idx', default=0, type=int, help='OSR splits for each dataset, see data/open_set_splits/osr_splits.py')
parser.add_argument('--use_softmax_in_eval', default=False, type=str2bool, help='Do we use softmax or logits for evaluation', metavar='BOOL')
parser.add_argument('--save_path', type=str, default='experiments')

def save_networks(network, filename):
    weights = network.state_dict()
    torch.save(weights, filename)

class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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

def train_one_epoch(net, criterion, optimizer, trainloader):
    net.train()
    class_losses, train_acc = AverageMeter(), AverageMeter()
    torch.cuda.empty_cache()
    loss_all, acc_tr = 0, 0
    n_correct, total = 0, 0

    for batch_idx, (data, labels) in enumerate(tqdm(trainloader)):
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            logits_class = net(data)
            # compute loss for logits_class and labels
            loss_class = criterion(logits_class, labels)
            probs_class = logits_class.softmax(dim=1)
            preds_class = probs_class.max(dim=1)[1]
            loss_class.backward()
            optimizer.step()
            n_correct += (preds_class == labels).sum()
            total += labels.size(0)

        class_losses.update(loss_class.item(), data.size(0))
        train_acc.update(n_correct/total, data.size(0))
        acc_tr = 100*train_acc.avg # for printing
        loss_all += class_losses.avg

    return class_losses.avg, acc_tr, get_mean_lr(optimizer)


def get_optimizer(args, params_list):
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        if args.weight_decay == 0:
            optimizer = torch.optim.Adam(params_list, lr=args.lr)
        else:
            optimizer = torch.optim.AdamW(params_list, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    return optimizer


def get_mean_lr(optimizer):
    return torch.mean(torch.Tensor([param_group['lr'] for param_group in optimizer.param_groups])).item()

def train_model(args):

    if args.dropout_p == 0:
        from test import test_model
    else:
        from test import test_dropout_model as test_model

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Currently using GPU: {}".format(args.device))
        cudnn.benchmark = False
    else:
        print("Currently using CPU")

    train_loader, val_loader, out_loader = dataloaders['train'], dataloaders['val'], dataloaders['test_unknown']

    # Get base network and criterion
    print("Creating model: {}".format(args.model))
    net = get_arch(args.model, len(args.train_classes), additional_classes=args.additional_classes, dropout_p=args.dropout_p, pretrained=True)
    criterion = torch.nn.CrossEntropyLoss()

    if use_gpu:
        net = net.cuda()

    optimizer = get_optimizer(args=args, params_list=net.parameters())
    scheduler = get_scheduler(optimizer, args)
    start = time.time()

    best_acc, best_auroc, best_epoch, checkpointed_model = 0, 0, 0, False

    with open(osp.join(args.log_dir, 'log.txt'), 'a') as f: print(100 * "=", file=f)

    for epoch in range(1, args.max_epoch + 1):
        print("==> Epoch {}/{}".format(epoch, args.max_epoch))
        # print('NO TRAINING')
        m1, m2, m3 = train_one_epoch(net, criterion, optimizer, train_loader)
        print('Class loss= {:.4f}, Class acc = {:.2f} -- LR = {:.7f}'.format(m1, m2, m3))

        if epoch % args.eval_freq == 0 or epoch == args.max_epoch:
            with open(osp.join(args.log_dir, 'log.txt'), 'a') as f:
                print("==> Epoch {}/{}".format(epoch, args.max_epoch), file=f)
            # test train_loader/val_loader
            acc_train = test_model(net, train_loader, args)
            acc_val = test_model(net, val_loader, args)

            print(100 * "-")
            print('TRAIN Set Acc. = {:.2f} -- CLOSED VAL Set Accuracy = {:.2f}'.format(acc_train, acc_val))
            with open(osp.join(args.log_dir, 'log.txt'), 'a') as f:
                print('TRAIN Set Acc. = {:.2f} -- CLOSED VAL Set Accuracy = {:.2f}'.format(acc_train, acc_val), file=f)
                print(100 * "-", file=f)

            # checkpointing best acc_val model only if we are halfway in training, avoids spurious performance peaks
            if acc_val > best_acc:
                if epoch > args.max_epoch//2:
                    print('-------- Best Closed Set Accuracy Attained, {:.2f} --> {:.2f}. Checkpointing. --------'.format(best_acc, acc_val))
                    checkpointed_model = True
                    for f in os.listdir(args.model_dir): os.remove(osp.join(args.model_dir, f))
                    save_path_best = osp.join(args.model_dir, 'net_state_dict_ep{}_acc{:.2f}.pth'.format(epoch, best_acc))
                    save_networks(net, save_path_best)
                else:
                    print('-------- Best Closed Set Accuracy Attained, {:.2f} --> {:.2f}. --------'.format(best_acc, acc_val))
                best_acc, best_epoch = acc_val, epoch
            else:
                print('-------- Best Closed Set Accuracy so far {:.2f} at epoch {:d} --------'.format(best_acc, best_epoch))
            print(100 * "-")
        if args.scheduler == 'plateau' or args.scheduler == 'warm_restarts_plateau':
            scheduler.step(best_acc, epoch)
        elif args.scheduler == 'multi_step':
            scheduler.step()
        else:
            scheduler.step(epoch=epoch)
        if epoch == args.stop_epoch: break

    if not checkpointed_model: # save the last model if no model was saved during training
        save_path_best = osp.join(args.model_dir, 'net_state_dict_ep{}_acc{:.2f}.pth'.format(epoch, acc_val))
        save_networks(net, save_path_best)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)

    print('Finished. Total elapsed time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))
    with open(osp.join(args.log_dir, 'log.txt'), 'a') as f:
        print('Finished. Total elapsed time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds), file=f)

    test_loader, out_loader = dataloaders['test_known'], dataloaders['test_unknown']
    net.load_state_dict(torch.load(save_path_best, map_location=torch.device('cpu')))
    if use_gpu: net = net.cuda()
    results = test_model(net, test_loader, args, out_loader)
    print(100 * "*")
    print('CLOSED TEST Set Accuracy = {:.2f} -- AUROC = {:.2f} --- Open Set Score = {:.2f}'.format(results['ACC'], results['AUROC'], results['OSCR']))
    with open(osp.join(args.log_dir, 'log.txt'), 'a') as f:
        print(100 * "*", file=f)
        print('CLOSED TEST Set Accuracy = {:.2f} -- AUROC = {:.2f} --- Open Set Score = {:.2f}'.format(results['ACC'], results['AUROC'], results['OSCR']), file=f)
        print(100 * "*", file=f)
    print(100 * "*")
    return save_path_best


if __name__ == '__main__':
    args = parser.parse_args()
    seed_torch(args.seed)
    args.epochs = args.max_epoch
    img_size = args.image_size
    results = dict()

    print('dataset:',args.dataset, 'split_idx:', args.split_idx)
    args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx)
    args.save_path = osp.join(args.save_path, 'split_{}/seed_{}'.format(args.split_idx, args.seed))
    args = init_experiment(args, args.save_path)

    # get_datasets receives args.split_idx, which determines what split to use
    # alternatively one can manually pass known_classes, open_set_classes as lists, that overrides evth.
    datasets = get_datasets(args.dataset, transform=args.transform, image_size=args.image_size, seed=args.seed, args=args)
    # # datasets is a dict with keys 'train', 'val', 'test_known', 'test_unknown', check them out:
    # print(type(datasets), list(datasets.keys()))

    dataloaders = {}
    for k, ds, in datasets.items():
        shuffle = True if k == 'train' else False
        dataloaders[k] = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)
    # dataloaders is a dict with keys 'train', 'val', 'test_known', 'test_unknown', check it out:
    additional_classes = 0 # len(args.train_classes)  # this will be for # of transforms
    vars(args).update(
        {
            'known': args.train_classes,
            'unknown': args.open_set_classes,
            'img_size': img_size,
            'dataloaders': dataloaders,
            'num_classes': len(args.train_classes),
            'additional_classes': additional_classes
        }
    )

    train_model(args)

