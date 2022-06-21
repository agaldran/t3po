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
import sys, os, os.path as osp
from utils.get_model import get_arch

from tqdm import tqdm
from test import test_model_t3po_single


########################################################################################################################

parser = argparse.ArgumentParser("Training")
# Dataset
parser.add_argument('--dataset', type=str, default='kather2016', help="")
parser.add_argument('--out-num', type=int, default=10, help='For cifar-10-100')
parser.add_argument('--image_size', type=int, default=150)
# optimization
parser.add_argument('--optim', type=str, default='adam', help="Which optimizer to use {adam, sgd, adam_sam}")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01, help="learning rate for model")
parser.add_argument('--weight_decay', type=float, default=0.0, help="LR regularisation on weights")
parser.add_argument('--weighted_ce', default=True, type=str2bool, help='weigh ce values per nr of transforms', metavar='BOOL')
parser.add_argument('--momentum', type=float, default=0.0, help="momentum for SGD")
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--stop_epoch', type=int, default=-1)
parser.add_argument('--scheduler', type=str, default='cosine_warm_restarts_warmup')
parser.add_argument('--num_restarts', type=int, default=0, help='How many restarts for cosine_warm_restarts schedule')
# model
parser.add_argument('--model', type=str, default='mobilenet_2heads')
parser.add_argument('--dropout_p', type=float, default=0.0, help="dropout for classifier")
# aug
parser.add_argument('--transform', type=str, default='T3PO_color_wide')
# misc
parser.add_argument('--verbose', default=False, type=str2bool, help='print stats to screen during training', metavar='BOOL')
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--device', default='cuda:0', type=str, help='device (cuda or cpu, default: cuda:0)')
parser.add_argument('--eval_freq', type=int, default=20)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--split_idx', default=0, type=int, help='OSR splits for each dataset, see data/open_set_splits/osr_splits.py')
parser.add_argument('--use_softmax_in_eval', default=False, type=str2bool, help='Do we use softmax or logits for evaluation', metavar='BOOL')
parser.add_argument('--save_path', type=str, default='')

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
    n_augs = trainloader.dataset.transforms.transform.n_augs
    weight=torch.zeros(n_augs)
    weight[0]=n_augs

    net.train()
    losses_class, losses_transforms, train_acc_class, train_acc_transforms = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    torch.cuda.empty_cache()
    loss_all, acc_tr = 0, 0
    n_correct_class, n_correct_transforms, total = 0, 0, 0
    for (images, labels_transform), labels_class in tqdm(trainloader):
        if torch.cuda.is_available():
            images, labels_transform, labels_class, weight  = images.cuda(), labels_transform.cuda(), labels_class.cuda(), weight.cuda()
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()

            logits_class, logits_transforms = net(images)

            # compute loss for classification
            loss_class = criterion(logits_class, labels_class)
            # compute loss for transform prediction
            if args.weighted_ce:
                loss_transforms = torch.nn.functional.cross_entropy(logits_transforms, labels_transform, weight=weight)
            else:
                loss_transforms = torch.nn.functional.cross_entropy(logits_transforms, labels_transform)
            preds_class = logits_class.max(dim=1)[1]
            preds_transforms = logits_transforms.max(dim=1)[1]

            (loss_class+loss_transforms).backward()
            optimizer.step()
            n_correct_class += (preds_class == labels_class).sum()
            n_correct_transforms += (preds_transforms == labels_transform).sum()
            total += labels_class.size(0)

        losses_class.update(loss_class.item(), images.size(0))
        losses_transforms.update(loss_transforms.item(), images.size(0))
        train_acc_class.update(n_correct_class/total, images.size(0))
        train_acc_transforms.update(n_correct_transforms/total, images.size(0))
        acc_tr_class = 100*train_acc_class.avg # for printing
        acc_tr_transforms = 100 * train_acc_transforms.avg  # for printing
        loss_all += (losses_class.avg+losses_transforms.avg)

    return losses_class.avg, losses_transforms.avg, acc_tr_class, acc_tr_transforms, get_mean_lr(optimizer)

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

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Currently using GPU: {}".format(args.device))
        cudnn.benchmark = False
    else:
        print("Currently using CPU")

    train_loader, val_loader, out_loader = dataloaders['train'], dataloaders['val'], dataloaders['test_unknown']
    out_loader.dataset.transforms.transform.fast_test = True

    # Get base network and criterion
    print("Creating model: {}".format(args.model))
    net = get_arch(args.model, len(args.train_classes), additional_classes=args.additional_classes, dropout_p=args.dropout_p, pretrained=True)
    criterion = torch.nn.CrossEntropyLoss()
    if use_gpu: net = net.cuda()

    optimizer = get_optimizer(args=args, params_list=net.parameters())
    scheduler = get_scheduler(optimizer, args)
    start = time.time()

    best_acc, best_auroc, best_epoch, checkpointed_model = 0, 0, 0, False

    with open(osp.join(args.log_dir, 'log.txt'), 'a') as f: print(100 * "=", file=f)

    for epoch in range(1, args.max_epoch + 1):
        print("==> Epoch {}/{}".format(epoch, args.max_epoch))
        # # print('NO TRAINING')
        l_class, l_tr, acc_class, acc_tr, lr = train_one_epoch(net, criterion, optimizer, train_loader)
        print('Class loss= {:.4f}, Transform loss= {:.4f}, Class acc = {:.2f}, Transf. acc = {:.2f} -- '
              'LR = {:.7f}'.format(l_class, l_tr, acc_class, acc_tr, lr))

        if epoch % args.eval_freq == 0 or epoch == args.max_epoch:
            with open(osp.join(args.log_dir, 'log.txt'), 'a') as f:
                print("==> Epoch {}/{}".format(epoch, args.max_epoch), file=f)
            # test train_loader
            acc_train = test_model_t3po_single(net, train_loader, args)
            # test val_loader
            val_loader.dataset.transforms.transform.fast_test = True
            acc_val = test_model_t3po_single(net, val_loader, args)
            val_loader.dataset.transforms.transform.fast_test = False
            print(100 * "-")
            print('TRAIN Set Acc. = {:.2f} -- CLOSED VAL Set Accuracy = {:.2f}'.format(acc_train, acc_val))
            with open(osp.join(args.log_dir, 'log.txt'), 'a') as f:
                print('TRAIN Set Acc. = {:.2f} -- CLOSED VAL Set Accuracy = {:.2f}'.format(acc_train, acc_val), file=f)
                print(100 * "-", file=f)

            if acc_val > best_acc: # checkpointing best acc_val model only if we are halfway in training, avoids spurious performance peaks
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
    test_loader.dataset.transforms.transform.fast_test = True
    out_loader.dataset.transforms.transform.fast_test = True

    net.load_state_dict(torch.load(save_path_best, map_location=torch.device('cpu')))
    if torch.cuda.is_available(): net = net.cuda()
    results = test_model_t3po_single(net, test_loader, args, out_loader)

    print(100 * "=")
    print('CLOSED TEST Set Accuracy = {:.2f} -- AUROC/AUROC_TR = {:.2f}/{:.2f} --- Open Set Score = {:.2f}'.format(results['ACC'], results['AUROC'], results['AUROC_TR'], results['OSCR']))
    with open(osp.join(args.log_dir, 'log.txt'), 'a') as f:
        print(100 * "*", file=f)
        print('CLOSED TEST Set Accuracy = {:.2f} -- AUROC/AUROC_TR = {:.2f}/{:.2f} --- Open Set Score = {:.2f}'.format(results['ACC'], results['AUROC'], results['AUROC_TR'], results['OSCR']), file=f)
        print(100 * "*", file=f)
    print(100 * "*")
    return save_path_best


if __name__ == '__main__':
    args = parser.parse_args()
    seed_torch(args.seed)
    args.epochs = args.max_epoch
    img_size = args.image_size
    results = dict()

    print('dataset:', args.dataset, 'split_idx:', args.split_idx)
    args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx)
    args.save_path = osp.join(args.save_path, 'split_{}/seed_{}'.format(args.split_idx, args.seed))
    args = init_experiment(args, args.save_path)

    # get_datasets receives args.split_idx, which determines what split to use
    # alternatively one can manually pass known_classes, open_set_classes as lists, that overrides evth.
    datasets = get_datasets(args.dataset, transform=args.transform, image_size=args.image_size, seed=args.seed, args=args)
    # # datasets is a dict with keys 'train', 'val', 'test_known', 'test_unknown', check them out:
    # print(type(datasets), list(datasets.keys()))


    # a = datasets['train']
    # item = a[0]
    # print(len(item)) # item contains two items, first one is (image, transform_idx), second one is label
    # data, label = item
    # image, transform_idx = data
    # print('item[0]=(image, transform_idx)', image.shape, transform_idx)
    # print('item[1]=label', label)
    # print(5*'-')
    # b = datasets['test_known']
    # item = b[0]
    # print(len(item))  # item contains two items, first one is (image_LIST, transform_idx_LIST), second one is label
    # data, label = item
    # image_list, transform_idx_list = item[0]
    # print('item[0]=(image_LIST, transform_idx_LIST)', len(image_list), len(transform_idx_list))
    # # transform_idx_LIST contains n_augs x2 -1, all transforms applied twice (+/-) unless identity.
    # print('item[1]=label', label)

    dataloaders = {}
    n_augs = datasets['train'].transforms.transform.n_augs
    for k, ds, in datasets.items():
        shuffle = True if k == 'train' else False
        batch_size = args.batch_size if k == 'train' else args.batch_size//n_augs
        batch_size = args.batch_size
        dataloaders[k] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=args.num_workers)
    # dataloaders is a dict with keys 'train', 'val', 'test_known', 'test_unknown', check it out:
    # print(dataloaders['train'].dataset.transforms.transform.n_augs)
    # # in training, dataloader returns [(batch_of_images, batch_of_transform_idxs), batch_of_labels]
    # (images_batch, transform_idxs_batch), label_batch = next(iter(dataloaders['train']))
    # print('TRAINING: batch=(images_batch, transform_idxs_batch), label_batch', images_batch.shape, transform_idxs_batch, label_batch)
    #
    # # in test, dataloader returns [(LIST_OF_batch_of_images, LIST_OF_batch_of_transform_idxs), LIST_OF_batch_of_labels]
    # (images_batch_list, transform_idxs_batch_list), label_batch = next(iter(dataloaders['test_known']))
    # print('TEST: batch=(LIST_OF_images_batch, LIST_OF_transform_idxs_batch), label_batch', len(images_batch_list), len(transform_idxs_batch_list), label_batch)
    # print('LEN of LIST_OF_images_batch/LIST_OF_transform_idxs_batch is 2xn_transforms -1 = {}'.format(2*n_augs-1))
    # print(('each item containing a batch of images and a batch of transform_idxs, batch_size//n_augs'))
    # for im, tr_idx in zip(images_batch_list, transform_idxs_batch_list):
    #     print(im.shape, tr_idx.shape)

    additional_classes = n_augs
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
