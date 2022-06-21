import os, os.path as osp
import torch
import random
import numpy as np

from datetime import datetime
import argparse

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


def seed_torch(seed=1029):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def strip_state_dict(state_dict, strip_key='module.'):

    """
    Strip 'module' from start of state_dict keys
    Useful if model has been trained as DataParallel model
    """

    for k in list(state_dict.keys()):
        if k.startswith(strip_key):
            state_dict[k[len(strip_key):]] = state_dict[k]
            del state_dict[k]

    return state_dict


def remove_empty_folders(root):
    walk = list(os.walk(root))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)
            if path.count('/')==1:
                with open(osp.join(root, 'deleted_dirs.txt'), 'a') as f:
                    print('Removed empty directory {}\n'.format(path), file=f)

def init_experiment(args, logs_dir):
    remove_empty_folders(logs_dir)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device # correct in case it was changed to 'cpu'
    print(20*'--', logs_dir)
    os.makedirs(logs_dir, exist_ok=True)
    # Instantiate directory to save model weights to
    model_root_dir = os.path.join(logs_dir, 'checkpoint')
    os.makedirs(model_root_dir, exist_ok=True)

    args.log_dir = logs_dir
    args.model_dir = model_root_dir

    print('Experiment log saved to: {}'.format(args.log_dir))
    print('Model weights saved to:  {}'.format(args.model_dir))


    return args

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


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class ClassificationPredSaver(object):

    def __init__(self, length, save_path=None):

        if save_path is not None:

            # Remove filetype from save_path
            save_path = save_path.split('.')[0]
            self.save_path = save_path

        self.length = length

        self.all_preds = None
        self.all_labels = None

        self.running_start_idx = 0

    def update(self, preds, labels=None):

        # Expect preds in shape B x C

        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()

        b, c = preds.shape

        if self.all_preds is None:
            self.all_preds = np.zeros((self.length, c))

        self.all_preds[self.running_start_idx: self.running_start_idx + b] = preds

        if labels is not None:
            if torch.is_tensor(labels):
                labels = labels.detach().cpu().numpy()

            if self.all_labels is None:
                self.all_labels = np.zeros((self.length,))

            self.all_labels[self.running_start_idx: self.running_start_idx + b] = labels

        # Maintain running index on dataset being evaluated
        self.running_start_idx += b

    def save(self):

        # Softmax over preds
        preds = torch.from_numpy(self.all_preds)
        preds = torch.nn.Softmax(dim=-1)(preds)
        self.all_preds = preds.numpy()

        pred_path = self.save_path + '.pth'
        print(f'Saving all predictions to {pred_path}')

        torch.save(self.all_preds, pred_path)

        if self.all_labels is not None:

            # Evaluate
            self.evaluate()
            torch.save(self.all_labels, self.save_path + '_labels.pth')

    def evaluate(self):

        topk = [1, 5, 10]
        topk = [k for k in topk if k < self.all_preds.shape[-1]]
        acc = accuracy(torch.from_numpy(self.all_preds), torch.from_numpy(self.all_labels), topk=topk)

        for k, a in zip(topk, acc):
            print(f'Top{k} Acc: {a.item()}')

# def get_acc_auroc_curves(logdir):
#
#     """
#     :param logdir: Path to logs: E.g '/work/sagar/open_set_recognition/methods/ARPL/log/(12.03.2021_|_32.570)/'
#     :return:
#     """
#
#     event_acc = EventAccumulator(logdir)
#     event_acc.Reload()
#
#     # Only gets scalars
#     log_info = {}
#     for tag in event_acc.Tags()['scalars']:
#
#         log_info[tag] = np.array([[x.step, x.value] for x in event_acc.scalars._buckets[tag].items])
#
#     return log_info
