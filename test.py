import numpy as np
import torch
from tqdm import tqdm
from utils import evaluation
from sklearn.metrics import average_precision_score, roc_auc_score
import sys
from data.open_set_datasets import get_class_splits, get_datasets
import argparse
from utils.utils import seed_torch, str2bool
from torch.utils.data import DataLoader
import os, os.path as osp
from sklearn.metrics import roc_auc_score


def test_model_t3po_single(net, testloader, args, outloader=None):
    # if outloader = None,  we are in training time, so we do not look at Open Set performance, only closed set accuracy
    # if outloader != None, we are testing the model, os we do look into Open Set performance

    use_softmax_in_eval = args.use_softmax_in_eval

    net.eval()
    correct_class, correct_transforms_k, total_k, correct_transforms_u, total_u = 0, 0, 0, 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_k_transforms, _pred_u, _pred_u_transforms, _labels_class_k = [], [], [], [], []
    _labels_transform_k, _labels_transform_u = [], []
    with torch.no_grad():
        for (images, labels_transforms), labels_class in tqdm(testloader):
            if torch.cuda.is_available():
                images, labels_transforms, labels_class = images.cuda(), labels_transforms.cuda(), labels_class.cuda()

            with torch.set_grad_enabled(False):
                logits_class, logits_transforms = net(images)
                preds_class = logits_class.max(dim=1)[1]
                preds_transforms = logits_transforms.max(dim=1)[1]

                correct_class += (preds_class == labels_class).sum()
                correct_transforms_k += (preds_transforms == labels_transforms).sum()
                total_k += labels_class.size(0)

                if use_softmax_in_eval:
                    logits_class = logits_class.softmax(dim=1)
                    logits_transforms = logits_transforms.softmax(dim=1)

                _pred_k.append(logits_class.cpu().numpy())
                _pred_k_transforms.append(logits_transforms.cpu().numpy())
                _labels_class_k.append(labels_class.cpu().numpy())
                _labels_transform_k.append(labels_transforms.cpu().numpy())


        acc_class = float(correct_class) * 100. / float(total_k)
        acc_transforms_k = float(correct_transforms_k) * 100. / float(total_k)
        _pred_k = np.concatenate(_pred_k, 0)
        _pred_k_transforms = np.concatenate(_pred_k_transforms, 0)
        _labels_class_k = np.concatenate(_labels_class_k, 0)
        _labels_transform_k = np.concatenate(_labels_transform_k, 0)

        if outloader is None:  # we have finished training and want test set metrics, in and out-of-distribution
            return acc_class

        for (images, labels_transform), _ in tqdm(outloader):
            if torch.cuda.is_available():
                images, labels_transform = images.cuda(), labels_transform.cuda()

            with torch.set_grad_enabled(False):
                logits_class, logits_transforms = net(images)
                preds_transforms = logits_transforms.max(dim=1)[1]

                correct_transforms_u += (preds_transforms == labels_transform).sum()
                total_u += labels_transform.size(0)
                if use_softmax_in_eval:
                    logits = logits.softmax(dim=1)
                    logits_transforms = logits_transforms.softmax(dim=1)

                _pred_u.append(logits_class.cpu().numpy())
                _pred_u_transforms.append(logits_transforms.cpu().numpy())
                _labels_transform_u.append(labels_transform.cpu().numpy())
    acc_transforms_u = float(correct_transforms_u) * 100. / float(total_u)
    _pred_u = np.concatenate(_pred_u, 0)
    _pred_u_transforms = np.concatenate(_pred_u_transforms, 0)
    _labels_transform_u = np.concatenate(_labels_transform_u, 0)

    # Out-of-Distribution detection evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2, verbose=False)['Bas']
    # print(_pred_k.shape, _pred_k_transforms.shape)
    # print(_pred_u.shape, _pred_u_transforms.shape)
    x11, x22 = np.max(_pred_k_transforms, axis=1), np.max(_pred_u_transforms, axis=1)
    auroc_transforms = evaluation.metric_ood(x11, x22, verbose=False)['Bas']
    # OSCR
    _oscr_score = evaluation.compute_oscr(_pred_k, _pred_u, _labels_class_k)

    # Average precision
    ap_score = average_precision_score([0] * len(_pred_k) + [1] * len(_pred_u),
                                       list(-np.max(_pred_k, axis=-1)) + list(-np.max(_pred_u, axis=-1)))
    print('acc tr K/U = {:.2f}/{:.2f}'.format(acc_transforms_k, acc_transforms_u))
    print('AUROC tr = {:.2f}'.format(auroc_transforms['AUROC']))
    results['ACC'] = acc_class
    results['ACC_Transf_k'] = acc_transforms_k
    results['ACC_Transf_u'] = acc_transforms_u
    results['OSCR'] = _oscr_score * 100.
    results['AUPR'] = ap_score * 100
    results['AUROC_TR'] = auroc_transforms['AUROC']

    return results


def test_model(net, testloader, args, outloader=None):
    # if outloader = None,  we are in training time, so we do not look at Open Set performance, only closed set accuracy
    # if outloader != None, we are testing the model, os we do look into Open Set performance

    use_softmax_in_eval=args.use_softmax_in_eval

    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for data, labels in tqdm(testloader):
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(False):
                logits = net(data)
                predictions = logits.max(dim=1)[1]
                total += labels.size(0)
                correct += (predictions == labels).sum()
                if use_softmax_in_eval:
                    logits = logits.softmax(dim=1)

                _pred_k.append(logits.cpu().numpy())
                _labels.append(labels.cpu().numpy())
        acc = float(correct) * 100. / float(total)
        _pred_k = np.concatenate(_pred_k, 0)
        _labels = np.concatenate(_labels, 0)

        if outloader is None: # we have finished training and want test set metrics, in and out-of-distribution
            return acc
        for data, _ in tqdm(outloader):
            if torch.cuda.is_available():
                data = data.cuda()

            with torch.set_grad_enabled(False):
                logits = net(data)
                if use_softmax_in_eval:
                    logits = logits.softmax(dim=1)

                _pred_u.append(logits.cpu().numpy())

    _pred_u = np.concatenate(_pred_u, 0)


    # Out-of-Distribution detection evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2, verbose=False)['Bas']
    
    # OSCR
    _oscr_score = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    # Average precision
    ap_score = average_precision_score([0] * len(_pred_k) + [1] * len(_pred_u),
                                       list(-np.max(_pred_k, axis=-1)) + list(-np.max(_pred_u, axis=-1)))

    results['ACC'] = acc
    results['OSCR'] = _oscr_score * 100.
    results['AUPR'] = ap_score * 100

    return results


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
def get_monte_carlo_predictions(loader, net, forward_passes=8):
    net.eval()
    enable_dropout(net)

    dropout_preds = []
    for _ in tqdm(range(forward_passes)):
        predictions, labels = [], []
        for _, (image, label) in enumerate(loader):
            image = image.to(torch.device('cuda'))
            output = net(image)
            output = output.softmax(dim=1)  # shape (n_samples, n_classes)
            predictions.append(output.cpu().numpy())
            labels.append(label.cpu().numpy())
        # print(len(loader), len(predictions))
        predictions = np.concatenate(predictions, 0)
        labels = np.concatenate(labels, 0)
        # print('predictions - len(data_loader) x n_classes', predictions.shape)
        # predictions - len(data_loader) x n_classes
        dropout_preds.append(predictions)

    # print(len(dropout_preds), dropout_preds[0].shape)
    dropout_preds = np.stack(dropout_preds, 0)
    # print('dropout_preds - forward_passes x len(data_loader) x n_classes', dropout_preds.shape)
    # dropout_preds - forward_passes x len(data_loader) x n_classes

    # Calculating mean across multiple MCD forward passes
    mean_probs = np.mean(dropout_preds, axis=0)  # shape (n_samples, n_classes)
    predictions = np.argmax(mean_probs, axis=1)

    # Calculating variance across multiple MCD forward passes
    variance = np.var(mean_probs, axis=0)  # shape n_samples
    epsilon = sys.float_info.min
    epsilon = 1e-8
    # Calculating entropy across multiple MCD forward passes
    entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=-1)  # shape n_samples

    return mean_probs, predictions, labels, variance, entropy

def test_dropout_model(net, testloader, args, outloader=None, forward_passes=8):
    # if outloader = None,  we are in training time, so we do not look at Open Set performance, only closed set accuracy
    # if outloader != None, we are testing the model, os we do look into Open Set performance

    use_softmax_in_eval = args.use_softmax_in_eval

    net.eval()
    torch.cuda.empty_cache()

    with torch.no_grad():
        _probs_k, _preds_k, _labels, variance_k, entropy_k = get_monte_carlo_predictions(testloader, net, forward_passes=forward_passes)
    acc = 100.*(_preds_k == _labels).sum() / len(_preds_k)

    if outloader is None:
        return acc

    with torch.no_grad():
        _probs_u, _preds_u, _labels, variance_u, entropy_u = get_monte_carlo_predictions(outloader, net, forward_passes=forward_passes)

    # print('closed set: mean entropy = {:.4f}'.format(np.mean(entropy_k)))
    # print('open set: mean entropy = {:.4f}'.format(np.mean(entropy_u)))

    labels_ood = np.concatenate([np.zeros(len(entropy_k)), np.ones(len(entropy_u))]) # greater entropy for ood
    probs_ood = np.concatenate([entropy_k, entropy_u])
    try:
        auroc_entropy = roc_auc_score(labels_ood, probs_ood)
    except: auroc_entropy = 0
    labels_ood = np.concatenate([np.ones(len(variance_k)), np.zeros(len(variance_u))]) # greater variance for iid
    probs_ood = np.concatenate([variance_k, variance_u])
    try:
        auroc_variance = roc_auc_score(labels_ood, probs_ood)
    except: auroc_variance = 0

    # Out-of-Distribution detection evaluation
    x1, x2 = np.max(_probs_k, axis=1), np.max(_probs_u, axis=1)

    results = evaluation.metric_ood(x1, x2, verbose=False)['Bas']

    # OSCR
    _oscr_score = evaluation.compute_oscr(_probs_k, _probs_u, _labels)

    # Average precision
    ap_score = average_precision_score([0] * len(_probs_k) + [1] * len(_probs_u),
                                       list(-np.max(_probs_k, axis=-1)) + list(-np.max(_probs_u, axis=-1)))

    results['ACC'] = acc
    results['OSCR'] = _oscr_score * 100.
    results['AUPR'] = ap_score * 100.
    results['AUROC_ENT'] = auroc_entropy * 100.
    results['AUROC_VAR'] = auroc_variance * 100.

    return results

parser = argparse.ArgumentParser("Training")
# Dataset
parser.add_argument('--dataset', type=str, default='cifar-10-10', help="")
parser.add_argument('--out-num', type=int, default=10, help='For cifar-10-100')
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--model', type=str, default='classifier32_aux_branch')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--transform', type=str, default='trivial-augment_wide') # needed for normalization!
parser.add_argument('--resnet50_pretrain', type=str, default='places_moco', help='Which pretraining to use if --model=timm_resnet50_pretrained.'
                             'Options are: {imagenet_moco, places_moco, places}', metavar='BOOL')
parser.add_argument('--split_train_val', default=False, type=str2bool, help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--device', default='cuda:0', type=str, help='device (cuda or cpu, default: cuda:0)')
parser.add_argument('--split_idx', default=-1, type=int, help='0-4 OSR splits for each dataset, -1 for fine-grained')
parser.add_argument('--use_softmax_in_eval', default=False, type=str2bool, help='Do we use softmax or logits for evaluation', metavar='BOOL')
parser.add_argument('--balance_open_set_eval', default=False, type=str2bool, help='Balanced open set', metavar='BOOL')
parser.add_argument('--load_path', type=str, default='')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_workers', default=0, type=int)
# parser.add_argument('--hpc', default=False, type=str2bool, help='testing at the HPC or locally', metavar='BOOL')
parser.add_argument('--verbose', default=False, type=str2bool, help='print stats to screen during training', metavar='BOOL')
parser.add_argument('--load_path_suffix', type=str, default='')

def load_network(network, result_dir, name='', loss=''):
    filename = '{}/checkpoints/{}_{}.pth'.format(result_dir, name, loss)
    network.load_state_dict(torch.load(filename))
    return network

def print_test_results(results, args, difficulty=''):

    acc_test, auroc_baseline, auroc_auxbranch, auroc_auxbranch_at_correct_mainbranch, auc_xent, auc_kl = results['ACC'], results['AUROC'],  \
                                results['AUC_auxbranch'], results['AUC_auxbranch_at_correct_mainbranch'], results['AUC_xent'], results['AUC_kl']

    oscr_baseline, oscr_auxbranch, oscr_auxbranch_at_correct_mainbranch, oscr_xent, oscr_kl = results['OSCR'], \
                                results['OSCR_auxbranch'], results['OSCR_auxbranch_at_correct_mainbranch'], results['OSCR_xent'], results['OSCR_kl']
    print(100 * "-")

    if args.split_idx == -1:  split_idx=difficulty # fine-grained, no split but different difficulties
    else: split_idx=args.split_idx
    print('dataset_{}{}'.format(args.dataset, args.load_path_suffix), 'split_{}'.format(split_idx), 'seed_{}'.format(args.seed))
    print('CLOSED Set Accuracy = {:.2f}'.format(acc_test))
    print('AUROCs: BLINE = {:.2f} - AUX-BRANCH BLINE = {:.2f} - AUX-AT-CORRECT-MAIN = {:.2f} - '
          'XENT = {:.2f} - KL = {:.2f}'.format(auroc_baseline, auroc_auxbranch, auroc_auxbranch_at_correct_mainbranch, auc_xent, auc_kl))
    print('OSCRs:  BLINE = {:.2f} - AUX-BRANCH BLINE = {:.2f} - AUX-AT-CORRECT-MAIN = {:.2f} - '
          'XENT = {:.2f} - KL = {:.2f}'.format(oscr_baseline, oscr_auxbranch, oscr_auxbranch_at_correct_mainbranch, oscr_xent, oscr_kl))


    with open(osp.join(args.log_dir, 'dataset_{}{}'.format(args.dataset, args.load_path_suffix), 'test_results.txt'), 'a') as f:
        print(100 * "-", file=f)
        print('dataset_{}{}'.format(args.dataset, args.load_path_suffix), 'split_{}'.format(split_idx), 'seed_{}'.format(args.seed), file=f)
        print('CLOSED Set Accuracy = {:.2f}'.format(acc_test), file=f)
        print('AUROCs: BLINE = {:.2f} - AUX-BRANCH BLINE = {:.2f} - AUX-AT-CORRECT-MAIN = {:.2f} - '
              'XENT = {:.2f} - KL = {:.2f}'.format(auroc_baseline, auroc_auxbranch, auroc_auxbranch_at_correct_mainbranch, auc_xent, auc_kl), file=f)
        print('OSCRs:  BLINE = {:.2f} - AUX-BRANCH BLINE = {:.2f} - AUX-AT-CORRECT-MAIN = {:.2f} - '
              'XENT = {:.2f} - KL = {:.2f}'.format(oscr_baseline, oscr_auxbranch, oscr_auxbranch_at_correct_mainbranch, oscr_xent, oscr_kl), file=f)


if __name__ == '__main__':
    '''
    python train_match_distributions.py --dataset cifar-10-10 --image_size 32 --split_idx=0
    python test.py --dataset cifar-10-10 --split_idx 0
    '''
    args = parser.parse_args()
    # if args.hpc:
    #     from config_hpc import logs_dir

    seed_torch(args.seed)
    img_size = args.image_size
    results = dict()

    args.feat_dim = 128 if args.model == 'classifier32' else 2048
    # print('dataset:',args.dataset, 'split_idx:', args.split_idx, 'out_num:', args.out_num)
    args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx, cifar_plus_n=args.out_num)

    img_size = args.image_size


    if args.split_idx >= 0:
        fine_grained=False
        load_path = osp.join(logs_dir, 'dataset_{}{}'.format(args.dataset, args.load_path_suffix), 'split_{}'.format(args.split_idx))
    else:
        fine_grained = True
        load_path = osp.join(logs_dir, 'dataset_{}{}'.format(args.dataset, args.load_path_suffix), 'seed_{}'.format(args.seed))
    vars(args).update({'log_dir': logs_dir})

    # from utils.utils import seed_torch, str2bool, init_experiment
    # args = init_experiment(args, logs_dir, args.save_path)

    checkpoint_path = os.listdir(osp.join(load_path, 'checkpoint'))
    if len(checkpoint_path)>1: sys.exit('More than one checkpoint at {}'.format(load_path))
    checkpoint_path = osp.join(load_path, 'checkpoint', checkpoint_path[0])


    if args.model == 'timm_resnet50_pretrained':
        wrapper_class = TimmResNetWrapper
    else:
        wrapper_class = None


    net = get_model(args, wrapper_class=wrapper_class, additional_classes=len(args.train_classes))
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Currently using GPU: {}".format(args.device))
        net.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(args.device)))
        net = net.cuda()
    else:
        print("Currently using CPU")
        net.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

    ####################################################################################################################
    if fine_grained:
        import pickle
        from config import osr_split_dir
        osr_path = os.path.join(osr_split_dir, '{}_osr_splits.pkl'.format(args.dataset))
        with open(osr_path, 'rb') as f:
            class_info = pickle.load(f)
        open_set_diff_classes = class_info['unknown_classes']

        for difficulty in ('Easy', 'Medium', 'Hard'):
            datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                                    image_size=args.image_size, balance_open_set_eval=False,
                                    split_train_val=False, open_set_classes=open_set_diff_classes[difficulty])
            dataloaders = {}
            for k, ds, in datasets.items():
                shuffle = True if k == 'train' else False
                dataloaders[k] = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                                            num_workers=args.num_workers)
            test_loader, out_loader = dataloaders['test_known'], dataloaders['test_unknown']
            ####################################################################################################################
            results = test_model_distribution_match(net, test_loader, out_loader, args.use_softmax_in_eval,
                                                    is_train=False, log_dir=args.log_dir, verbose=args.verbose)
            print_test_results(results, args, difficulty=difficulty)
    else:
        datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                                open_set_classes=args.open_set_classes, balance_open_set_eval=args.balance_open_set_eval,
                                split_train_val=args.split_train_val, image_size=args.image_size, seed=args.seed, args=args)

        dataloaders = {}
        for k, ds, in datasets.items():
            shuffle = True if k == 'train' else False
            dataloaders[k] = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)

        test_loader, out_loader = dataloaders['test_known'], dataloaders['test_unknown']
        ####################################################################################################################

        results = test_model_distribution_match(net, test_loader, out_loader, args.use_softmax_in_eval, is_train=False,
                                                log_dir=args.log_dir, verbose=args.verbose)

        print_test_results(results, args)