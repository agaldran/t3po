from data.kather2016 import get_kather2016_datasets
from data.kather100k import get_kather100k_datasets
from data.open_set_splits.osr_splits import osr_splits
from data.augmentations import get_transform
# from config import osr_split_dir

import os
import sys
import pickle
import torch

"""
For each dataset, define function which returns:
    training set
    validation set
    open_set_known_images
    open_set_unknown_images
"""

get_dataset_funcs = {
    'kather2016': get_kather2016_datasets,
    'kather100k': get_kather100k_datasets,
}

def get_datasets(name, transform='default', image_size=150, seed=0, args=None, known_classes=None, open_set_classes=None):

    """
    :param name: Dataset name
    :param transform: Either tuple of train/test transforms or string of transform type
    :return:
    """

    print('Loading dataset {}'.format(name))

    if isinstance(transform, tuple):
        train_transform, test_transform = transform
    else:
        train_transform, test_transform = get_transform(transform_type=transform, image_size=image_size, args=args)

    if known_classes is None and open_set_classes is None:
        known_classes, open_set_classes = get_class_splits(name, args.split_idx)

    if name in get_dataset_funcs.keys():
        datasets = get_dataset_funcs[name](train_transform, test_transform,
                                           known_classes=known_classes,
                                           open_set_classes=open_set_classes,
                                           seed=seed)
    else:
        raise NotImplementedError

    return datasets

def get_class_splits(dataset, split_idx=0):

    if dataset == 'kather2016':

        known_classes = osr_splits[dataset][split_idx]
        open_set_classes = [x for x in range(8) if x not in known_classes]
        print('training on known classes:', known_classes)
        print('open set classes:', open_set_classes)

    elif dataset == 'kather100k': # this one has nine classes
        known_classes = osr_splits[dataset][split_idx]
        open_set_classes = [x for x in range(9) if x not in known_classes]
        print('training on known classes:', known_classes)
        print('open set classes:', open_set_classes)

    else:

        raise NotImplementedError

    return known_classes, open_set_classes
#
# # Disable
# def blockPrint():
#     sys.stdout = open(os.devnull, 'w')
#
# # Restore
# def enablePrint():
#     sys.stdout = sys.__stdout__