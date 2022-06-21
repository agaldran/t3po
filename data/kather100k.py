import torchvision
import numpy as np
import shutil
import random, os, os.path as osp
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from config import kather100k_root
import argparse

def create_val_img_folder(path_data_in='data/NCT-CRC-HE-100K'):
    '''
    This method is responsible for separating kather100kk images into separate sub folders
    path_data_in points to a folder containing the 9 subfolders resulting from downloading and uncompressing:
    https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip
    '''
    path_data = path_data_in.replace('NCT-CRC-HE-100K', 'kather100k')
    path_train_data = osp.join(path_data, 'train')
    path_val_data = osp.join(path_data, 'val')
    path_test_data = osp.join(path_data, 'test')

    os.makedirs(osp.join(path_train_data), exist_ok=True)
    os.makedirs(osp.join(path_val_data), exist_ok=True)
    os.makedirs(osp.join(path_test_data), exist_ok=True)

    subfs = os.listdir(path_data_in)
    # loop over subfolders in train data folder
    for f in subfs:
        os.makedirs(osp.join(path_train_data, f), exist_ok=True)
        os.makedirs(osp.join(path_val_data, f), exist_ok=True)
        os.makedirs(osp.join(path_test_data, f), exist_ok=True)

        path_this_f = osp.join(path_data_in, f)
        print(path_this_f)
        im_list = os.listdir(path_this_f)

        train_len = int(0.70 * len(im_list))  # random 70% for train
        val_len = int(0.15 * len(im_list))  # random 15% for val
        test_len = int(0.15 * len(im_list))  # random 15% for test

        # take 15% out of im_list for test
        im_list_test = random.sample(im_list, test_len)
        im_list_train = list(set(im_list) - set(im_list_test))

        # take 15% out of im_list for val, the rest is train
        im_list_val = random.sample(im_list_train, val_len)
        im_list_train = list(set(im_list_train) - set(im_list_val))

        # loop over subfolders and move train/val/test images over to corresponding subfolders
        for im in im_list_train:
            shutil.move(osp.join(path_data_in, f, im), osp.join(path_train_data, f, im))
        for im in im_list_val:
            shutil.move(osp.join(path_data_in, f, im), osp.join(path_val_data, f, im))
        for im in im_list_test:
            shutil.move(osp.join(path_data_in, f, im), osp.join(path_test_data, f, im))

class Kather100k(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform):

        super(Kather100k, self).__init__(root, transform)
        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]
        return img, label

def subsample_dataset(dataset, idxs):
    # dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in idxs]
    # dataset.samples = [x for i, x in enumerate(dataset.samples) if i in idxs]
    # I am replacing the above awful code by a faster version, no need to enumerate for each idx the same set again and
    # again, people should be more careful with comprehensions (agaldran).
    imgs, sampls = [], []
    for i in idxs:
        imgs.append(dataset.imgs[i])
        sampls.append(dataset.samples[i])
    dataset.imgs = imgs
    dataset.samples = sampls
    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset

def subsample_classes(dataset, include_classes=(0,1)):
    _ = enumerate(dataset.targets)
    cls_idxs = [x for x, t in _ if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]

    # torchvision ImageFolder dataset have a handy class_to_idx attribute that we have spoiled and need to re-do
    # filter class_to_idx to keep only include_classes
    new_class_to_idx = {key: val for key, val in dataset.class_to_idx.items() if val in target_xform_dict.keys()}
    # fix targets so that they start in 0 and are correlative
    new_class_to_idx = {k: target_xform_dict[v] for k, v in new_class_to_idx.items()}
    # replace bad class_to_idx with good one
    dataset.class_to_idx = new_class_to_idx

    # and let us also add a idx_to_class attribute
    dataset.idx_to_class = dict((v, k) for k, v in new_class_to_idx.items())

    return dataset

def get_equal_len_datasets(dataset1, dataset2):
    """
    Make two datasets the same length
    """

    if len(dataset1) > len(dataset2):

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2, )))
        subsample_dataset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1, )))
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2

def get_kather100k_datasets(train_transform, test_transform, known_classes=(0,1,2,3,4,5), open_set_classes=(6,7,8), seed=0):

    kather_train_dir = osp.join(kather100k_root, 'train')
    kather_val_dir = osp.join(kather100k_root, 'val')
    kather_test_dir = osp.join(kather100k_root, 'test')

    np.random.seed(seed)
    # Build train/val dataset and subsample known_classes
    train_dataset = Kather100k(root=kather_train_dir, transform=train_transform)
    train_dataset = subsample_classes(train_dataset, include_classes=known_classes)

    val_dataset = Kather100k(root=kather_val_dir, transform=test_transform)
    val_dataset = subsample_classes(val_dataset, include_classes=known_classes)

    # Build test dataset and subsample known/unknown classes
    test_dataset_known = Kather100k(root=kather_test_dir, transform=test_transform)
    test_dataset_known = subsample_classes(test_dataset_known, include_classes=known_classes)

    # Build test dataset and subsample known/unknown classes
    test_dataset_unknown = Kather100k(root=kather_test_dir, transform=test_transform)
    test_dataset_unknown = subsample_classes(test_dataset_unknown, include_classes=open_set_classes)

    balance_open_set_eval=False
    if balance_open_set_eval:
        test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)
        test_dataset_known, val_dataset = get_equal_len_datasets(test_dataset_known, val_dataset)

    all_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Training")
    parser.add_argument('--path_data_in', type=str, default='data/NCT-CRC-HE-100K', help="")

    args = parser.parse_args()
    create_val_img_folder(args.path_data_in)
