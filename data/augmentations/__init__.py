from torchvision import transforms
from data.augmentations import aug_lib
from data.augmentations import t3po_augment

def get_transform(transform_type='trivial-augment_wide', image_size=32, args=None):

    if transform_type == 'trivial-augment':

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        '''
        From: https://github.com/automl/trivialaugment
        "Generally, a good position to augment an image with the augmenter is right as you get it out of the dataset,
        before you apply any custom augmentations." I'd say, however, that it would be best after resize, wouldn't it?
        '''
        augmenter = aug_lib.TrivialAugment()

        train_transform.transforms.insert(0, augmenter)

        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    elif transform_type == 'trivial-augment_wide':

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        '''
        From: https://github.com/automl/trivialaugment
        "Generally, a good position to augment an image with the augmenter is right as you get it out of the dataset,
        before you apply any custom augmentations." I'd say, however, that it would be best after resize, wouldn't it?
        '''
        aug_lib.set_augmentation_space('wide_standard', 31)
        augmenter = aug_lib.TrivialAugment()


        train_transform.transforms.insert(0, augmenter)

        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    elif 'T3PO' in transform_type:

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        num_magnitude_bins = 32

        pre_transforms_train = [
            transforms.Resize((image_size, image_size)),
        ]
        pre_transforms_test = [
            transforms.Resize((image_size, image_size))
        ]

        post_transforms_train = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        post_transforms_test = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        aug_space_name = transform_type[5:]

        if 'T3PO' in transform_type and 'wide' in transform_type:
            aug_space_name = 'wide'
        if 'T3PO' in transform_type and 'default' in transform_type:
            aug_space_name = 'default'

        if 'color' in transform_type:  # so we would use T3PO_color_default or T3PO_color_wide
            augClass = t3po_augment.Augment_T3PO_Color
        else:
            augClass = t3po_augment.Augment_T3PO

        print('---------', transform_type, augClass, aug_space_name, '---------')
        mode = 'return_transforms'
        train_transform = augClass(num_magnitude_bins, augmentation_space_name=aug_space_name, translate_mode='rel',
                                   mode=mode, fill='reflect', is_test=False, pre_transforms=pre_transforms_train,
                                   post_transforms=post_transforms_train)
        test_transform = augClass(num_magnitude_bins, augmentation_space_name=aug_space_name, translate_mode='rel',
                                  mode=mode, fill='reflect', is_test=True, pre_transforms=pre_transforms_test,
                                  post_transforms=post_transforms_test)

    else:

        raise NotImplementedError

    return (train_transform, test_transform)