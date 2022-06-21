from PIL import Image
from skimage import transform
import numpy as np
import math, random
from typing import List, Tuple, Optional, Dict
import sys
import torch
from torch import Tensor
from torchvision.transforms import functional_pil as F_pil

pil_sk_modes_mapping = {
    Image.NEAREST: 0,
    Image.BILINEAR: 1,
    Image.BICUBIC: 3,
}

def shear_x_impl(pil_img: Image.Image, magnitude: float,
                 interpolation: int = Image.NEAREST,
                 fill: str = 'constant',) -> Image.Image:
    # pil and skimage.transform.warp expect shear angle in counter-clockwise direction as radians
    # to be consistent with rotation below, this function expects angle as clockwise degrees
    # As a reference, 0.3 radians is the "default" value in Trivial Augment, and 0.99 the "wide" value
    # magnitude = x-shear angle in counter-clockwise direction as radians
    # interpolation is of type int
    # fill is of type str, can be 'constant', 'edge', 'symmetric', 'reflect'
    # if 'constant', pil transform is used with fill=0
    # for all other cases, (slower) skimage implementation is used
    magnitude = math.radians(magnitude)
    if fill=='constant':
        return pil_img.transform(pil_img.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0))
    interpolation = pil_sk_modes_mapping[interpolation]
    tf_mat = transform.AffineTransform(np.array([[1, magnitude, 0], [0, 1, 0], [0, 0, 1]]))
    return Image.fromarray(transform.warp(np.array(pil_img), inverse_map=tf_mat, mode=fill,
                            order=interpolation, preserve_range=True).astype(np.uint8))


def shear_y_impl(pil_img: Image.Image, magnitude: float,
                 interpolation: int = Image.NEAREST,
                 fill: str = 'constant',) -> Image.Image:
    # pil and skimage.transform.warp expect shear angle in counter-clockwise direction as radians
    # to be consistent with rotation below, this function expects angle as clockwise degrees
    # As a reference, 0.3 radians is the "default" value in Trivial Augment, and 0.99 the "wide" value
    # magnitude = y-shear angle in counter-clockwise direction as radians
    # interpolation is of type int
    # fill is of type str, can be 'constant', 'edge', 'symmetric', 'reflect'
    # if 'constant', pil transform is used with fill=0
    # for all other cases, (slower) skimage implementation is used
    magnitude = math.radians(magnitude)
    if fill=='constant':
        return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0))
    interpolation = pil_sk_modes_mapping[interpolation]
    tf_mat = transform.AffineTransform(np.array([[1, 0, 0], [magnitude, 1, 0], [0, 0, 1]]))
    return Image.fromarray(transform.warp(np.array(pil_img), inverse_map=tf_mat, mode=fill,
                            order=interpolation, preserve_range=True).astype(np.uint8))

def shear_impl(pil_img: Image.Image, magnitude: float,
                 interpolation: int = Image.NEAREST,
                 fill: str = 'constant',) -> Image.Image:
    if random.random() > 0.5:
        return shear_x_impl(pil_img, magnitude, interpolation, fill)
    return shear_y_impl(pil_img, magnitude, interpolation, fill)

def translate_x_impl(pil_img: Image.Image, magnitude: float,
                     interpolation: int = Image.NEAREST,
                     fill: str = 'constant', translate_mode: str = 'abs') -> Image.Image:
    # magnitude = horizontally translated pixels(int)
    # interpolation is of type int
    # fill is of type str, can be 'constant', 'edge', 'symmetric', 'reflect'
    # if 'constant', pil transform is used with fill=0
    # for all other cases, (slower) skimage implementation is used

    # if mode=='rel', translates pixels = percentage of width (cols) in the x direction
    # therefore, we expect magnitude to be in [0,1] in that case
    # Note that I invert magnitude because I like it better when positive means towards the right side
    magnitude = -magnitude

    if translate_mode == 'rel':
        assert -1 <= magnitude <= 1
        cols, rows = pil_img.size
        magnitude = int(magnitude * cols)
    if fill == 'constant':
        return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, magnitude, 0, 1, 0))
    interpolation = pil_sk_modes_mapping[interpolation]
    # this below is slower but enables boundary reflection
    pil_img = F_pil.pad(pil_img, padding=(0, 0, int(magnitude), 0), padding_mode=fill)
    return F_pil.pad(pil_img, padding=(int(-magnitude), 0, 0, 0), padding_mode=fill)



def translate_y_impl(pil_img: Image.Image, magnitude: float,
                     interpolation: int = Image.NEAREST,
                     fill: str = 'constant', translate_mode: str = 'abs') -> Image.Image:
    # magnitude = vertically translated pixels(int)
    # interpolation is of type int
    # fill is of type str, can be 'constant', 'edge', 'symmetric', 'reflect'
    # if 'constant', pil transform is used with fill=0
    # for all other cases, (slower) skimage implementation is used

    # if mode=='rel', translates pixels = percentage of width (cols) in the x direction
    # therefore, we expeted magnitude to be in [0,1]
    if translate_mode == 'rel':
        assert -1 <= magnitude <= 1
        cols, rows = pil_img.size
        magnitude = int(magnitude * rows)

    if fill == 'constant':
        return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude))
    interpolation = pil_sk_modes_mapping[interpolation]
    # this below is slower but enables boundary reflection
    pil_img = F_pil.pad(pil_img, padding=(0, 0, 0, int(magnitude)), padding_mode=fill)
    return F_pil.pad(pil_img, padding=(0, -int(magnitude), 0, 0), padding_mode=fill)

def translate_impl(pil_img: Image.Image, magnitude: float,
                     interpolation: int = Image.NEAREST,
                     fill: str = 'constant', translate_mode: str='abs') -> Image.Image:
    if random.random() > 0.5:
        return translate_x_impl(pil_img, magnitude, interpolation, fill, translate_mode)
    return translate_y_impl(pil_img, magnitude, interpolation, fill, translate_mode)

def rotate_impl(pil_img: Image.Image, magnitude: float,
                interpolation: int = Image.NEAREST,
                 fill: str = 'constant') -> Image.Image:
    # magnitude = rotation angle in degrees in counter-clockwise direction
    # interpolation is of type int
    # fill is of type str, can be 'constant', 'edge', 'symmetric', 'reflect'
    # if 'constant', pil transform is used with fill=0
    # for all other cases, (slower) skimage implementation is used
    if fill=='constant':
        return pil_img.rotate(magnitude)
    interpolation = pil_sk_modes_mapping[interpolation]
    return Image.fromarray(transform.rotate(np.array(pil_img),magnitude, mode=fill,
                            order=interpolation, preserve_range=True).astype(np.uint8))


def contrast_impl(pil_img: Image.Image, magnitude: float) -> Image.Image:
    # magnitude = contrast_factor (float): How much to adjust the contrast.
    # Can be any non negative number. 0 gives a solid gray image, 1 gives the
    # original image while 2 increases the contrast by a factor of 2.

    return F_pil.adjust_contrast(pil_img, magnitude)


def brightness_impl(pil_img: Image.Image, magnitude: float) -> Image.Image:
    # magnitude = brightness_factor (float): How much to adjust the brightness.
    # Can be any non negative number. 0 gives a black image, 1 gives the
    # original image while 2 increases the brightness by a factor of 2

    return F_pil.adjust_brightness(pil_img, magnitude)


def saturation_impl(pil_img: Image.Image, magnitude: float) -> Image.Image:
    # magnitude = saturation_factor (float): How much to adjust the saturation.
    # Can be any non negative number. 0 gives a black and white image, 1 gives the
    # original image while 2 increases the saturation by a factor of 2

    return F_pil.adjust_saturation(pil_img, magnitude)


def hue_impl(pil_img: Image.Image, magnitude: float) -> Image.Image:
    # magnitude = saturation_factor (float): How much to adjust the hue by shifting the hue channel.
    # Must be in[-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in HSV space in
    # positive and negative direction respectively. 0 means no shift. Therefore, both -0.5 and 0.5
    # will give an image with complementary colors while 0 gives the original image.
    # This in practice is visually meaningless. Using magnitude \in [-0.05, 0.05] is more realistic

    return F_pil.adjust_hue(pil_img, magnitude)


def gamma_impl(pil_img: Image.Image, magnitude: float) -> Image.Image:
    # magnitude = gamma correction (float): How much to adjust the gamma.
    # Can be any non negative number. gamma larger than 1 make the shadows darker,
    # while gamma smaller than 1 make dark regions lighter.
    # In practice, a value in between [0.5, 2] is realistic

    return F_pil.adjust_gamma(pil_img, magnitude)


def sharpness_impl(pil_img: Image.Image, magnitude: float) -> Image.Image:
    # magnitude = sharpness_factor (float): How much to adjust the sharpness.
    # Can be any non negative number. 0 gives blurred image, 1 gives the
    # original image while 2 increases the sharpness by a factor of 2
    # In practice, a value in between [0.5, 2] is realistic

    return F_pil.adjust_sharpness(pil_img, magnitude)


# note: these transforms fall back to default pil if no boundary reflection is asked for,
#       but use more expensive versions if user asks for reflected boundary (using the `mode` parameter)
# note2: need to make conversions: op_name <-> op_id, and return it if asked for

def _apply_op(img: Image.Image, op_name: str, magnitude: float,
              interpolation: int, fill: str, translate_mode: str) -> Image.Image:
    if op_name == "ShearX":
        # no need to convert using math.degrees, since skimage.transform.warp expects radians too
        img = shear_x_impl(img, magnitude, interpolation, fill)
    elif op_name == "ShearY":
        # no need to convert using math.degrees, since skimage.transform.warp expects radians too
        img = shear_y_impl(img, magnitude, interpolation, fill)
    elif op_name == "Shear":
        img = shear_impl(img, magnitude, interpolation, fill)

    elif op_name == "TranslateX":
        # slightly optimized impl using pil instead of skimage, still allowing mode=reflect
        img = translate_x_impl(img, magnitude, interpolation, fill, translate_mode)
    elif op_name == "TranslateY":
        img = translate_y_impl(img, magnitude, interpolation, fill, translate_mode)
    elif op_name == "Translate":
        img = translate_impl(img, magnitude, interpolation, fill, translate_mode)

    elif op_name == "Rotate":
        # no need to convert using math.radians, since skimage.transform.rotate expects degrees
        img = rotate_impl(img, magnitude, interpolation, fill)

    elif op_name == "Brightness":
        img = brightness_impl(img, magnitude)
    elif op_name == "Contrast":
        img = contrast_impl(img, magnitude)
    elif op_name == "Saturation":
        img = saturation_impl(img, magnitude)
    elif op_name == "Hue":
        img = hue_impl(img, magnitude)
    elif op_name == "Gamma":
        img = gamma_impl(img, magnitude)
    elif op_name == "Sharpness":
        img = sharpness_impl(img, magnitude)

    elif op_name == "Identity":
        pass
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))

    return img


class Augment_T3PO_old(torch.nn.Module):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (FillMode): Desired filling scheme enum defined above in
            :class:`FillMode`. Default is ``FillMode.CONSTANT``. If FillMode.REFLECT is used,
            most geometric transforms resort to skimage, and are more expensive
        """

    def __init__(self, num_magnitude_bins: int = 32, augmentation_space_name: str = 'default',
                 mode='return_transforms', is_test: bool = False, fast_test: bool = False,
                 interpolation: int = Image.NEAREST, fill: str = 'constant', translate_mode: str = 'abs',
                 post_transforms: List = ()) -> None:
        super().__init__()
        self.interpolation = interpolation
        self.fill = fill
        self.translate_mode = translate_mode
        self.num_magnitude_bins = num_magnitude_bins
        self.mode = mode
        if augmentation_space_name == 'default':
            self.augmentation_space = self._default_augmentation_space()
        elif augmentation_space_name == 'wide':
            self.augmentation_space = self._wide_augmentation_space()
        elif augmentation_space_name == 'debug':
            self.augmentation_space = self._debug_augmentation_space()
        elif augmentation_space_name == 'debug_wide':
            self.augmentation_space = self._debug_wide_augmentation_space()

        elif augmentation_space_name == 'reduced':
            self.augmentation_space = self._reduced_augmentation_space()

        else:
            sys.exit('not a valid augmentation space')

        self.n_augs = len(list(self.augmentation_space))
        self.post_transforms = post_transforms
        self.is_test = is_test
        self.fast_test = fast_test

    def _default_augmentation_space(self, ) -> Dict[str, Tuple[Tensor, bool]]:
        max_translate = 10.0
        if self.translate_mode == 'rel': max_translate = 0.25
        shear_angle = math.degrees(0.30)
        return {
            # op_name: magnitudes, neutral_magnitude
            "Identity": (torch.zeros(self.num_magnitude_bins), 0.0),
            "ShearX": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            "ShearY": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            "TranslateX": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            "TranslateY": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            "Rotate": (torch.linspace(-30.0, 30.0, self.num_magnitude_bins), 0.0),
            "Brightness": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
            "Contrast": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
            "Saturation": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
            "Hue": (torch.linspace(-0.05, 0.05, self.num_magnitude_bins), 0.0),
            "Gamma": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
            "Sharpness": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
        }

    def _wide_augmentation_space(self, ) -> Dict[str, Tuple[Tensor, bool]]:
        max_translate = 32.0
        if self.translate_mode == 'rel': max_translate = 0.50
        shear_angle = math.degrees(0.99)
        return {
            # op_name: magnitudes, neutral_magnitude
            "Identity": (torch.zeros(self.num_magnitude_bins), 0.0),
            "ShearX": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            "ShearY": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            "TranslateX": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            "TranslateY": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            "Rotate": (torch.linspace(-135.0, 135.0, self.num_magnitude_bins), 0.0),
            "Brightness": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
            "Contrast": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
            "Saturation": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
            "Hue": (torch.linspace(-0.10, 0.10, self.num_magnitude_bins), 0.0),
            "Gamma": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
            "Sharpness": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
        }

    def _debug_augmentation_space(self, ) -> Dict[str, Tuple[Tensor, bool]]:
        max_translate = 10.0
        if self.translate_mode == 'rel': max_translate = 0.25
        shear_angle = math.degrees(0.30)
        return {
            # op_name: magnitudes, neutral_magnitude
            "Identity": (torch.zeros(self.num_magnitude_bins), 0.0),
            # "ShearX": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            # "ShearY": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            "Shear": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            # "TranslateX": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            # "TranslateY": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            "Translate": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            "Rotate": (torch.linspace(-30.0, 30.0, self.num_magnitude_bins), 0.0),
            "Brightness": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
#             "Contrast": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
            "Saturation": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
            "Hue": (torch.linspace(-0.05, 0.05, self.num_magnitude_bins), 0.0),
#             "Gamma": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
            "Sharpness": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
        }

    def _debug_wide_augmentation_space(self, ) -> Dict[str, Tuple[Tensor, bool]]:
        max_translate = 32.0
        if self.translate_mode == 'rel': max_translate = 0.50
        shear_angle = math.degrees(0.99)
        return {
            # op_name: magnitudes, neutral_magnitude
            "Identity": (torch.zeros(self.num_magnitude_bins), 0.0),
            # "ShearX": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            # "ShearY": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            "Shear": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            # "TranslateX": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            # "TranslateY": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            "Translate": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            "Rotate": (torch.linspace(-135.0, 135.0, self.num_magnitude_bins), 0.0),
            "Brightness": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
            # "Contrast": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
            "Saturation": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
            "Hue": (torch.linspace(-0.10, 0.10, self.num_magnitude_bins), 0.0),
            # "Gamma": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
            "Sharpness": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
        }

#     def _reduced_augmentation_space(self, ) -> Dict[str, Tuple[Tensor, bool]]:
#         max_translate = 10.0
#         if self.translate_mode == 'rel': max_translate = 0.25
#         shear_angle = math.degrees(0.30)
#         return {
#             # op_name: magnitudes, neutral_magnitude
#             "Identity": (torch.zeros(self.num_magnitude_bins), 0.0),
#             "Shear": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
#             "Translate": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
#             "Rotate": (torch.linspace(-30.0, 30.0, self.num_magnitude_bins), 0.0),
#             "Brightness": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
#             "Contrast": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
#             "Hue": (torch.linspace(-0.05, 0.05, self.num_magnitude_bins), 0.0),
#             "Sharpness": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
#         }


    def forward(self, img: Image.Image) -> Tuple[Image.Image, int, int, int]:
        """
            img (PIL Image): Image to be transformed.

            # if not self.is_test, it will return a transformed version of img using a randomly chosen transform
            # from augmentation_space, together with the corresponding operation index, magnitude index, and sign.

            # if self.is_test, it will return a list of images including the original one plus versions
            # transformed by using all transforms in augmentation_space, with midway negative/positive magnitude
            # it will also return the corresponding operation indexes, magnitude indexes, and signs
            # this amounts to returning four lists of 2*len(augmentation_space)-1, one containing images and the
            # other three containing corresponding integers.
        """

        aug_keys = list(self.augmentation_space.keys())
        if self.fast_test: # we only return identity
            n_mags = self.num_magnitude_bins
            for t in self.post_transforms:
                img = t(img)
            op_index, op_magnitude_index, op_sign = 0, n_mags // 2, 0
            if self.mode == 'return_transforms':
                return img, op_index
            elif self.mode == 'return_signs':
                return img, 0
            return img, op_index, op_magnitude_index, op_sign

        elif self.is_test:

            n_mags = self.num_magnitude_bins
            # no need to return more than one copy of identity(img)
            aug_keys.remove('Identity')
            # but we do need to apply post_transforms to it
            tr_img = img.copy()
            for t in self.post_transforms:
                tr_img = t(tr_img)

            tr_imgs, op_indexes, op_magnitude_indexes, op_signs = [tr_img], [0], [n_mags // 2], [0]

            for op_index in range(len(aug_keys)):
                op_name = aug_keys[op_index]
                magnitudes, neutral_magnitude = self.augmentation_space[op_name]
                magnitudes = np.array(magnitudes)
                mid_neg_mag_idx, mid_pos_mag_idx = int(0.25 * n_mags), int(0.75 * n_mags)
                mid_neg_magnitude = magnitudes[mid_neg_mag_idx]
                mid_pos_magnitude = magnitudes[mid_pos_mag_idx]

                # transform image with mid_neg_magnitude
                tr_img = _apply_op(img, op_name, mid_neg_magnitude, interpolation=self.interpolation,
                                   fill=self.fill, translate_mode=self.translate_mode)
                # add post_transforms
                for t in self.post_transforms:
                    tr_img = t(tr_img)
                tr_imgs.append(tr_img)

                op_indexes.append(op_index + 1)
                op_magnitude_indexes.append(mid_neg_mag_idx)
#                 op_signs.append(0)  # because its negative magnitude
                op_sign = 0 if op_name in ["Identity","Shear", "ShearX", "ShearY",
                                           "Translate", "TranslateX","TranslateY","Rotate"] else 1
                op_signs.append(op_sign)  # because its a transformed image

                # transform image with mid_pos_magnitude
                tr_img = _apply_op(img, op_name, mid_pos_magnitude, interpolation=self.interpolation,
                                   fill=self.fill, translate_mode=self.translate_mode)
                # add post_transforms
                for t in self.post_transforms:
                    tr_img = t(tr_img)
                tr_imgs.append(tr_img)

                op_indexes.append(op_index + 1)  # because we removed identity in the first place
                op_magnitude_indexes.append(mid_pos_mag_idx)
#                 op_signs.append(1)  # because its positive magnitude
                op_sign = 0 if op_name in ["Identity","Shear", "ShearX", "ShearY",
                                           "Translate", "TranslateX","TranslateY","Rotate"] else 1
                op_signs.append(op_sign)  # because its a transformed image
            if self.mode == 'return_transforms':
                return tr_imgs, op_indexes
            elif self.mode == 'return_signs':
                return tr_imgs, op_signs
            return tr_imgs, op_indexes, op_magnitude_indexes, op_signs
        else:
            # pick an image operation at random, keep index
            op_index = int(torch.randint(len(self.augmentation_space), (1,)).item())
            op_name = aug_keys[op_index]
            # pick an operation strength at random, keep index
            magnitudes, neutral_magnitude = self.augmentation_space[op_name]
            op_magnitude_index = int(torch.randint(len(magnitudes), (1,), dtype=torch.long))
            magnitude = float(magnitudes[op_magnitude_index])
#             # check if magnitude is above neutral value for this op
#             op_sign = int(magnitude >= neutral_magnitude)

            # check if operation is identity (sign=0) or not (sign=1)
            op_sign = 0 if op_name in ["Identity", "Shear", "ShearX", "ShearY",
                                       "Translate", "TranslateX", "TranslateY", "Rotate"] else 1
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=self.fill,
                            translate_mode=self.translate_mode)
            # add post_transforms
            for t in self.post_transforms:
                img = t(img)
            if self.mode == 'return_transforms':
                return img, op_index
            elif self.mode == 'return_signs':
                return img, op_sign
            return img, op_index, op_magnitude_index, op_sign

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'num_magnitude_bins={num_magnitude_bins}'
        s += ', interpolation={interpolation}'
        s += ', fill={fill}'
        s += ')'
        return s.format(**self.__dict__)

class Augment_T3PO_Color(torch.nn.Module):
    def __init__(self, num_magnitude_bins: int = 5, augmentation_space_name: str = 'default',
                 mode='return_all', is_test: bool = False, fast_test: bool = False,
                 interpolation: int = Image.NEAREST, fill: str = 'constant', translate_mode: str = 'abs',
                 pre_transforms: List = (), post_transforms: List = ()) -> None:
        super().__init__()
        self.interpolation = interpolation
        self.fill = fill
        self.translate_mode = translate_mode
        self.num_magnitude_bins = num_magnitude_bins
        self.mode = mode
        if augmentation_space_name == 'default':
            self.geometry_augmentation_space = self._geometry_default_augmentation_space()
            self.color_augmentation_space = self._color_default_augmentation_space()
        elif augmentation_space_name == 'wide':
            self.geometry_augmentation_space = self._geometry_wide_augmentation_space()
            self.color_augmentation_space = self._color_wide_augmentation_space()
        elif augmentation_space_name == 'mixed':
            self.geometry_augmentation_space = self._geometry_default_augmentation_space()
            self.color_augmentation_space = self._color_wide_augmentation_space()

        else:
            sys.exit('not a valid augmentation space')
        self.augmentation_space = self.color_augmentation_space # for compatiblity later
        self.n_augs = len(list(self.color_augmentation_space))
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms
        self.is_test = is_test
        self.fast_test = fast_test

    def _color_default_augmentation_space(self, ) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: magnitudes, neutral_magnitude
            "Identity": (torch.zeros(self.num_magnitude_bins), 0.0),
            "Brightness": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
            "Contrast": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
            "Saturation": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
            "Hue": (torch.linspace(-0.05, 0.05, self.num_magnitude_bins), 0.0),
            "Gamma": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
            "Sharpness": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
        }

    def _color_wide_augmentation_space(self, ) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: magnitudes, neutral_magnitude
            "Identity": (torch.zeros(self.num_magnitude_bins), 0.0),
            "Brightness": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
            "Contrast": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
            "Saturation": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
            "Hue": (torch.linspace(-0.10, 0.10, self.num_magnitude_bins), 0.0),
            "Gamma": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
            "Sharpness": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
        }

    def _geometry_default_augmentation_space(self, ) -> Dict[str, Tuple[Tensor, bool]]:
        max_translate = 10.0
        if self.translate_mode == 'rel': max_translate = 0.25
        shear_angle = math.degrees(0.30)
        return {
            # op_name: magnitudes, neutral_magnitude
            "Identity": (torch.zeros(self.num_magnitude_bins), 0.0),
            "Shear": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            "Translate": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            "Rotate": (torch.linspace(-30.0, 30.0, self.num_magnitude_bins), 0.0),
        }

    def _geometry_wide_augmentation_space(self, ) -> Dict[str, Tuple[Tensor, bool]]:
        max_translate = 32.0
        if self.translate_mode == 'rel': max_translate = 0.50
        shear_angle = math.degrees(0.99)
        return {
            # op_name: magnitudes, neutral_magnitude
            "Identity": (torch.zeros(self.num_magnitude_bins), 0.0),
            "Shear": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            "Translate": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            "Rotate": (torch.linspace(-30.0, 30.0, self.num_magnitude_bins), 0.0),
        }

    def apply_random_geometric_op(self, img):
        aug_keys_geometry = list(self.geometry_augmentation_space.keys())
        # pick a geometric image operation at random
        idx = int(torch.randint(len(self.geometry_augmentation_space), (1,)).item())
        op_name = aug_keys_geometry[idx]
        # pick an operation strength at random, no need to keep index
        magnitudes, _ = self.geometry_augmentation_space[op_name]
        op_magnitude_index = int(torch.randint(len(magnitudes), (1,), dtype=torch.long))
        magnitude = float(magnitudes[op_magnitude_index])
        # apply geometric op and return img
        return _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=self.fill,
                         translate_mode=self.translate_mode)

    def forward(self, img: Image.Image) -> Tuple[Image.Image, int, int, int]:
        # first of all pre-transforms
        for t in self.pre_transforms:
            img = t(img)
        aug_keys_color = list(self.color_augmentation_space.keys())

        if self.fast_test:  # we only return identity
            n_mags = self.num_magnitude_bins
            for t in self.post_transforms:
                img = t(img)
            # identity has op_index=op_sign=0
            op_index, op_magnitude_index, op_sign = 0, n_mags // 2, 0
            if self.mode == 'return_transforms':
                return img, op_index
            elif self.mode == 'return_signs':
                return img, 0
            return img, op_index, op_magnitude_index, op_sign

        elif self.is_test:

            n_mags = self.num_magnitude_bins
            # no need to return more than one copy of identity(img)
            aug_keys_color.remove('Identity')
            # but we do need to apply post_transforms to it
            tr_img = img.copy()
            for t in self.post_transforms:
                tr_img = t(tr_img)
            # add first item to returning lists, the (post-transformed) identity
            tr_imgs, op_indexes, op_magnitude_indexes, op_signs = [tr_img], [0], [n_mags // 2], [0]
            # print('how many samples? ', len(aug_keys_color))
            # i=0
            for op_index in range(len(aug_keys_color)):

                # apply random operation from geometry augmentation space
                # not really, this is test time, we are fine with identity for geometric
                # img = self.apply_random_geometric_op(img)
                op_name = aug_keys_color[op_index]
                magnitudes, neutral_magnitude = self.color_augmentation_space[op_name]
                magnitudes = np.array(magnitudes)
                mid_neg_mag_idx, mid_pos_mag_idx = int(0.25 * n_mags), int(0.75 * n_mags)
                mid_neg_magnitude = magnitudes[mid_neg_mag_idx]
                mid_pos_magnitude = magnitudes[mid_pos_mag_idx]

                # transform image with mid_neg_magnitude
                tr_img = _apply_op(img, op_name, mid_neg_magnitude, interpolation=self.interpolation,
                                   fill=self.fill, translate_mode=self.translate_mode)
                # add post_transforms
                for t in self.post_transforms:
                    tr_img = t(tr_img)
                tr_imgs.append(tr_img)

                op_indexes.append(op_index + 1)  # account for identity
                op_magnitude_indexes.append(mid_neg_mag_idx)
                #                 op_signs.append(0)  # because its negative magnitude
                op_sign = 0 if op_name == "Identity" else 1
                op_signs.append(op_sign)
                # transform image with mid_pos_magnitude
                tr_img = _apply_op(img, op_name, mid_pos_magnitude, interpolation=self.interpolation,
                                   fill=self.fill, translate_mode=self.translate_mode)
                # add post_transforms
                for t in self.post_transforms:
                    tr_img = t(tr_img)
                tr_imgs.append(tr_img)

                op_indexes.append(op_index + 1)  # because we removed identity in the first place
                op_sign = 0 if op_name == "Identity" else 1
                op_signs.append(op_sign)
                # op_signs.append(1)  # because its positive magnitude
                op_magnitude_indexes.append(mid_pos_mag_idx)

            #     i+=1
            # print(i, 'samples')
            # print(len(op_signs))
            # sys.exit()
            if self.mode == 'return_transforms':
                return tr_imgs, op_indexes
            elif self.mode == 'return_signs':
                return tr_imgs, op_signs
            return tr_imgs, op_indexes, op_magnitude_indexes, op_signs

        else:  # training mode
            aug_keys_geometry = list(self.geometry_augmentation_space.keys())
            # pick a geometric image operation at random, no need to keep index
            idx = int(torch.randint(len(self.geometry_augmentation_space), (1,)).item())
            op_name = aug_keys_geometry[idx]
            # pick an operation strength at random, keep index
            magnitudes, _ = self.geometry_augmentation_space[op_name]
            op_magnitude_index = int(torch.randint(len(magnitudes), (1,), dtype=torch.long))
            magnitude = float(magnitudes[op_magnitude_index])
            # apply geometric op
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=self.fill,
                            translate_mode=self.translate_mode)

            # pick a color image operation at random, keep index
            op_index = int(torch.randint(len(self.color_augmentation_space), (1,)).item())
            op_name = aug_keys_color[op_index]
            # pick an operation strength at random, keep index
            magnitudes, neutral_magnitude = self.color_augmentation_space[op_name]
            op_magnitude_index = int(torch.randint(len(magnitudes), (1,), dtype=torch.long))
            magnitude = float(magnitudes[op_magnitude_index])
            # check if magnitude is above neutral value for this op
            op_sign = 0 if op_name == "Identity" else 1
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=self.fill,
                            translate_mode=self.translate_mode)
            # add post_transforms
            for t in self.post_transforms:
                img = t(img)
            if self.mode == 'return_transforms':
                return img, op_index
            elif self.mode == 'return_signs':
                return img, op_sign
            return img, op_index, op_magnitude_index, op_sign

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'num_magnitude_bins={num_magnitude_bins}'
        s += ', interpolation={interpolation}'
        s += ', fill={fill}'
        s += ')'
        return s.format(**self.__dict__)


class Augment_T3PO(torch.nn.Module):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (FillMode): Desired filling scheme enum defined above in
            :class:`FillMode`. Default is ``FillMode.CONSTANT``. If FillMode.REFLECT is used,
            most geometric transforms resort to skimage, and are more expensive
        """

    def __init__(self, num_magnitude_bins: int = 32, augmentation_space_name: str = 'default',
                 mode='return_transforms', is_test: bool = False, fast_test: bool = False,
                 interpolation: int = Image.NEAREST, fill: str = 'constant', translate_mode: str = 'abs',
                 post_transforms: List = ()) -> None:
        super().__init__()
        self.interpolation = interpolation
        self.fill = fill
        self.translate_mode = translate_mode
        self.num_magnitude_bins = num_magnitude_bins
        self.mode = mode
        if augmentation_space_name == 'default':
            self.augmentation_space = self._default_augmentation_space()
        elif augmentation_space_name == 'wide':
            self.augmentation_space = self._wide_augmentation_space()
        elif augmentation_space_name == 'debug':
            self.augmentation_space = self._debug_augmentation_space()
        elif augmentation_space_name == 'debug_wide':
            self.augmentation_space = self._debug_wide_augmentation_space()

        elif augmentation_space_name == 'reduced':
            self.augmentation_space = self._reduced_augmentation_space()

        else:
            sys.exit('not a valid augmentation space')

        self.n_augs = len(list(self.augmentation_space))
        self.post_transforms = post_transforms
        self.is_test = is_test
        self.fast_test = fast_test

    def _default_augmentation_space(self, ) -> Dict[str, Tuple[Tensor, bool]]:
        max_translate = 10.0
        if self.translate_mode == 'rel': max_translate = 0.25
        shear_angle = math.degrees(0.30)
        return {
            # op_name: magnitudes, neutral_magnitude
            "Identity": (torch.zeros(self.num_magnitude_bins), 0.0),
            "ShearX": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            "ShearY": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            "TranslateX": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            "TranslateY": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            "Rotate": (torch.linspace(-30.0, 30.0, self.num_magnitude_bins), 0.0),
            "Brightness": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
            "Contrast": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
            "Saturation": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
            "Hue": (torch.linspace(-0.05, 0.05, self.num_magnitude_bins), 0.0),
            "Gamma": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
            "Sharpness": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
        }

    def _wide_augmentation_space(self, ) -> Dict[str, Tuple[Tensor, bool]]:
        max_translate = 32.0
        if self.translate_mode == 'rel': max_translate = 0.50
        shear_angle = math.degrees(0.99)
        return {
            # op_name: magnitudes, neutral_magnitude
            "Identity": (torch.zeros(self.num_magnitude_bins), 0.0),
            "ShearX": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            "ShearY": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            "TranslateX": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            "TranslateY": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            "Rotate": (torch.linspace(-135.0, 135.0, self.num_magnitude_bins), 0.0),
            "Brightness": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
            "Contrast": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
            "Saturation": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
            "Hue": (torch.linspace(-0.10, 0.10, self.num_magnitude_bins), 0.0),
            "Gamma": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
            "Sharpness": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
        }

    def _debug_augmentation_space(self, ) -> Dict[str, Tuple[Tensor, bool]]:
        max_translate = 10.0
        if self.translate_mode == 'rel': max_translate = 0.25
        shear_angle = math.degrees(0.30)
        return {
            # op_name: magnitudes, neutral_magnitude
            "Identity": (torch.zeros(self.num_magnitude_bins), 0.0),
            # "ShearX": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            # "ShearY": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            "Shear": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            # "TranslateX": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            # "TranslateY": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            "Translate": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            "Rotate": (torch.linspace(-30.0, 30.0, self.num_magnitude_bins), 0.0),
            "Brightness": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
#             "Contrast": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
            "Saturation": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
            "Hue": (torch.linspace(-0.05, 0.05, self.num_magnitude_bins), 0.0),
#             "Gamma": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
            "Sharpness": (torch.linspace(0.5, 2.0, self.num_magnitude_bins), 1.0),
        }

    def _debug_wide_augmentation_space(self, ) -> Dict[str, Tuple[Tensor, bool]]:
        max_translate = 32.0
        if self.translate_mode == 'rel': max_translate = 0.50
        shear_angle = math.degrees(0.99)
        return {
            # op_name: magnitudes, neutral_magnitude
            "Identity": (torch.zeros(self.num_magnitude_bins), 0.0),
            # "ShearX": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            # "ShearY": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            "Shear": (torch.linspace(-shear_angle, shear_angle, self.num_magnitude_bins), 0.0),
            # "TranslateX": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            # "TranslateY": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            "Translate": (torch.linspace(-max_translate, max_translate, self.num_magnitude_bins), 0.0),
            "Rotate": (torch.linspace(-135.0, 135.0, self.num_magnitude_bins), 0.0),
            "Brightness": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
            # "Contrast": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
            "Saturation": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
            "Hue": (torch.linspace(-0.10, 0.10, self.num_magnitude_bins), 0.0),
            # "Gamma": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
            "Sharpness": (torch.linspace(0.10, 2.5, self.num_magnitude_bins), 1.0),
        }

    # def forward(self, img: Image.Image) -> Tuple[Image.Image, int, int, int]:
    def forward(self, img):
        """
            img (PIL Image): Image to be transformed.

            # if not self.is_test, it will return a transformed version of img using a randomly chosen transform
            # from augmentation_space, together with the corresponding operation index, magnitude index, and sign.

            # if self.is_test, it will return a list of images including the original one plus versions
            # transformed by using all transforms in augmentation_space, with midway negative/positive magnitude
            # it will also return the corresponding operation indexes, magnitude indexes, and signs
            # this amounts to returning four lists of 2*len(augmentation_space)-1, one containing images and the
            # other three containing corresponding integers.
        """

        aug_keys = list(self.augmentation_space.keys())
        if self.fast_test: # we only return identity
            n_mags = self.num_magnitude_bins
            for t in self.post_transforms:
                img = t(img)
            op_index, op_magnitude_index, op_sign = 0, n_mags // 2, 0
            if self.mode == 'return_transforms':
                return img, op_index
            elif self.mode == 'return_signs':
                return img, 0
            return img, op_index, op_magnitude_index, op_sign

        elif self.is_test:

            n_mags = self.num_magnitude_bins
            # no need to return more than one copy of identity(img)
            aug_keys.remove('Identity')
            # but we do need to apply post_transforms to it
            tr_img = img.copy()
            for t in self.post_transforms:
                tr_img = t(tr_img)

            tr_imgs, op_indexes, op_magnitude_indexes, op_signs = [tr_img], [0], [n_mags // 2], [0]

            for op_index in range(len(aug_keys)):
                op_name = aug_keys[op_index]
                magnitudes, neutral_magnitude = self.augmentation_space[op_name]
                magnitudes = np.array(magnitudes)
                mid_neg_mag_idx, mid_pos_mag_idx = int(0.25 * n_mags), int(0.75 * n_mags)
                mid_neg_magnitude = magnitudes[mid_neg_mag_idx]
                mid_pos_magnitude = magnitudes[mid_pos_mag_idx]

                # transform image with mid_neg_magnitude
                tr_img = _apply_op(img, op_name, mid_neg_magnitude, interpolation=self.interpolation,
                                   fill=self.fill, translate_mode=self.translate_mode)
                # add post_transforms
                for t in self.post_transforms:
                    tr_img = t(tr_img)
                tr_imgs.append(tr_img)

                op_indexes.append(op_index + 1)
                op_magnitude_indexes.append(mid_neg_mag_idx)
#                 op_signs.append(0)  # because its negative magnitude
                op_sign = 0 if op_name in ["Identity","Shear", "ShearX", "ShearY",
                                           "Translate", "TranslateX","TranslateY","Rotate"] else 1
                op_signs.append(op_sign)  # because its a transformed image

                # transform image with mid_pos_magnitude
                tr_img = _apply_op(img, op_name, mid_pos_magnitude, interpolation=self.interpolation,
                                   fill=self.fill, translate_mode=self.translate_mode)
                # add post_transforms
                for t in self.post_transforms:
                    tr_img = t(tr_img)
                tr_imgs.append(tr_img)

                op_indexes.append(op_index + 1)  # because we removed identity in the first place
                op_magnitude_indexes.append(mid_pos_mag_idx)
#                 op_signs.append(1)  # because its positive magnitude
                op_sign = 0 if op_name in ["Identity","Shear", "ShearX", "ShearY",
                                           "Translate", "TranslateX","TranslateY","Rotate"] else 1
                op_signs.append(op_sign)  # because its a transformed image
            if self.mode == 'return_transforms':
                return tr_imgs, op_indexes
            elif self.mode == 'return_signs':
                return tr_imgs, op_signs
            return tr_imgs, op_indexes, op_magnitude_indexes, op_signs
        else:
            # pick an image operation at random, keep index
            op_index = int(torch.randint(len(self.augmentation_space), (1,)).item())
            op_name = aug_keys[op_index]
            # pick an operation strength at random, keep index
            magnitudes, neutral_magnitude = self.augmentation_space[op_name]
            op_magnitude_index = int(torch.randint(len(magnitudes), (1,), dtype=torch.long))
            magnitude = float(magnitudes[op_magnitude_index])
#             # check if magnitude is above neutral value for this op
#             op_sign = int(magnitude >= neutral_magnitude)

            # check if operation is identity (sign=0) or not (sign=1)
            op_sign = 0 if op_name in ["Identity", "Shear", "ShearX", "ShearY",
                                       "Translate", "TranslateX", "TranslateY", "Rotate"] else 1
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=self.fill,
                            translate_mode=self.translate_mode)
            # add post_transforms
            for t in self.post_transforms:
                img = t(img)
            if self.mode == 'return_transforms':
                # print(type(img), img.shape, type(op_index))
                return img, op_index
            elif self.mode == 'return_signs':
                return img, op_sign
            return img, op_index, op_magnitude_index, op_sign

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'num_magnitude_bins={num_magnitude_bins}'
        s += ', interpolation={interpolation}'
        s += ', fill={fill}'
        s += ')'
        return s.format(**self.__dict__)