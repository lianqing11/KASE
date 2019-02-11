from __future__ import division
import cv2
import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings
from skimage import img_as_float
from . import functional as F

__all__ = ["Compose", "ToTensor", "ToPILImage", "Normalize", "Resize", "Scale", "CenterCrop", "Pad",
           "Lambda", "RandomCrop", "RandomHorizontalFlip", "RandomVerticalFlip", "RandomResizedCrop",
           "RandomSizedCrop", "FiveCrop", "TenCrop", "LinearTransformation", "ColorJitter", "RandomRotation",
           "Grayscale", "RandomGrayscale"]


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2):
        for t in self.transforms:
            img1, img2 = t(img1, img2)
        return img1, img2


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic1, pic2):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        pic1 = F.to_tensor(pic1)
        pic2 = F.to_tensor(pic2)

        return pic1, pic2


class ToPILImage(object):
    """Convert a tensor or an ndarray to PIL Image.

    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.

    Args:
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
            If ``mode`` is ``None`` (default) there are some assumptions made about the input data:
            1. If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
            2. If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
            3. If the input has 1 channel, the ``mode`` is determined by the data type (i,e,
            ``int``, ``float``, ``short``).

    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes
    """
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic1, pic2):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.

        Returns:
            PIL Image: Image converted to PIL Image.

        """
        return F.to_pil_image(pic1, self.mode), F.to_pil_image(pic2, self.mode)


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor1, tensor2):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor1, self.mean, self.std), F.normalize(tensor2, self.mean, self.std)


class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img1, img2):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        return F.resize(img1, self.size, self.interpolation), F.resize(img2, self.size, self.interpolation)


class Scale(Resize):
    """
    Note: This transform is deprecated in favor of Resize.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("The use of the transforms.Scale transform is deprecated, " +
                      "please use transforms.Resize instead.")
        super(Scale, self).__init__(*args, **kwargs)


class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        return F.center_crop(img, self.size)


class Pad(object):
    """Pad the given PIL Image on all sides with the given "pad" value.

    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
    """

    def __init__(self, padding, fill=0):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill

    def __call__(self, img1, img2):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img1, self.padding, self.fill), F.pad(img2, self.padding, self.fill)


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img1):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = F.pad(img, self.padding)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img1, i, j, h, w), F.crop(img2, i, j, h, w)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a probability of 0.5."""

    def __call__(self, img1, img2):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            img1 = F.hflip(img1)
        if random.random() < 0.5:
            img2 = F.hflip(img2)
        return img1, img2


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a probability of 0.5."""

    def __call__(self, img1):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            img1 = F.vflip(img1)
        if random.random() < 0.5:
            img2 = F.vflip(img2)
        return img1, img2


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img1, img2):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly cropped and resize image.
        """
        i, j, h, w = self.get_params(img1, self.scale, self.ratio)
        img1 = F.resized_crop(img1, i, j, h, w, self.size, self.interpolation)
        img2 = F.resized_crop(img2, i, j, h, w, self.size, self.interpolation)
        return img1, img2

class RandomSizedCrop(RandomResizedCrop):
    """
    Note: This transform is deprecated in favor of RandomResizedCrop.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("The use of the transforms.RandomSizedCrop transform is deprecated, " +
                      "please use transforms.RandomResizedCrop instead.")
        super(RandomSizedCrop, self).__init__(*args, **kwargs)


class FiveCrop(object):
    """Crop the given PIL Image into four corners and the central crop

    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.

    Args:
         size (sequence or int): Desired output size of the crop. If size is an ``int``
            instead of sequence like (h, w), a square crop of size (size, size) is made.

    Example:
         >>> transform = Compose([
         >>>    FiveCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """

    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return F.five_crop(img, self.size)


class TenCrop(object):
    """Crop the given PIL Image into four corners and the central crop plus the flipped version of
    these (horizontal flipping is used by default)

    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        vertical_flip(bool): Use vertical flipping instead of horizontal

    Example:
         >>> transform = Compose([
         >>>    TenCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """

    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, img):
        return F.ten_crop(img, self.size, self.vertical_flip)


class LinearTransformation(object):
    """Transform a tensor image with a square transformation matrix computed
    offline.

    Given transformation_matrix, will flatten the torch.*Tensor, compute the dot
    product with the transformation matrix and reshape the tensor to its
    original shape.

    Applications:
    - whitening: zero-center the data, compute the data covariance matrix
                 [D x D] with np.dot(X.T, X), perform SVD on this matrix and
                 pass it as transformation_matrix.

    Args:
        transformation_matrix (Tensor): tensor [D x D], D = C x H x W
    """

    def __init__(self, transformation_matrix):
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            raise ValueError("transformation_matrix should be square. Got " +
                             "[{} x {}] rectangular matrix.".format(*transformation_matrix.size()))
        self.transformation_matrix = transformation_matrix

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be whitened.

        Returns:
            Tensor: Transformed image.
        """
        if tensor.size(0) * tensor.size(1) * tensor.size(2) != self.transformation_matrix.size(0):
            raise ValueError("tensor and transformation matrix have incompatible shape." +
                             "[{} x {} x {}] != ".format(*tensor.size()) +
                             "{}".format(self.transformation_matrix.size(0)))
        flat_tensor = tensor.view(1, -1)
        transformed_tensor = torch.mm(flat_tensor, self.transformation_matrix)
        tensor = transformed_tensor.view(tensor.size())
        return tensor


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img1, img2):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        img1 = transform(img1)
        img2 = transform(img2)
        return img1, img2

class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = np.random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img1, img2):
        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        img1 = F.rotate(img1, angle, self.resample, self.expand, self.center)
        angle = self.get_params(self.degrees)
        img2 = F.rotate(img2, angle, self.resample, self.expand, self.center)
        return img1, img2

class Grayscale(object):
    """Convert image to grayscale.

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image

    Returns:
        PIL Image: Grayscale version of the input.
        - If num_output_channels == 1 : returned image is single channel
        - If num_output_channels == 3 : returned image is 3 channel with r == g == b

    """

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        return F.to_grayscale(img, num_output_channels=self.num_output_channels)


class RandomGrayscale(object):
    """Randomly convert image to grayscale with a probability of p (default 0.1).

    Args:
        p (float): probability that image should be converted to grayscale.

    Returns:
        PIL Image: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    """

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        num_output_channels = 1 if img.mode == 'L' else 3
        if random.random() < self.p:
            return F.to_grayscale(img, num_output_channels=num_output_channels)
        return img


def join_xf_col(xf, col_mtx, col_off):
    return np.concatenate([xf.reshape((-1, 6)),
                           col_mtx.reshape((-1, 9)),
                           col_off.reshape((-1, 3))], axis=1).astype(np.float32)


def identity_xf(N):
    """
    Construct N identity 2x3 transformation matrices
    :return: array of shape (N, 2, 3)
    """
    xf = np.zeros((N, 2, 3), dtype=np.float32)
    xf[:, 0, 0] = xf[:, 1, 1] = 1.0
    return xf


def inv_nx2x2(X):
    """
    Invert the N 2x2 transformation matrices stored in X; a (N,2,2) array
    :param X: transformation matrices to invert, (N,2,2) array
    :return: inverse of X
    """
    rdet = 1.0 / (X[:, 0, 0] * X[:, 1, 1] - X[:, 1, 0] * X[:, 0, 1])
    y = np.zeros_like(X)
    y[:, 0, 0] = X[:, 1, 1] * rdet
    y[:, 1, 1] = X[:, 0, 0] * rdet
    y[:, 0, 1] = -X[:, 0, 1] * rdet
    y[:, 1, 0] = -X[:, 1, 0] * rdet
    return y

def inv_nx2x3(m):
    """
    Invert the N 2x3 transformation matrices stored in X; a (N,2,3) array
    :param X: transformation matrices to invert, (N,2,3) array
    :return: inverse of X
    """
    m2 = m[:, :, :2]
    mx = m[:, :, 2:3]
    m2inv = inv_nx2x2(m2)
    mxinv = np.matmul(m2inv, -mx)
    return np.append(m2inv, mxinv, axis=2)

def cat_nx2x3(a, b):
    """
    Multiply the N 2x3 transformations stored in `a` with those in `b`
    :param a: transformation matrices, (N,2,3) array
    :param b: transformation matrices, (N,2,3) array
    :return: `a . b`
    """
    a2 = a[:, :, :2]
    b2 = b[:, :, :2]

    ax = a[:, :, 2:3]
    bx = b[:, :, 2:3]

    ab2 = np.matmul(a2, b2)
    abx = ax + np.matmul(a2, bx)
    return np.append(ab2, abx, axis=2)

def rotation_matrices(thetas):
    """
    Generate rotation matrices
    :param thetas: rotation angles in radians as a (N,) array
    :return: rotation matrices, (N,2,3) array
    """
    N = thetas.shape[0]
    rot_xf = np.zeros((N, 2, 3), dtype=np.float32)
    rot_xf[:, 0, 0] = rot_xf[:, 1, 1] = np.cos(thetas)
    rot_xf[:, 1, 0] = np.sin(thetas)
    rot_xf[:, 0, 1] = -np.sin(thetas)
    return rot_xf

def centre_xf(xf, size):
    """
    Centre the transformations in `xf` around (0,0), where the current centre is assumed to be at the
    centre of an image of shape `size`
    :param xf: transformation matrices, (N,2,3) array
    :param size: image size
    :return: centred transformation matrices, (N,2,3) array
    """
    height, width = size

    # centre_to_zero moves the centre of the image to (0,0)
    centre_to_zero = np.zeros((1, 2, 3), dtype=np.float32)
    centre_to_zero[0, 0, 0] = centre_to_zero[0, 1, 1] = 1.0
    centre_to_zero[0, 0, 2] = -float(width) * 0.5
    centre_to_zero[0, 1, 2] = -float(height) * 0.5

    # centre_to_zero then xf
    xf_centred = cat_nx2x3(xf, centre_to_zero)

    # move (0,0) back to the centre
    xf_centred[:, 0, 2] += float(width) * 0.5
    xf_centred[:, 1, 2] += float(height) * 0.5

    return xf_centred


def identity_col_mtx(N):
    """
    Construct N identity 3x3 colour matrices
    :return: array of shape (N, 3, 3)
    """
    col_mtx = np.zeros((N, 3, 3), dtype=np.float32)
    col_mtx[:, 0, 0] = col_mtx[:, 1, 1] = col_mtx[:, 2, 2] = 1.0
    return col_mtx


def identity_col_off(N):
    """
    Construct N identity colour offsets
    :return: array of shape (N, 3)
    """
    col_off = np.zeros((N, 3), dtype=np.float32)
    return col_off


def identity_xf_col(N):
    xf = identity_xf(N)
    col_mtx = identity_col_mtx(N)
    col_off = identity_col_off(N)
    return join_xf_col(xf, col_mtx, col_off)


def axis_angle_rotation_matrices(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    # Normalize
    axis = axis/np.sqrt(np.sum(axis*axis, axis=1, keepdims=True))
    a = np.cos(theta/2)
    axis_sin_theta = -axis*np.sin(theta/2)[:, None]
    b = axis_sin_theta[:, 0]
    c = axis_sin_theta[:, 1]
    d = axis_sin_theta[:, 2]
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    rot = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                    [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                    [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    return rot.transpose(2, 0, 1)


class ImageAugmentation (object):
    def __init__(self, hflip, xlat_range, affine_std, rot_std=0.0,
                 intens_scale_range_lower=None, intens_scale_range_upper=None,
                 colour_rot_std=0.0, colour_off_std=0.0,
                 greyscale=False,
                 scale_u_range=None, scale_x_range=None, scale_y_range=None,
                 cutout_size=None, cutout_probability=0.0):
        self.hflip = hflip
        self.xlat_range = xlat_range
        self.affine_std = affine_std
        self.rot_std = rot_std
        self.intens_scale_range_lower = intens_scale_range_lower
        self.intens_scale_range_upper = intens_scale_range_upper
        self.colour_rot_std = colour_rot_std
        self.colour_off_std = colour_off_std
        self.greyscale = greyscale
        self.scale_u_range = scale_u_range
        self.scale_x_range = scale_x_range
        self.scale_y_range = scale_y_range
        self.cutout_size = cutout_size
        self.cutout_probability = cutout_probability


    def aug_xforms(self, N, image_size):
        xf = identity_xf(N)

        if self.hflip:
            x_hflip = np.random.binomial(1, 0.5, size=(N,)) * 2 - 1
            xf[:, 0, 0] = x_hflip.astype(np.float32)

        if self.scale_u_range is not None and self.scale_u_range[0] is not None:
            scl = np.exp(np.random.uniform(low=np.log(self.scale_u_range[0]), high=np.log(self.scale_u_range[1]), size=(N,)))
            xf[:, 0, 0] *= scl
            xf[:, 1, 1] *= scl
        if self.scale_x_range is not None and self.scale_x_range[0] is not None:
            xf[:, 0, 0] *= np.exp(np.random.uniform(low=np.log(self.scale_x_range[0]), high=np.log(self.scale_x_range[1]), size=(N,)))
        if self.scale_y_range is not None and self.scale_y_range[0] is not None:
            xf[:, 1, 1] *= np.exp(np.random.uniform(low=np.log(self.scale_y_range[0]), high=np.log(self.scale_y_range[1]), size=(N,)))

        if self.affine_std > 0.0:
            xf[:, :, :2] += np.random.normal(scale=self.affine_std, size=(N, 2, 2))
        xlat_y_bounds = self.xlat_range * 2.0 / float(image_size[0])
        xlat_x_bounds = self.xlat_range * 2.0 / float(image_size[1])
        xf[:, 0, 2] += np.random.uniform(low=-xlat_x_bounds, high=xlat_x_bounds, size=(N,))
        xf[:, 1, 2] += np.random.uniform(low=-xlat_y_bounds, high=xlat_y_bounds, size=(N,))

        if self.rot_std > 0.0:
            thetas = np.random.normal(scale=self.rot_std, size=(N,))
            rot_xf = rotation_matrices(thetas)
            xf = cat_nx2x3(xf, rot_xf)

        return centre_xf(xf, image_size)

    def aug_colour_xforms(self, N):
        colour_matrix = np.zeros((N, 3, 3))
        colour_matrix[:, 0, 0] = colour_matrix[:, 1, 1] = colour_matrix[:, 2, 2] = 1.0
        if self.colour_rot_std > 0.0:
            # Colour rotation: random thetas
            col_rot_thetas = np.random.normal(scale=self.colour_rot_std, size=(N,))
            # Colour rotation: random axes
            col_rot_axes = np.random.normal(size=(N, 3))
            invalid_axis_mask = np.dot(col_rot_axes, col_rot_axes.T) == 0

            # Re-draw invalid axes
            while invalid_axis_mask.any():
                col_rot_axes[col_rot_axes, :] = np.random.normal(scale=self.colour_rot_std,
                                                                 size=(int(invalid_axis_mask.sum()), 3))
                invalid_axis_mask = np.dot(col_rot_axes, col_rot_axes.T) == 0

            colour_matrix = axis_angle_rotation_matrices(col_rot_axes, col_rot_thetas)

        if self.greyscale:
            grey_factors = np.array([0.2125, 0.7154, 0.0721])
            grey_mtx = np.repeat(grey_factors[None, None, :], 3, axis=1)
            eye_mtx = np.eye(3)[None, :, :]
            factors = np.random.uniform(0.0, 1.0, size=(N, 1, 1))
            greyscale_mtx = eye_mtx + (grey_mtx - eye_mtx) * factors
            colour_matrix = np.matmul(colour_matrix, greyscale_mtx)

        colour_offset = np.zeros((N, 3))
        if self.colour_off_std > 0.0:
            colour_offset = np.random.normal(scale=self.colour_off_std, size=(N, 3))

        if self.intens_scale_range_lower is not None:
            col_factor = np.random.uniform(low=self.intens_scale_range_lower, high=self.intens_scale_range_upper,
                                           size=(N,))
            colour_matrix = colour_matrix * col_factor[:, None, None]

        return colour_matrix, colour_offset


    def aug_colours(self, X):
        colour_matrix, colour_offset = self.aug_colour_xforms(X.shape[0])

        X_c = np.zeros_like(X)
        for i in range(X.shape[0]):
            img = X[i, :, :, :].transpose(1, 2, 0)
            img = np.tensordot(img, colour_matrix[i, :, :], [[2], [1]]) + colour_offset[i, None, None, :]
            X_c[i, :, :, :] = img.transpose(2, 0, 1)

        return X_c


    def aug_cutouts(self, N, img_size):
        if self.cutout_probability > 0.0:
            img_size = np.array(img_size)
            cutout_shape = (np.array(img_size) * self.cutout_size + 0.5).astype(int)
            cutout_p = np.random.uniform(0.0, 1.0, size=(N,))
            cutout_pos_y = np.random.randint(0, img_size[0], size=(N,))
            cutout_pos_x = np.random.randint(0, img_size[1], size=(N,))
            cutout_pos = np.append(cutout_pos_y[:, None], cutout_pos_x[:, None], axis=1)
            cutout_lower = cutout_pos - (cutout_shape[None, :]//2)
            cutout_upper = cutout_lower + cutout_shape[None, :]

            cutout_flags = cutout_p <= self.cutout_probability
            cutout_lower = np.clip(cutout_lower, 0, img_size[None, :]-1)
            cutout_upper = np.clip(cutout_upper, 0, img_size[None, :]-1)

            return cutout_flags, cutout_lower, cutout_upper
        else:
            return None, None, None


    def augment(self, X):
        X = X.copy()
        N = X.shape[0]

        xf = self.aug_xforms(N, X.shape[2:])
        colour_matrix, colour_offset = self.aug_colour_xforms(X.shape[0])

        cutout_flags, cutout_lower, cutout_upper = self.aug_cutouts(N, X.shape[2:])

        for i in range(X.shape[0]):
            img = X[i, :, :, :].transpose(1, 2, 0)
            img = cv2.warpAffine(img, xf[i, :, :], (X.shape[3], X.shape[2]))
            img = np.tensordot(img, colour_matrix[i, :, :], [[2], [1]]) + colour_offset[i, None, None, :]

            if cutout_flags is not None and cutout_flags[i]:
                img[cutout_lower[i, 0]:cutout_upper[i,0], cutout_lower[i, 1]:cutout_upper[i,1], :] = 0.0

            X[i, :, :, :] = img.transpose(2, 0, 1)

        return X

from skimage.util import img_as_float


def _compute_scale_and_crop(image_size, crop_size, padding, random_crop):
    padded_size = crop_size[0] + padding[0], crop_size[1] + padding[1]
    # Compute size ratio from the padded region size to the image_size
    scale_y = float(image_size[0]) / float(padded_size[0])
    scale_x = float(image_size[1]) / float(padded_size[1])
    # Take the minimum as this is the factor by which we must scale to take a `padded_size` sized chunk
    scale_factor = min(scale_y, scale_x)

    # Compute the size of the region that we must extract from the image
    region_height = int(float(crop_size[0]) * scale_factor + 0.5)
    region_width = int(float(crop_size[1]) * scale_factor + 0.5)

    # Compute the additional space available
    if scale_x > scale_y:
        # Crop in X
        extra_x = image_size[1] - region_width
        extra_y = padding[0]
    else:
        # Crop in Y
        extra_y = image_size[0] - region_height
        extra_x = padding[1]

    # Either choose the centre piece or choose a random piece
    if random_crop:
        pos_y = np.random.randint(0, extra_y + 1, size=(1,))[0]
        pos_x = np.random.randint(0, extra_x + 1, size=(1,))[0]
    else:
        pos_y = extra_y // 2
        pos_x = extra_x // 2

    return (pos_y, pos_x), (region_height, region_width)



def _compute_scales_and_crops(image_sizes, crop_size, padding, random_crop):
    padded_size = crop_size[0] + padding[0], crop_size[1] + padding[1]
    # Compute size ratio from the padded region size to the image_size
    image_sizes = image_sizes.astype(float)
    scale_ys = image_sizes[:, 0] / float(padded_size[0])
    scale_xs = image_sizes[:, 1] / float(padded_size[1])
    # Take the minimum as this is the factor by which we must scale to take a `padded_size` sized chunk
    scale_factors = np.minimum(scale_ys, scale_xs)

    # Compute the size of the region that we must extract from the image
    region_sizes = (np.array(crop_size)[None, :] * scale_factors[:, None] + 0.5).astype(int)

    # Compute the additional space available
    extra_space = np.repeat(np.array(padding, dtype=int)[None, :], image_sizes.shape[0], axis=0)
    # Crop in X
    crop_in_x = scale_xs > scale_ys
    extra_space[crop_in_x, 1] = image_sizes[crop_in_x, 1] - region_sizes[crop_in_x, 1]
    # Crop in Y
    crop_in_y = ~crop_in_x
    extra_space[crop_in_y, 0] = image_sizes[crop_in_y, 0] - region_sizes[crop_in_y, 0]

    # Either choose the centre piece or choose a random piece
    if random_crop:
        t = np.random.uniform(0.0, 1.0, size=image_sizes.shape)
        pos = (t * (extra_space + 1.0)).astype(int)
    else:
        pos = extra_space // 2

    return pos, region_sizes


def _compute_scales_and_crops_pairs(image_sizes, crop_size, padding, random_crop, pair_offset_size):
    padded_size = crop_size[0] + padding[0], crop_size[1] + padding[1]
    # Compute size ratio from the padded region size to the image_size
    image_sizes = image_sizes.astype(float)
    scale_ys = image_sizes[:, 0] / float(padded_size[0])
    scale_xs = image_sizes[:, 1] / float(padded_size[1])
    # Take the minimum as this is the factor by which we must scale to take a `padded_size` sized chunk
    scale_factors = np.minimum(scale_ys, scale_xs)

    # Compute the size of the region that we must extract from the image
    region_sizes = (np.array(crop_size)[None, :] * scale_factors[:, None] + 0.5).astype(int)

    # Compute the additional space available
    extra_space = np.repeat(np.array(padding, dtype=int)[None, :], image_sizes.shape[0], axis=0)
    # Crop in X
    crop_in_x = scale_xs > scale_ys
    extra_space[crop_in_x, 1] = image_sizes[crop_in_x, 1] - region_sizes[crop_in_x, 1]
    # Crop in Y
    crop_in_y = ~crop_in_x
    extra_space[crop_in_y, 0] = image_sizes[crop_in_y, 0] - region_sizes[crop_in_y, 0]

    # Either choose the centre piece or choose a random piece
    if random_crop:
        t = np.random.uniform(0.0, 1.0, size=image_sizes.shape)
        pos_a = (t * (extra_space + 1.0)).astype(int)
        if pair_offset_size > 0:
            pos_b_off = np.random.randint(-pair_offset_size, pair_offset_size, size=image_sizes.shape)
            pos_b = np.clip(pos_a + pos_b_off, 0, extra_space)
        else:
            pos_b = pos_a
    else:
        pos_a = extra_space // 2
        pos_b = pos_a

    return pos_a, pos_b, region_sizes



def _compute_scale_and_crop_matrix(img_size, crop_size, padding, random_crop):
    (pos_y, pos_x), (reg_h, reg_w) = _compute_scale_and_crop(img_size, crop_size, padding, random_crop)

    scale_y = float(crop_size[0]) / float(reg_h)
    scale_x = float(crop_size[1]) / float(reg_w)
    off_y = float(pos_y) * scale_y
    off_x = float(pos_x) * scale_x

    scale_and_crop_matrix = np.array([
        [scale_x, 0.0, -off_x,],
        [0.0, scale_y, -off_y]
    ])

    return scale_and_crop_matrix


def _positions_and_sizes_to_matrices(pos, sz, crop_size):
    scales = np.array(crop_size, dtype=float)[None, :] / sz
    offsets = pos * scales

    scale_and_crop_matrices = np.zeros((scales.shape[0], 2, 3))
    scale_and_crop_matrices[:, 0, 0] = scales[:, 1]
    scale_and_crop_matrices[:, 1, 1] = scales[:, 0]
    scale_and_crop_matrices[:, :, 2] = -offsets[:,::-1]

    return scale_and_crop_matrices


def _compute_scale_and_crop_matrices(image_sizes, crop_size, padding, random_crop):
    pos, sz = _compute_scales_and_crops(image_sizes, crop_size, padding, random_crop)
    return _positions_and_sizes_to_matrices(pos, sz, crop_size)


def _compute_scale_and_crop_matrix_pairs(image_sizes, crop_size, padding, random_crop, pair_offset_size):
    pos_a, pos_b, sz = _compute_scales_and_crops_pairs(image_sizes, crop_size, padding, random_crop, pair_offset_size)

    mtx_a = _positions_and_sizes_to_matrices(pos_a, sz, crop_size)
    mtx_b = _positions_and_sizes_to_matrices(pos_b, sz, crop_size)
    return mtx_a, mtx_b




class ScaleAndCrop (object):
    def __init__(self, crop_size, padding, random_crop, mean, std):
        """
        Constructor

        Section `(height + pad_y, width + pad_x)` will be taken from the centre
        of the image
        A `(height, width)` section will be taken from the centre of that

        :param crop_size: the desired image size `(height, width)`
        :param padding: padding around the crop `(pad_y, pad_x)`
        """
        self.crop_size = crop_size
        self.padding = padding
        self.random_crop = random_crop
        self.mean = mean
        self.std = std

    def __call__(self, images):
        images = np.asarray(images, dtype=np.uint8)
        images = [images]
        image_sizes = np.array([list(img.shape[:2]) for img in images])

        pos, sz = _compute_scales_and_crops(image_sizes, self.crop_size, self.padding, self.random_crop)

        result = []
        for i, img in enumerate(images):
            # Compute scale factor to maintain aspect ratio
            cropped_img = img[pos[i,0]:pos[i,0] + sz[i,0], pos[i,1]:pos[i,1] + sz[i,1], :]

            # Scale
            result.append(cv2.resize(cropped_img, (self.crop_size[1], self.crop_size[0])))

        result = result[0]
        result = img_as_float(result).astype(np.float32)
        result = (result - self.mean[None, None, :]) / self.std[None, None, :]
        result = result.transpose(2,0,1)
        result = img_as_float(result).astype(np.float32)
        result = torch.from_numpy(result)
        return result


class ScaleCropAndAugmentAffine(object):

    def __init__(self, crop_size, padding, random_crop, aug, border_value, mean, std):

        self.crop_size = crop_size
        self.padding = padding
        self.random_crop = random_crop
        self.border_value = border_value
        self.aug = aug
        self.mean = mean
        self.std = std




    def __call__(self, images):
        images = [np.asarray(images, dtype=np.uint8)]
        image_sizes = np.array([list(img.shape[:2]) for img in images])
        scale_crop_mtx  = _compute_scale_and_crop_matrices(image_sizes, self.crop_size, self.padding, self.random_crop)

        aug_xf = self.aug.aug_xforms(1, self.crop_size)

        scale_crop_aug_xf = cat_nx2x3(aug_xf, scale_crop_mtx)
        colour_matrix, colour_offset = self.aug.aug_colour_xforms(len(images))
        cutout_flags, cutout_lower, cutout_upper = self.aug.aug_cutouts(len(images), self.crop_size)

        result = []
        for i, img in enumerate(images):
            img = cv2.warpAffine(img, scale_crop_aug_xf[i, :, :], self.crop_size[::-1], borderValue=self.border_value,
                                 borderMode=cv2.BORDER_REFLECT_101)
            img = img_as_float(img).astype(np.float32)
            img = (img - self.mean[None, None, :]) / self.std[None, None, :]
            img = np.tensordot(img, colour_matrix[i, :, :], [[2], [1]]) + colour_offset[i, None, None, :]
            if cutout_flags is not None and cutout_flags[i]:
                img[cutout_lower[i, 0]:cutout_upper[i, 0], cutout_lower[i, 1]:cutout_upper[i, 1], :] = 0.0

            img = img_as_float(img).astype(np.float32)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img)
            result.append(img)


        image = result[0]

        return image





class ScaleCropAndAugmentAffinePair (object):
    def __init__(self, crop_size, padding, pair_offset_size, random_crop, aug, border_value, mean_value, std_value):
        """
        Constructor

        Section `(height + pad_y, width + pad_x)` will be taken from the centre
        of the image
        A `(height, width)` section will be taken from the centre of that

        :param crop_size: the desired image size `(height, width)`
        :param padding: padding around the crop `(pad_y, pad_x)`
        """
        self.crop_size = crop_size
        self.padding = padding
        self.pair_offset_size = pair_offset_size
        self.random_crop = random_crop
        self.aug = aug
        self.border_value = border_value
        self.mean = mean_value
        self.std = std_value




    def __call__(self, images):
        images = np.asarray(images, dtype=np.uint8)
        images = [images]
        image_sizes = np.array([list(img.shape[:2]) for img in images])
        scale_crop_mtx_a, sclcrop_mtx_b = _compute_scale_and_crop_matrix_pairs(
            image_sizes, self.crop_size, self.padding, self.random_crop, self.pair_offset_size)

        aug_xf_a = self.aug.aug_xforms(len(images), self.crop_size)
        aug_xf_b = self.aug.aug_xforms(len(images), self.crop_size)
        scale_crop_aug_xf_a = cat_nx2x3(aug_xf_a, scale_crop_mtx_a)
        scale_crop_aug_xf_b = cat_nx2x3(aug_xf_b, sclcrop_mtx_b)
        colour_matrix_a, colour_offset_a = self.aug.aug_colour_xforms(len(images))
        colour_matrix_b, colour_offset_b = self.aug.aug_colour_xforms(len(images))
        cutout_flags_a, cutout_lower_a, cutout_upper_a = self.aug.aug_cutouts(len(images), self.crop_size)
        cutout_flags_b, cutout_lower_b, cutout_upper_b = self.aug.aug_cutouts(len(images), self.crop_size)

        result_a = []
        result_b = []
        for i, img in enumerate(images):
            img_a = cv2.warpAffine(img, scale_crop_aug_xf_a[i, :, :], self.crop_size[::-1],
                                  borderValue=self.border_value,
                                   borderMode=cv2.BORDER_REFLECT_101)
            img_b = cv2.warpAffine(img, scale_crop_aug_xf_b[i, :, :], self.crop_size[::-1],
                                  borderValue=self.border_value,
                                   borderMode=cv2.BORDER_REFLECT_101)
            img_a = img_as_float(img_a).astype(np.float32)
            img_b = img_as_float(img_b).astype(np.float32)
            img_a = (img_a - self.mean[None, None, :]) / self.std[None, None, :]
            img_b = (img_b - self.mean[None, None, :]) / self.std[None, None, :]
            img_a = np.tensordot(img_a, colour_matrix_a[i, :, :], [[2], [1]]) + colour_offset_a[i, None, None, :]
            img_b = np.tensordot(img_b, colour_matrix_b[i, :, :], [[2], [1]]) + colour_offset_b[i, None, None, :]
            if cutout_flags_a is not None and cutout_flags_a[i]:
                img_a[cutout_lower_a[i, 0]:cutout_upper_a[i, 0], cutout_lower_a[i, 1]:cutout_upper_a[i, 1], :] = 0.0
            if cutout_flags_b is not None and cutout_flags_b[i]:
                img_b[cutout_lower_b[i, 0]:cutout_upper_b[i, 0], cutout_lower_b[i, 1]:cutout_upper_b[i, 1], :] = 0.0


            img_a = img_as_float(img_a).astype(np.float32)
            img_b = img_as_float(img_b).astype(np.float32)
            img_a = img_a.transpose(2, 0, 1)
            img_b = img_b.transpose(2, 0, 1)
            img_a = torch.from_numpy(img_a)
            img_b = torch.from_numpy(img_b)

            result_a.append(img_a)
            result_b.append(img_b)
        img_a = result_a[0]
        img_b = result_b[0]
        return [img_a, img_b]

class ScaleCropAndAugmentAffineMultiple (object):
    def __init__(self, N, crop_size, padding, random_crop, aug, border_value, mean_value, std_value):
        """
        Constructor

        Section `(height + pad_y, width + pad_x)` will be taken from the centre
        of the image
        A `(height, width)` section will be taken from the centre of that

        :param crop_size: the desired image size `(height, width)`
        :param padding: padding around the crop `(pad_y, pad_x)`
        """
        self.N = N
        self.crop_size = crop_size
        self.padding = padding
        self.random_crop = random_crop
        self.aug = aug
        self.border_value = border_value
        self.mean = mean_value
        self.std = std_value


    def __call__(self, images):
        images = np.asarray(images, dtype=np.uint8)
        images = [images]
        image_sizes = np.array([list(img.shape[:2]) for img in images])

        result = []
        for aug_i in range(self.N):
            scale_crop_mtx = _compute_scale_and_crop_matrices(
                image_sizes, self.crop_size, self.padding, self.random_crop)

            aug_xf = self.aug.aug_xforms(len(images), self.crop_size)
            scale_crop_aug_xf = cat_nx2x3(aug_xf, scale_crop_mtx)
            colour_matrix, colour_offset = self.aug.aug_colour_xforms(len(images))
            cutout_flags, cutout_lower, cutout_upper = self.aug.aug_cutouts(len(images), self.crop_size)

            result_aug = []
            for i, img in enumerate(images):
                img = cv2.warpAffine(img, scale_crop_aug_xf[i, :, :], self.crop_size[::-1],
                                      borderValue=self.border_value,
                                       borderMode=cv2.BORDER_REFLECT_101)
                img = img_as_float(img).astype(np.float32)
                img = (img - self.mean[None, None, :]) / self.std[None, None, :]
                img = np.tensordot(img, colour_matrix[i, :, :], [[2], [1]]) + colour_offset[i, None, None, :]
                if cutout_flags is not None and cutout_flags[i]:
                    img[cutout_lower[i, 0]:cutout_upper[i, 0], cutout_lower[i, 1]:cutout_upper[i, 1], :] = 0.0
                result_aug.append(img)
            result.append(result_aug)

        images = []
        for idx, i in enumerate(range(self.N)):
            images.append(result[idx][0].transpose(2,0,1)[np.newaxis])
        images = torch.from_numpy(np.concatenate(images))
        return images

