from PIL import Image
import numpy as np
import torch
import random
import torchvision.transforms as T
import re
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch import nn
import matplotlib.pyplot as plt
from seaborn import color_palette
from torch.optim.lr_scheduler import _LRScheduler
import cv2
from transformers import Mask2FormerForUniversalSegmentation
import torch.nn.functional as F


class Normalize:

    forward = \
        T.Normalize(
            mean=[.485, .456, .406],
            std=[.229, .224, .225]
        )

    reverse = \
        T.Normalize(
            mean=[-.485/.229, -.456/.224, -.406/.225],
            std=[1/.229, 1/.224, 1/.225]
        )


def set_randomseed(seed=None, return_seed=False):

    if seed is None:
        seed = np.random.randint(2147483647)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if return_seed:
        return seed


def get_gtpath(p):
    pgt = re.sub('\.jpg$', '.png', p)
    pgt = pgt.replace('/im/', '/gt/')
    return pgt


class SimpleDataset(Dataset):

    def __init__(
        self, annotation_file, dirbase=None, has_label=True,
        transform=None, transform_target=None,
        transform_color=None, ix_nolabel=255, pad=False,
        long=352, **kwargs
    ):

        with open(annotation_file, 'r') as f:
            self.impaths = f.read().split('\n')[:-1]

        if dirbase is not None:
            self.impaths = [os.path.join(dirbase, p) for p in self.impaths]

        if has_label:
            self.labelpaths = [get_gtpath(p) for p in self.impaths]

        self.dirbase = dirbase
        self.transform = transform
        self.transform_color = transform_color
        self.fl_transform_color = transform_color is not None
        self.transform_target = transform_target
        self.ix_nolabel = ix_nolabel
        self.has_label = has_label

        self.normalize = Normalize.forward
        self.long = long
        self.pad = pad

        rtk2mocamba = torch.Tensor([0, 2, 15+1, 12, 6, 15+2, 3, 15+3, 15+4, 15+5, 15+6, 15+7, 4]).long()
        self.n_classes = rtk2mocamba.max() + 1
        self.remap_labels = rtk2mocamba

    def __len__(self):
        return len(self.impaths)

    def __getitem__(self, idx):
        self.path = self.impaths[idx]
        image = read_image(self.impaths[idx])
        oldh, oldw = image.shape[1:]

        if self.fl_transform_color:
            # The color transformation doesn't follow the set seed.
            image = self.transform_color(image)

        if self.transform:
            state = set_randomseed(return_seed=True)
            image = self.transform(image)

        image = self.normalize(image / 255)

        if self.has_label:
            label = read_image(self.labelpaths[idx])

            # Change RTK integers to mocamba standard.
            if 'RTK' in self.path and not 'unlabeled' in self.path:
                label = self.remap_labels[label.long()]

            if self.transform:
                set_randomseed(seed=state)
                label = self.transform_target(label).to(torch.uint8)
        else:
            label = torch.zeros((1, oldh, oldw))

        if self.pad and oldw != self.long:
            neww, newh = get_newshape(oldh, oldw, self.long)

            image = T.functional.resize(
                image, (newh, neww), antialias=True,
                interpolation=T.InterpolationMode.BICUBIC)

            padh = self.long - newh
            padw = self.long - neww
            image = nn.functional.pad(image, (0, padw, 0, padh)).float()

            label = T.functional.resize(
                label, (newh, neww), antialias=True,
                interpolation=T.InterpolationMode.NEAREST)

            label = nn.functional.pad(label, (0, padw, 0, padh)).float()

        return image, label.long()


def get_holemask(final_size, prob=.3, base_size=16):

    resize_mask = lambda x: \
        T.functional.resize(
            x, final_size,
            interpolation=T.InterpolationMode.NEAREST
        )

    hole_mask32 = torch.tensor(prob).repeat((1, 32, 32)).bernoulli()
    hole_mask32 = resize_mask(hole_mask32)

    hole_mask16 = torch.tensor(prob).repeat((1, 16, 16)).bernoulli()
    hole_mask16 = resize_mask(hole_mask16)

    hole_mask8 = torch.tensor(prob).repeat((1, 8, 8)).bernoulli()
    hole_mask8 = resize_mask(hole_mask8)

    hole_mask = hole_mask32 + hole_mask16 + hole_mask8

    hole_mask = (hole_mask > 0)
    return hole_mask


def get_newshape(oldh, oldw, long=1024):
    scale = long * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (neww, newh)


def resize_im(im, long=1024, fl_pad=False, inter_nearest=False):

    target_size = get_newshape(*im.shape[:2], long=long)
    if im.ndim == 3 and im.shape[2] == 1:
        im = im[:, :, 0]

    if inter_nearest:
        inter = Image.Resampling.NEAREST
    else:
        inter = Image.Resampling.BICUBIC

    newim = Image.fromarray(im).resize(target_size, inter)
    newim = np.array(newim)

    if fl_pad:
        newh, neww = newim.shape[:2]
        padh = long - newh
        padw = long - neww

        if newim.ndim == 3 and newim.shape[2] == 3:
            pad_values = (124, 116, 104)
        else:
            pad_values = (0,)

        newim = cv2.copyMakeBorder(
            newim, 0, padh, 0, padw, cv2.BORDER_CONSTANT, value=pad_values)

        return newim, (padh, padw)
    else:
        return newim


class SignDataset(SimpleDataset):

    def __init__(self, hole_prob=0.3, **kwargs):

        super().__init__(**kwargs)

        mocamba_signlabels = torch.Tensor([3, 6, 15+2]).long()
        self.remap_signs = torch.zeros(self.n_classes).long()
        self.remap_signs[mocamba_signlabels] = torch.arange(len(mocamba_signlabels)) + 1
        self.mocamba_signlabels = mocamba_signlabels

        self.hole_prob = hole_prob

    def __len__(self):
        return len(self.impaths)

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        h, w = image.shape[1:]

        if self.has_label:
            label = self.remap_signs[label]
            hole_mask = get_holemask(label.shape[-2:], self.hole_prob)
            hole_label = label.clone()
            hole_label[hole_mask] = 0
        else:
            hole_label = label

        return hole_label, label, image,


def plot_grid(inp, fl_normalize_back=False):
    tmp = inp.clone()
    if fl_normalize_back:
        tmp = Normalize.reverse(tmp)

    tmp = tmp.moveaxis(0, -2).flatten(-2, -1).permute(1, 2, 0)
    plt.imshow(tmp)
    plt.show()


class Colorizer:

    def __init__(self, n_classes):
        range = np.linspace(0, 1, n_classes)
        colors = color_palette('gist_rainbow', as_cmap=True)(range)
        colors = colors[..., :3]
        colors = np.concatenate([np.array([[0, 0, 0]]), colors])
        self.colors = colors

    def __call__(self, classmap):
        return self.colors[classmap]


class TorchColorizer(Colorizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.colors = torch.Tensor(self.colors)

    def __call__(self, classmap):
        tmp = self.colors[classmap]
        return tmp.permute(0, 4, 2, 3, 1)[..., 0]


def float_to_uint8(x):
    return (x * 255).round().clip(0, 255).astype(np.uint8)


def get_single_image_miou(pred, label, n_classes):
    CM_abs = get_CM(pred, label, n_classes)
    pred_P = CM_abs.sum(axis=0)
    gt_P = CM_abs.sum(axis=1)
    true_P = np.diag(CM_abs)

    CM_iou = true_P / (pred_P + gt_P - true_P)
    miou = np.nanmean(CM_iou)
    return miou


def get_CM(pred, label, n_classes):
    cm = np.bincount(
        n_classes * label.flatten() + pred.flatten(),
        minlength=n_classes ** 2)

    return cm.reshape(
        n_classes, n_classes, order='F').astype(int)


def get_CM_fromloader(
    dloader, model, n_classes, ix_nolabel=255, filling_signs=False
):

    CM_abs = np.zeros((n_classes, n_classes), dtype=int)

    if filling_signs:
        for inp_labelholes, inp_label, inp_im in dloader:
            test_preds = model(inp_labelholes.cuda(), inp_im.cuda()).cpu()
            test_preds = test_preds.argmax(1, keepdim=True)
            for pr_i, y_i in zip(test_preds, inp_label.cpu()):
                CM_abs += get_CM(pr_i, y_i, n_classes)
    else:
        for inp_im, inp_label in dloader:
            test_preds = model(inp_im.cuda())[0].cpu()
            for pr_i, y_i in zip(test_preds, inp_label.cpu()):
                CM_abs += get_CM(pr_i, y_i, n_classes)

    pred_P = CM_abs.sum(axis=0)
    gt_P = CM_abs.sum(axis=1)
    true_P = np.diag(CM_abs)

    CM_iou = true_P / (pred_P + gt_P - true_P)
    miou = np.nanmean(CM_iou)
    return miou, CM_iou, CM_abs


class WarmupLR(_LRScheduler):

    def __init__(
        self, optimizer, min_lr_factor=0.001, logger=None,
        n_warmup_max=5, **kwargs
    ):

        # avoid lr changes after calling super
        base_lrs_bkp = [pg['lr'] for pg in optimizer.param_groups]

        self.min_lr_factor = min_lr_factor
        self.n_warmup_max = n_warmup_max
        self.n_warmup = 1
        self.fl_warmup = True

        super().__init__(optimizer)
        self.base_lrs = base_lrs_bkp

    def get_lr(self):
        wu_lrs = []
        if self.fl_warmup:
            for blr in self.base_lrs:
                wu_lr = blr * self.min_lr_factor + \
                    blr * (1 - self.min_lr_factor) * \
                    (self.n_warmup - 1) / self.n_warmup_max
                wu_lrs.append(wu_lr)

            self.n_warmup += 1.
            if self.n_warmup > (self.n_warmup_max + 1):
                self.fl_warmup = False
        else:
            wu_lrs = self.base_lrs

        return wu_lrs


def save_ckpt(p, model, opt, miou_val, it):
    torch.save({
        'state': model.state_dict(),
        'opt_state': opt.state_dict(),
        'best_miou': miou_val,
        'it': it,
    }, p)


def create_dirpath_ifneeded(p):
    d = os.path.dirname(p)
    if not os.path.exists(d):
        os.makedirs(d)


class CocoProcessor:

    def __init__(self, coco):
        self.coco = coco
        self.cat_ids = coco.getCatIds()

    def ann2mask(self, ann):
        return ann['category_id'] * self.coco.annToMask(ann)

    def mask_from_anns(self, anns):
        gt = np.stack(list(map(self.ann2mask, anns)), axis=0)
        # In case of class overlapping, higher classes are priorotized
        newgt = gt[-1]
        for x in gt[:-1][::-1]:
            mask = (newgt == 0)
            newgt[mask] = x[mask]
        return newgt

    def mask_from_cocoim(self, cocoim):
        anns_ids = self.coco.getAnnIds(
            imgIds=cocoim, catIds=self.cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(anns_ids)
        mask = self.mask_from_anns(anns)
        return mask


class MyMask2Former(Mask2FormerForUniversalSegmentation):

    def forward(self, input, label=None):

        if label is not None:
            mask_labels = F.one_hot(label, num_classes=self.n_classes).float()
            mask_labels = mask_labels.permute(0, 3, 1, 2)
            mask_labels = [m for m in mask_labels]
            class_labels = [
                torch.arange(self.n_classes).to(input.device)] * len(input)
        else:
            mask_labels = None
            class_labels = None

        output = super().forward(
            pixel_values=input,
            mask_labels=mask_labels,
            class_labels=class_labels
        )

        # Get classes probs and remove the last class (the empty class)
        masks_classes = output.class_queries_logits.softmax(-1)[..., :-1]
        masks_probs = output.masks_queries_logits
        masks_probs = F.interpolate(masks_probs, input.shape[-2:])

        masks_probs = masks_probs.sigmoid()
        segmentation_map = torch.einsum(
            "bqc, bqhw -> bchw", masks_classes, masks_probs)
        segmentation_map = F.interpolate(segmentation_map, input.shape[-2:])
        segmentation_map = segmentation_map.argmax(dim=1, keepdim=True)

        return segmentation_map, output.loss



class ModelHelper:

    input_from_path = \
        T.Compose([
            read_image,
            T.Lambda(lambda x: x / 255),
            Normalize.forward,
            T.Lambda(lambda x: x[None].cuda())
        ])


def read_trainlist(fpath):
    with open(fpath, 'r') as f:
        lines = f.read().split('\n')
    if len(lines[-1]) == 0:
        lines = lines[:-1]
    return lines

