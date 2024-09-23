import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np
from lib.dice_loss import dice_coeff
from lib.unet_utils import batch


def eval_net(net, dataset, lendata, gpu=False, batch_size=8, is_loss=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    criterion = nn.BCELoss()
    with torch.no_grad():
        for i, b in tqdm(enumerate(batch(dataset, batch_size))):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)[0]

            if is_loss:
                loss = criterion(masks_pred, true_masks)
                tot += loss.item()
                # progress_bar.update(batch_size)
                # progress_bar.set_postfix(BCE=loss.item())
            else:
                masks_pred = (masks_pred > 0.5).float()
                dice = dice_coeff(masks_pred, true_masks).item()
                tot += dice
                # progress_bar.update(batch_size)
                # progress_bar.set_postfix(DICE=dice)
    value = tot / i
    return value


def eval_net_multi_stream(net, dataset, lendata, gpu=True, batch_size=8, is_loss=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    criterion = nn.BCELoss()
    with torch.no_grad(), tqdm(total=lendata) as progress_bar:
        for i, b in enumerate(batch(dataset, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs, multi_stream = True)[0]

            if is_loss:
                loss = criterion(masks_pred, true_masks)
                tot += loss.item()
                progress_bar.update(batch_size)
                progress_bar.set_postfix(BCE=loss.item())
            else:
                masks_pred = (masks_pred > 0.5).float()
                dice = dice_coeff(masks_pred, true_masks).item()
                tot += dice
                progress_bar.update(batch_size)
                progress_bar.set_postfix(DICE=dice)

    value = tot / i
    return value