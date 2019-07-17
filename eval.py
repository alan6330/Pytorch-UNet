import torch
import torch.nn.functional as F
import numpy as np
from dice_loss import dice_coeff

def eval(net, dataset):
    net.eval()
    tot = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)

        mask_pred = net(img)[0]
        mask_pred = (mask_pred > 0.5).float()
        true_mask = np.transpose(true_mask, axes=[1, 2, 0])
        mask_pred=np.transpose(mask_pred, axes=[1, 2, 0])
        t=0
        for i in range(true_mask.shape[0]):
            for j in range(true_mask.shape[1]):
                if true_mask[i,j]==mask_pred[i,j]:
                    t+=1
        tot+=float(t)/(true_mask.shape[0]*true_mask.shape[1])
    return tot / (i + 1)


def eval_net(net, dataset, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)[0]
        mask_pred = (mask_pred > 0.5).float()

        tot += dice_coeff(mask_pred, true_mask).item()
    return tot / (i + 1)
