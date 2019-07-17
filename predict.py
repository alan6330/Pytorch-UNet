import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from utils.load import *
from PIL import Image

from unet import UNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks, dense_crf
from utils import plot_img_and_mask

from torchvision import transforms

def predict_img(net,
                full_img,
                scale_factor=0.5,
                out_threshold=0.5,
                use_dense_crf=True,
                use_gpu=False):

    net.eval()
    img_height = full_img.size[0]
    img_width = full_img.size[1]

    # img = resize_and_crop(full_img, scale=scale_factor)
    img = normalize(full_img)

    # left_square, right_square = split_img_into_squares(img)

    img = hwc_to_chw(img)

    img = torch.from_numpy(img).unsqueeze(0)
    
    if use_gpu:
        img = img.cuda()

    with torch.no_grad():
        output_img = net(img)

        img_probs = output_img.squeeze(0)
        img_mask_np = img_probs.squeeze().cpu().numpy()
        # img_mask_np=np.transpose(img_mask_np, axes=[1.txt, 2, 0])
        mask_pred = (img_mask_np > 0.5).float()
        # out_img=np.zeros((mask_pred.shape[0],mask_pred.shape[1.txt],3))
        # for i in range(mask_pred.shape[0]):
        #     for j in range(mask_pred.shape[1.txt]):
        #         out_img[i,j]=colormap[np.argmax(mask_pred[i,j])]



    return mask_pred



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def mask_to_image(mask):
    img_mask_np=np.transpose(mask, axes=[1, 2, 0])
    out_img=np.zeros((img_mask_np.shape[0],img_mask_np.shape[1],3))
    for i in range(img_mask_np.shape[0]):
        for j in range(img_mask_np.shape[1]):
            out_img[i,j]=colormap[np.argmax(img_mask_np[i,j])]
    return out_img

if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=1, n_classes=2)

    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))

        img = cv2.imread(fn,1)[...,::-1]
        if img.size[0] < img.size[1]:
            print("Error: image height larger than the width")

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_dense_crf= not args.no_crf,
                           use_gpu=not args.cpu)

        if args.viz:
            print("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])

            print("Mask saved to {}".format(out_files[i]))
