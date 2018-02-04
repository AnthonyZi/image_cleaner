#!/usr/bin/env python3
import skimage.io as skiio
import numpy as np
import sys
import os
import cv2
import skimage.morphology as skimo
import skimage.transform as skit
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="clean document-photographs")

parser.add_argument('files', metavar="INPUT_IMAGES", type=str, nargs='+', help="input-images to clean")
parser.add_argument('-t','--threshold', metavar="THRESHOLD", type=float, default=0.9, help="grayscale-threshold to binarise image")
parser.add_argument('-g','--grayscale', help="converts image to grayscale", action="store_true")
parser.add_argument('-s','--size', metavar="OUTPUTWIDTH", type=int, default=2000, help="specify output-width of clean image")


sqrt2 = 1.4142135623730951454746218587388284504413604736328125


def find_corners(binary_image):
    h,w = binary_image.shape
    tri_size_w = int(w/2)
    tri_size_h = int(h/2)

    ci_ur = np.triu_indices(w,tri_size_w)
    ci_ul = [ci_ur[0],w-1-ci_ur[1]]
    ci_ll = np.tril_indices(h,-(h-tri_size_w))
    ci_lr = [ci_ll[0],w-1-ci_ll[1]]

    corner_mask = np.zeros_like(binary_image)
    corner_mask[ci_ul] = 1
    binary_image_ul = binary_image*corner_mask
    white_points_ul = np.where(binary_image_ul==1)
    white_positions_ul = np.column_stack([white_points_ul[1],white_points_ul[0]])
    ul_positions = np.array(white_positions_ul)+1
    ul = np.sum(ul_positions**2, axis=1)
    min_ul = np.argmin(ul)
    topl = ul_positions[min_ul]-1

    corner_mask[:] = 0
    corner_mask[ci_ur] = 1
    binary_image_ur = binary_image*corner_mask
    white_points_ur = np.where(binary_image_ur==1)
    white_positions_ur = np.column_stack([white_points_ur[1],white_points_ur[0]])
    ur_positions = np.array(white_positions_ur)+1
    ur_positions = np.array(white_positions_ur)
    ur_positions[:,0] = w - ur_positions[:,0]
    ur_positions = ur_positions+1
    ur = np.sum(ur_positions**2, axis=1)
    min_ur = np.argmin(ur)
    topr = ur_positions[min_ur]-1
    topr[0] = w - topr[0]

    corner_mask[:] = 0
    corner_mask[ci_ll] = 1
    binary_image_ll = binary_image*corner_mask
    white_points_ll = np.where(binary_image_ll==1)
    white_positions_ll = np.column_stack([white_points_ll[1],white_points_ll[0]])
    ll_positions = np.array(white_positions_ll)+1
    ll_positions = np.array(white_positions_ll)
    ll_positions[:,1] = h - ll_positions[:,1]
    ll_positions = ll_positions+1
    ll = np.sum(ll_positions**2, axis=1)
    min_ll = np.argmin(ll)
    botl = ll_positions[min_ll]-1
    botl[1] = h - botl[1]

    corner_mask[:] = 0
    corner_mask[ci_lr] = 1
    binary_image_lr = binary_image*corner_mask
    white_points_lr = np.where(binary_image_lr==1)
    white_positions_lr = np.column_stack([white_points_lr[1],white_points_lr[0]])
    lr_positions = np.array(white_positions_lr)+1
    lr_positions = np.array(white_positions_lr)
    lr_positions[:,0] = w - lr_positions[:,0]
    lr_positions[:,1] = h - lr_positions[:,1]
    lr_positions = lr_positions+1
    lr = np.sum(lr_positions**2, axis=1)
    min_lr = np.argmin(lr)
    botr = lr_positions[min_lr]-1
    botr[0] = w - botr[0]
    botr[1] = h - botr[1]


    return topl,topr,botl,botr





if __name__ == "__main__":
    args = parser.parse_args()

    a4_width = args.size
    grayscale = args.grayscale
    threshold = args.threshold
    images_list = args.files

    a4_size = (a4_width,int(a4_width*sqrt2+0.5))
    wa4,ha4 = a4_size

    processing_width = 500
    p_size = (processing_width, int(processing_width*sqrt2+0.5))
    p_size_w,p_size_h = p_size


    endings = [".jpg", ".png", ".gif"]
    images_list = [i for i in images_list if any(e in i for e in endings)]

    for filename in images_list:
        print(filename, end=" - ", flush=True)
        original_image = np.array(skiio.imread(filename), dtype=np.uint8)
        h,w = original_image.shape[:2]
        h_factor = h/p_size_h
        w_factor = w/p_size_w

        print("find_corners", end=" - ", flush=True)
        binary_image = np.array(original_image, dtype=np.uint8)
        binary_image = skit.resize(binary_image, (p_size_h,p_size_w), mode="constant")
        if len(binary_image) >2:
            binary_image = binary_image.mean(axis=2)

        topl,topr,botl,botr = find_corners(binary_image)
        topl = (topl[0]*w_factor, topl[1]*h_factor)
        topr = (topr[0]*w_factor, topr[1]*h_factor)
        botl = (botl[0]*w_factor, botl[1]*h_factor)
        botr = (botr[0]*w_factor, botr[1]*h_factor)

        print("transform", end=" - ", flush=True)
        points_is = np.array([topl,topr,botl,botr], dtype=np.float32)
        points_shall = np.array([[0,0], [wa4,0], [0,ha4], [wa4,ha4]], dtype=np.float32)

        warp_mat = cv2.getPerspectiveTransform(points_is, points_shall)

        image = cv2.warpPerspective(original_image, warp_mat, a4_size, flags=cv2.INTER_NEAREST)

        print("extend_contrast", end=" - ", flush=True)
        image = np.array(image, dtype=np.float)
        valmax = image.max()
        valmin = image.min()
        image = image-valmin
        image = image/(valmax-valmin)*255
        image = np.array(image, dtype=np.uint8)

        if grayscale:
            print("to_grayscale", end=" - ", flush=True)
            if len(image.shape) > 2:
                image = image.mean(axis=2)


        save_head,save_tail = os.path.split(filename)
        save_tail = "clean_{}".format(save_tail)
        save_filename = os.path.join(save_head,save_tail)
        print("save {}".format(save_filename))
#        skiio.imsave(save_filename, image)

        plt.imshow(image)
        plt.show()
