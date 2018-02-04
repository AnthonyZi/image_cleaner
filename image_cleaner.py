#!/usr/bin/env python3
import skimage.io as skiio
import numpy as np
import sys
import os
import cv2
import skimage.morphology as skimo
import skimage.transform as skit
import argparse


parser = argparse.ArgumentParser(description="clean document-photographs")

parser.add_argument('files', metavar="INPUT_IMAGES", type=str, nargs='+', help="input-images to clean")
parser.add_argument('-t','--threshold', metavar="THRESHOLD", type=int, default=220, help="grayscale-threshold to binarise image")
parser.add_argument('-g','--grayscale', help="converts image to grayscale", action="store_true")
parser.add_argument('-s','--size', metavar="OUTPUTWIDTH", type=int, default=2000, help="specify output-width of clean image")


sqrt2 = 1.4142135623730951454746218587388284504413604736328125

if __name__ == "__main__":
    args = parser.parse_args()

    a4_width = args.size
    grayscale = args.grayscale
    threshold = args.threshold
    images_list = args.files

    a4_size = (a4_width,int(a4_width*sqrt2+0.5))
    wa4,ha4 = a4_size


    endings = [".jpg", ".png", ".gif"]
    images_list = [i for i in images_list if any(e in i for e in endings)]

    for filename in images_list:
        print("cleaning {}".format(filename))
        image = np.array(skiio.imread(filename), dtype=np.float)

        print("to grayscale", end=" - ", flush=True)
        if len(image.shape) > 2:
            image = image.mean(axis=2)

        h,w = image.shape

        valmax = image.max()
        valmin = image.min()
        image = image-valmin
        image = image/(valmax-valmin)*255
        image = np.array(image, dtype=np.uint8)

        bitimage = np.where(image>threshold,255,0)

        print("get paper-edges: prepare", end="...", flush=True)
        transform_bitimage = np.where(bitimage==255,1,0)
        transform_bitimage = skimo.binary_erosion(transform_bitimage, skimo.disk(5))
        transform_bitimage = skimo.binary_dilation(transform_bitimage, skimo.disk(5))

        print("get_potential_points", end="...", flush=True)
        white_points = np.where(transform_bitimage==1)
        white_positions = np.column_stack([white_points[1],white_points[0]])


        print("calc", end=" - ", flush=True)
        tl_positions = np.array(white_positions)+1

        tr_positions = np.array(white_positions)
        tr_positions[:,0] = w - tr_positions[:,0]
        tr_positions = tr_positions+1

        bl_positions = np.array(white_positions)
        bl_positions[:,1] = h - bl_positions[:,1]
        bl_positions = bl_positions+1

        br_positions = np.array(white_positions)
        br_positions[:,0] = w - br_positions[:,0]
        br_positions[:,1] = h - br_positions[:,1]
        br_positions = br_positions+1

        tl = np.sum(tl_positions*tl_positions, axis=1)
        tr = np.sum(tr_positions*tr_positions, axis=1)
        bl = np.sum(bl_positions*bl_positions, axis=1)
        br = np.sum(br_positions*br_positions, axis=1)

        min_tl = np.argmin(tl)
        min_tr = np.argmin(tr)
        min_bl = np.argmin(bl)
        min_br = np.argmin(br)

        topl = tl_positions[min_tl]-1

        topr = tr_positions[min_tr]-1
        topr[0] = w - topr[0]

        botl = bl_positions[min_bl]-1
        botl[1] = h - botl[1]

        botr = br_positions[min_br]-1
        botr[0] = w - botr[0]
        botr[1] = h - botr[1]

        quad_is = np.array([topl,topr,botl,botr], dtype=np.float32)
        quad_shall = np.array([[0,0], [wa4,0], [0,ha4], [wa4,ha4]], dtype=np.float32)


        print("cut paper", end=" - ", flush=True)
        warp_mat = cv2.getPerspectiveTransform(quad_is, quad_shall)

        if grayscale:
            image_warped = cv2.warpPerspective(bitimage, warp_mat, a4_size, flags=cv2.INTER_NEAREST)
        else:
            image_warped = cv2.warpPerspective(image, warp_mat, a4_size, flags=cv2.INTER_NEAREST)


        image = np.array(image, dtype=np.float)
        valmax = image.max()
        valmin = image.min()
        image = image-valmin
        image = image/(valmax-valmin)*255
        image = np.array(image, dtype=np.uint8)

        print("save result")
        save_head,save_tail = os.path.split(filename)
        save_tail = "clean_{}".format(save_tail)
        save_filename = os.path.join(save_head,save_tail)
        skiio.imsave(save_filename, image_warped)
