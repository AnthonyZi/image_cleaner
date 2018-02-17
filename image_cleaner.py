#!/usr/bin/env python3
import skimage.io as skiio
import numpy as np
import sys
import os
import cv2
import skimage.morphology as skimo
import skimage.transform as skit
import skimage.exposure as skie
import scipy.ndimage as scipynd
import argparse
import warnings


parser = argparse.ArgumentParser(description="clean document-photographs")

parser.add_argument('files', metavar="INPUT_IMAGES", type=str, nargs='+', help="input-images to clean")
parser.add_argument('-t','--threshold', metavar="THRESHOLD", type=float, default=0.80, help="grayscale-threshold to binarise image")
parser.add_argument('-g','--grayscale', help="converts image to grayscale", action="store_true")
parser.add_argument('-s','--size', metavar="OUTPUTWIDTH", type=int, default=2000, help="specify output-width of clean image")
parser.add_argument('-e','--enhance_text', help="activates automatic text-enhancement", action="store_true")


#t_y = np.round(np.array("0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000160 0.000568 0.000993 0.001436 0.001896 0.002374 0.002871 0.003386 0.003920 0.004473 0.005045 0.005636 0.006248 0.006879 0.007530 0.008202 0.008894 0.009607 0.010342 0.011098 0.011875 0.012675 0.013496 0.014340 0.015206 0.016096 0.017008 0.017944 0.018903 0.019886 0.020894 0.021925 0.022981 0.024062 0.025168 0.026299 0.027456 0.028639 0.029847 0.031082 0.032343 0.033631 0.034946 0.036289 0.037658 0.039056 0.040481 0.041935 0.043417 0.044928 0.046467 0.048036 0.049634 0.051262 0.052920 0.054608 0.056327 0.058076 0.059855 0.061666 0.063509 0.065383 0.067289 0.069226 0.071197 0.073200 0.075235 0.077304 0.079406 0.081542 0.083711 0.085914 0.088152 0.090424 0.092731 0.095073 0.097450 0.099863 0.102312 0.105469 0.108485 0.112487 0.117438 0.123303 0.130046 0.137631 0.146023 0.155186 0.165085 0.175684 0.186947 0.198838 0.211323 0.224365 0.237929 0.251979 0.266479 0.281394 0.296688 0.312326 0.328272 0.344490 0.360945 0.377601 0.394422 0.411373 0.428418 0.445312 0.465128 0.489724 0.518183 0.549591 0.583030 0.617584 0.652338 0.686376 0.718782 0.748639 0.775031 0.797044 0.813759 0.824219 0.831251 0.838081 0.844711 0.851144 0.857385 0.863436 0.869300 0.874981 0.880482 0.885805 0.890956 0.895935 0.900748 0.905396 0.909884 0.914214 0.918390 0.922414 0.926291 0.930024 0.933615 0.937068 0.940385 0.943572 0.946630 0.949562 0.952373 0.955065 0.957641 0.960105 0.962460 0.964709 0.966856 0.968903 0.970854 0.972712 0.974481 0.976163 0.977761 0.979280 0.980722 0.982091 0.983389 0.984621 0.985788 0.986895 0.987945 0.988940 0.989884 0.990781 0.991634 0.992445 0.993218 0.993957 0.994664 0.995342 0.995996 0.996628 0.997241 0.997839 0.998425 0.999002 0.999573 1.000000".split(" "), dtype=np.float)*255).astype(np.uint8)
t_y = np.round(np.array("0.000000 0.000536 0.001073 0.001611 0.002149 0.002689 0.003231 0.003775 0.004322 0.004871 0.005423 0.005979 0.006539 0.007103 0.007672 0.008246 0.008825 0.009410 0.010001 0.010598 0.011202 0.011813 0.012431 0.013058 0.013692 0.014335 0.014987 0.015648 0.016319 0.017000 0.017691 0.018392 0.019105 0.019829 0.020565 0.021312 0.022073 0.022846 0.023632 0.024431 0.025245 0.026073 0.026915 0.027772 0.028644 0.029532 0.030437 0.031357 0.032294 0.033248 0.034220 0.035209 0.036216 0.037242 0.038287 0.039351 0.040434 0.041538 0.042661 0.043805 0.044970 0.046157 0.047365 0.048595 0.049847 0.051122 0.052421 0.053742 0.055088 0.056458 0.057852 0.059271 0.060715 0.062185 0.063681 0.065203 0.066751 0.068327 0.069930 0.071561 0.073219 0.074907 0.076622 0.078368 0.080142 0.081946 0.083781 0.085646 0.087542 0.089469 0.091428 0.093419 0.095442 0.097498 0.099587 0.101709 0.103865 0.106055 0.108280 0.110539 0.112833 0.115163 0.117529 0.119931 0.122370 0.124845 0.127358 0.129909 0.132497 0.135124 0.137790 0.140495 0.143239 0.146023 0.148847 0.151711 0.154616 0.157563 0.160551 0.163581 0.166653 0.169768 0.172926 0.176127 0.179372 0.182661 0.185994 0.189372 0.192795 0.196264 0.199778 0.203339 0.206946 0.210600 0.214301 0.218050 0.221847 0.225692 0.229586 0.233528 0.237521 0.241562 0.245654 0.250000 0.254547 0.259828 0.265815 0.272478 0.279788 0.287717 0.296234 0.305311 0.314919 0.325030 0.335612 0.346639 0.358080 0.369907 0.382090 0.394601 0.407411 0.420489 0.433808 0.447338 0.461050 0.474915 0.488904 0.502988 0.517138 0.531325 0.545520 0.559693 0.573816 0.587859 0.601794 0.615591 0.629221 0.642656 0.655866 0.668822 0.681495 0.693856 0.705876 0.717526 0.728777 0.739600 0.749965 0.759843 0.769207 0.778026 0.786271 0.793913 0.800781 0.807397 0.813847 0.820133 0.826260 0.832228 0.838042 0.843703 0.849214 0.854578 0.859798 0.864877 0.869816 0.874619 0.879288 0.883826 0.888236 0.892520 0.896682 0.900723 0.904646 0.908454 0.912151 0.915737 0.919217 0.922592 0.925866 0.929041 0.932120 0.935105 0.937999 0.940805 0.943526 0.946163 0.948721 0.951201 0.953606 0.955939 0.958203 0.960400 0.962532 0.964604 0.966616 0.968573 0.970475 0.972328 0.974132 0.975890 0.977606 0.979282 0.980921 0.982525 0.984097 0.985639 0.987155 0.988647 0.990117 0.991569 0.993005 0.994428 0.995839 0.997243 0.998642 1.000000".split(" "), dtype=np.float)*255).astype(np.uint8)
t_x = np.arange(256)


sort_idx = np.argsort(t_x)
def text_enhancing_point_transform(input_image):
    idx = np.searchsorted(t_x, input_image, sorter = sort_idx)
    return t_y[sort_idx][idx]


sqrt2 = 1.4142135623730951454746218587388284504413604736328125

processing_width = 500


def find_corners(binary_image):
    min_val = binary_image.min()
    max_val = binary_image.max()
    img_thresh = min_val+(threshold*(max_val-min_val))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        binary_image = skie.equalize_adapthist(binary_image, kernel_size=processing_width/50)
    binary_image = np.where(binary_image>img_thresh,1,0)

    binary_image = skimo.binary_erosion(binary_image, skimo.disk(processing_width/200))
    binary_image = skimo.binary_dilation(binary_image, skimo.disk(processing_width/200))

    sx = scipynd.sobel(binary_image, axis=0, mode="constant")
    sy = scipynd.sobel(binary_image, axis=1, mode="constant")
    binary_image_edges = np.hypot(sx,sy)
    binary_image_edges = np.where(binary_image_edges > 0.0,1,0)

    h,w = binary_image_edges.shape
    tri_size_w = int(w/2)
    tri_size_h = int(h/2)

    ci_ur = np.triu_indices(w,tri_size_w)
    ci_ul = [ci_ur[0],w-1-ci_ur[1]]
    ci_ll = np.tril_indices(h,-(h-tri_size_w))
    ci_lr = [ci_ll[0],w-1-ci_ll[1]]

    corner_mask = np.zeros_like(binary_image_edges)
    corner_mask[ci_ul] = 1
    binary_image_ul = binary_image_edges*corner_mask
    white_points_ul = np.where(binary_image_ul==1)
    white_positions_ul = np.column_stack([white_points_ul[1],white_points_ul[0]])
    ul_positions = np.array(white_positions_ul)+1
    ul = np.sum(ul_positions**2, axis=1)
    min_ul = np.argmin(ul)
    topl = ul_positions[min_ul]-1


    corner_mask[:] = 0
    corner_mask[ci_ur] = 1
    binary_image_ur = binary_image_edges*corner_mask
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
    binary_image_ll = binary_image_edges*corner_mask
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
    binary_image_lr = binary_image_edges*corner_mask
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
    text_enhancement = args.enhance_text

    a4_size = (a4_width,int(a4_width*sqrt2+0.5))
    wa4,ha4 = a4_size

    p_size = (processing_width, int(processing_width*sqrt2+0.5))
    p_size_w,p_size_h = p_size


    endings = [".jpg", ".png", ".gif", ".jpeg"]
    images_list = [i for i in images_list if any(e in i for e in endings)]
    images_list = sorted(images_list)

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
#            binary_image = np.amax(binary_image, axis=2)

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

        if grayscale:
            print("to_grayscale", end=" - ", flush=True)
            if len(image.shape) > 2:
                image = image.mean(axis=2)

        print("extend_contrast", end=" - ", flush=True)
        image = np.array(image, dtype=np.float)
        valmin,valmax = image.min(),image.max()
        image = np.round((image-valmin)/(valmax-valmin)*255).astype(np.uint8)
        if text_enhancement:
            print("enhance_text", end=" - ", flush=True)
            image = text_enhancing_point_transform(image)

        save_head,save_tail = os.path.split(filename)
        save_tail = "clean_{}".format(save_tail)
        save_filename = os.path.join(save_head,save_tail)
        print("save {}".format(save_filename))
#        skiio.imsave(save_filename, image)
