import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt


def last_3chars(x):
    x = os.path.splitext(x)[0]
    # print(os.path.splitext(x)[0])
    x = x.split('_')[-1]
    # print(x)
    return(int(x))



prefix = "/HOMES/yigao/Documents/Below_knee/go_upstairs/5"


output_loc = prefix + "/forGDM"           # dataset for GDM
# print(output_loc)

depth_folder = prefix + "/depth_threshold/"
depthes = [img for img in sorted(os.listdir(depth_folder), key=last_3chars)]
# print("images: " , depthes)

rgb_folder = prefix + "/rgb_threshold/"
rgbs = [rgb for rgb in sorted(os.listdir(rgb_folder), key=last_3chars)]
# print("rgbs: " , rgbs)


mask_folder = prefix + "/masks/"
impainted_folder = prefix + "/inpainted/"
impainted_png_folder = prefix + "/inpainted_png/"



count = 0
pixel_values_all_depth_maps_each_scene = []


for d, r in zip(depthes, rgbs):

    rgb = cv2.imread(rgb_folder + r, cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(depth_folder + d, cv2.IMREAD_UNCHANGED)
    # print(rgb)
    width, height = depth.shape
    # print(width)
    mat = np.zeros((width, height))
    for w in range(width):
        for h in range(height):
            # print(pixels[i, j])
            if depth[w, h] == 0.0:
                # print("sdasd")
                # print(pixels[i, j])
                mat[w, h] = 255

    cv2.imwrite(mask_folder + 'mask_' + str(count) + '.png', mat)


    mask = cv2.imread(mask_folder + 'mask_' + str(count) + '.png', 0) # flag 0 for grayscale, 1 for rgb
    # check
    # print(img)
    # print(mask)
    # Inpaint.


    # dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)              # cv2.INPAINT_TELEA or cv2.INPAINT_NS
    #
    # # Write the output.
    # cv2.imwrite(impainted_folder + 'depth_inpainted_' + str(count) + '.png', dst)
    #
    # count = count + 1

    # print(rgb.shape)
    # print(depth)
    # print(mask)


    scale = np.abs(depth).max()
    depth = depth.astype(np.float32) / scale  # Has to be float32, 64 not supported.

    dst = cv2.inpaint(depth, mask, 1, cv2.INPAINT_NS)
    # print(img)
    # Back to original size and value range.
    # dst = dst[1:-1, 1:-1]
    dst = dst * scale
    # print(dst)
    # print(dst.shape)
    # print(type(dst))
    cv2.imwrite(impainted_folder + 'depth_inpainted_' + str(count) + '.tiff', dst)





    """histogram"""

    for m in dst:
        for n in m:
            pixel_values_all_depth_maps_each_scene.append(n)
    # print(len(pixel_values))
    # print(max(pixel_values))
    # print(min(pixel_values))

    """histogram Freedman–Diaconis"""
    # q25, q75 = np.percentile(pixel_values, [25, 75])        #  Interquartile range
    # bin_width = 2 * (q75 - q25) * len(pixel_values) ** (-1 / 3)
    # bins = round((max(pixel_values) - min(pixel_values)) / bin_width)
    # plt.hist(pixel_values, bins=bins)
    # plt.show()

    """When the image file is read with the OpenCV function imread(), the order of colors is BGR (blue, green, red)
    """
    """colorize the gray scale depth map"""
    # im_rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    # print("np.min(dst): ", np.min(dst))
    # print("np.max(dst): ", np.max(dst))
    # normalized_image = (dst - np.min(dst)) * (255.0 / (np.max(dst) - np.min(dst)))
    # dst_png = normalized_image.astype(np.uint8)
    # print(dst_png)

    # # dst_png = cv2.applyColorMap(dst_png, cv2.COLORMAP_JET)
    # # print("dst_png.shape", dst_png.shape)
    # dst_png = Image.fromarray(dst_png)
    # # dst_png = dst_png.convert("P")
    # # dst_png.show()
    # dst_png.save(impainted_png_folder + 'depth_png_inpainted_' + str(count) + '.png')

    plt.imsave(impainted_png_folder + 'depth_png_inpainted_' + str(count) + '.png', dst, cmap='gray')
    """generate the the dataset in .npy format for GDM"""
    np.savez_compressed(output_loc + "/%#05d" % (count+1), depth=dst, image=rgb)
    count = count + 1

