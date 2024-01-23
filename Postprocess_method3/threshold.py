import os
import cv2
from pathlib import Path

def last_3chars(x):
    """
    sort files according to names
    """
    x = os.path.splitext(x)[0]
    # print(os.path.splitext(x)[0])
    x = x.split('_')[-1]
    # print(x)
    return(int(x))

image_folder = '/HOMES/yigao/Documents/Below_knee/go_upstairs/1/depth_threshold/'

images = [img for img in sorted(os.listdir(image_folder), key=last_3chars)]         #if img.endswith(".png")
print("images: " , images)
for f in images:
        print("f: ", f)
        black_holes = 0
        i = cv2.imread(image_folder + f, cv2.IMREAD_UNCHANGED)
        width, height = i.shape
        all_pixels = []
        for x in range(width):
            for y in range(height):
                cpixel = i[x, y]
                # print("cpixel: ", cpixel)
                if cpixel == 0.0:
                    black_holes = black_holes + 1
                    # print("black_holes: ", black_holes)
                all_pixels.append(cpixel)
        print("procent of black holes in this image:  {}%".format(black_holes/len(all_pixels)*100))
        rgb_path = Path(image_folder + f).parent.parent.absolute()
        if black_holes/len(all_pixels)*100 >= 1:
            os.remove(image_folder + f)
            os.remove(str(rgb_path) + "/rgb_threshold/" + "color_" + f.split("_")[1].split(".")[0] + ".jpg")
