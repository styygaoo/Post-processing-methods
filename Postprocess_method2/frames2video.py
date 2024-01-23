import cv2
import os


def last_3chars(x):
    x = os.path.splitext(x)[0]
    # print(os.path.splitext(x)[0])
    x = x.split('_')[-1]
    # print(x)
    return(int(x))

image_folder = '/HOMES/yigao/Documents/Below_knee/go_upstairs/3/depth_png'
video_name = '/HOMES/yigao/Documents/Below_knee/go_upstairs/3/depth_png.avi'


# print(sorted(os.listdir(image_folder), key=last_3chars)


images = [img for img in sorted(os.listdir(image_folder), key=last_3chars)]         #if img.endswith(".png")


# print(os.path.join(image_folder, images[0]))



frame = cv2.imread(os.path.join(image_folder, images[0]))


height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 30, (width, height))

for image in images:
    # print(image)
    qq = cv2.imread(os.path.join(image_folder, image))
    # print(qq.shape)
    video.write(cv2.imread(os.path.join(image_folder, image)))


video.release()


