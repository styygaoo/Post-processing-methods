import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
from sklearn.preprocessing import MinMaxScaler
import time
import cv2

"""
Note:
cv.imread() will internally convert from rgb to bgr, and cv.imwrite() will do the opposite
"""

# configs to extract frames from bag file
scaler = MinMaxScaler()
value_min = 0.1
value_max = 4

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file("/HOMES/yigao/Documents/go_downstairs_first_recording/20240119_162245.bag", repeat_playback=False)
profile = pipe.start(cfg)
playback = profile.get_device().as_playback()
playback.set_real_time(False)

# set the output directory
output_depth = "/HOMES/yigao/Documents/Below_knee/go_upstairs/5/depth_tiff/"
output_depth_png = "/HOMES/yigao/Documents/Below_knee/go_upstairs/5/depth_png/"
output_color = "/HOMES/yigao/Documents/Below_knee/go_upstairs/5/rgb/"
# output_loc = "/HOMES/yigao/Downloads/eval_testset/eval_test/"           # for GDM

# extract the scaling factor
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()


# configs in order to colorize the depth maps
colorizer = rs.colorizer()
# colorizer.set_option(rs.option.color_scheme, 9) # 0 - Jet 1 - Classic 2 - WhiteToBlack3 - BlackToWhite4 - Bio 5 - Cold 6 - Warm 7 - Quantized8 - Pattern 9 - Hue
# colorizer.set_option(rs.option.visual_preset, 0) # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
# colorizer.set_option(rs.option.min_distance, value_min)
# colorizer.set_option(rs.option.max_distance, value_max)

# declare align method, decimation filter, spatial filter, temporal filter, hole filling filter and transformations
align = rs.align(rs.stream.color)

decimation = rs.decimation_filter(2)

spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 2)
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial.set_option(rs.option.filter_smooth_delta, 20)
spatial.set_option(rs.option.holes_fill, 0)     # performs much better than another hole filling filter!

temporal = rs.temporal_filter()
temporal.set_option(rs.option.holes_fill, 3)
temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
temporal.set_option(rs.option.filter_smooth_delta, 20)

hole_filling = rs.hole_filling_filter(1)      # [0-2]

depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)

align = rs.align(rs.stream.color)

count = 0
start_time = time.time()

# Streaming loop to extracting frames
try:
    while (time.time() - start_time) < 5000:
        # get frameset of color and depth
        frames = pipe.wait_for_frames()
        frames.keep()

        # apply various filters accordingly
        # frames = decimation.process(frames)
        # frames = depth_to_disparity.process(frames)
        # frames = spatial.process(frames)
        # frames = temporal.process(frames)
        # frames = disparity_to_depth.process(frames).as_frameset()
        # frames = hole_filling.process(frames).as_frameset()

        # align the depth frame to color frame
        frames = align.process(frames)

        # extract depth and color frames separately.
        colorized_depth = frames.get_depth_frame()
        colorized_color = frames.get_color_frame()

        # convert video frames to array and convert the depth unit from millimeter to meter
        colorized_depth = np.asanyarray(colorized_depth.get_data())
        colorized_depth = depth_scale * colorized_depth
        colorized_color = np.asanyarray(colorized_color.get_data())

        # colorization depth maps
        # colorized_depth = np.asanyarray(colorizer.colorize(colorized_depth).get_data())

        # normalization settings
        # norm = np.zeros((800, 800))
        # colorized_depth = cv2.normalize(colorized_depth, colorized_depth, 0.1, 10, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # scaling settings
        # colorized_depth = img_as_float(exposure.rescale_intensity(colorized_depth, out_range=(0.1, 10))).astype(np.float32)
        # scaler.fit(colorized_depth)
        # colorized_depth = scaler.transform(colorized_depth)


        # plot depth maps
        # plt.imshow(colorized_color)
        # plt.show()

        # cv2.imwrite(output_depth + "depth_" + "%#05d" % (count + 1) + ".tiff", colorized_depth)
        # plt.imsave(output_color + "color_" + "%#05d" % (count + 1) + ".jpg", colorized_color)
        # plt.imsave(output_depth_png + "depth_" + "%#05d" % (count + 1) + ".png", colorized_depth, cmap='gray')

        # save the rgb-depth pairs into .npy format
        # np.savez_compressed(output_loc + "/%#05d" % (count+1), depth=colorized_depth, image=colorized_color)
        count += 1
finally:
    print('This is always executed')
