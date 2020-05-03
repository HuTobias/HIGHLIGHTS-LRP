"""
    Implements the function overlay_stream(stream_directory), which overlays all frames in the given directory with the
    saliency maps stored in the given directory.
"""

import os
import image_utils
import cv2
import numpy as np
import stream_generator

def interpolate(array1, array2, t):
    '''
    linear interpolation between two frames of a state
    :param array1: starting array
    :param array2: end array
    :param t: time parameter, goes from -1 to 3 ( 0=0.25, 3=1 in the normal interpolation formula)
    :return: the interpolated array
    '''
    t = (t * 0.25) + 0.25
    return (array2 * t) + (array1 * (1 - t))


def overlay_stream(stream_directory):
    '''
    overlays all screens in the stream_directory
    :param stream_directory: see above
    :return: nothing
    '''
    stream_folder = stream_directory
    image_folder = stream_folder + "/screen"
    raw_argmax_base = stream_folder + "/raw_argmax/raw_argmax"
    save_folder = stream_folder + "/argmax_smooth"
    save_folder2 = stream_folder + "/screen_smooth"
    if not (os.path.isdir(save_folder2)):
        os.makedirs(save_folder2)

    images = [img for img in os.listdir(image_folder)]
    images = image_utils.natural_sort(images)

    old_saliency_map = None
    old_image = None
    for image in images:
        try:
            image_str = image.split('_')
            state_index = int(image_str[1])
            frame_index = int(image_str[2].replace(".png", ""))

            i = cv2.imread(os.path.join(image_folder, image))
            i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            if old_image is not None:
                smooth_i = np.maximum(old_image,i)
                old_image = i
                i = smooth_i
            else:
                old_image = i

            image_utils.save_image(os.path.join(save_folder2, image) ,i)

            saliency_filename = raw_argmax_base + "_" + str(state_index) + ".npy"
            saliency_map = np.load(saliency_filename)
            saliency_map = image_utils.normalise_image(saliency_map)
            if saliency_map.sum() > 0.9 * saliency_map.shape[0] * saliency_map.shape[1] * saliency_map.shape[2]:
                print(state_index)
                saliency_map = np.zeros(saliency_map.shape)
            if old_saliency_map is not None:
                saliency_map = interpolate(old_saliency_map, saliency_map, frame_index)
            saliency = image_utils.output_saliency_map(saliency_map[:, :, 3], i, edges=False)
            index = str(state_index) + '_' + str(frame_index)
            stream_generator.save_frame(saliency, save_folder + "/argmax", index)
            if frame_index == 3:
                old_saliency_map = saliency_map
        except Exception as e:
            print(e)
            print('Try next image.')
            continue



if __name__ == "__main__":

    stream_folder = 'stream'
    overlay_stream(stream_folder)