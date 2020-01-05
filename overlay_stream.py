import os
import image_utils
import cv2
import numpy as np
import stream_generator

image_folder = "stream_500k/screen"
raw_argmax_base = "stream_500k/raw_argmax/raw_argmax"
save_folder = "stream_500k/test"

images = [img for img in os.listdir(image_folder)]
images = image_utils.natural_sort(images)

def interpolate(array1, array2, t):
    '''
    linear interpolation between two frames of a state
    :param array1: starting array
    :param array2: end array
    :param t: time parameter, goes from -1 to 3 ( 0=0.25, 3=1 in the normal interpolation formula)
    :return: the interpolated array
    '''
    t = (t * 0.25) + 0.25
    return (array2 * t) + (array1 * (1-t))




if __name__ == "__main__":

    old_saliency_map = None
    for image in images:
        try:
            image_str = image.split('_')
            state_index = int(image_str[1])
            frame_index = int(image_str[2].replace(".png",""))

            i = cv2.imread(os.path.join(image_folder, image))
            i = cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
            # image_utils.show_image(i)
            saliency_filename = raw_argmax_base + "_" + str(state_index) + ".npy"
            saliency_map = np.load(saliency_filename)
            saliency_map = image_utils.normalise_image(saliency_map)
            if old_saliency_map is not None:
                saliency_map = interpolate(old_saliency_map,saliency_map, frame_index)
            saliency = image_utils.output_saliency_map(saliency_map[:, :, 3], i, edges=False)
            index = str(state_index)+ '_' + str(frame_index)
            stream_generator.save_frame(saliency, save_folder+"/argmax_smooth/argmax", index)
            # c0 = image_utils.saliency_in_channel(saliency_map[:,:,3], i, channel=0)
            # c1 = image_utils.saliency_in_channel(saliency_map[:,:,3], i, channel=1)
            # c2 = image_utils.saliency_in_channel(saliency_map[:,:,3], i, channel=2)
            # stream_generator.save_frame(c0,save_folder + "/c0/c0", index)
            # stream_generator.save_frame(c1, save_folder + "/c1/c1", index)
            # stream_generator.save_frame(c2, save_folder + "/c2/c2", state_index)
            if frame_index == 3:
                old_saliency_map = saliency_map
        except Exception as e:
            print(e)
            print('Try next image.')
            continue
