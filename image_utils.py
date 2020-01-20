import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import filters, io, transform, exposure
import cv2
import os
import re

def add_saliency_to_image(saliency, image, saliency_brightness = 2):
    '''
    adds a saliency map(in green) over a given image
    :param saliency: the saliency map to be applied
    :param image: the original image
    :param saliency_brightness: the brightness of the saliency map
    :return: the overlayed image
    '''

    image_shape =(image.shape[0],image.shape[1])
    saliency = transform.resize(saliency, image_shape, order=0, mode='reflect')
    zeros = np.zeros(image_shape)
    saliency = np.stack((zeros, saliency, zeros), axis=-1)
    saliency *= saliency_brightness
    final_image = image + saliency
    final_image = np.clip(final_image, 0, 1)
    return final_image

def create_edge_image(image, output_shape = None):
    ''' creates a edge version of an image
    :param image: the original image
    :return: edge only version of the image
    '''
    image = skimage.color.rgb2gray(image)
    image = filters.sobel(image)
    image = np.stack((image, image, image), axis=-1)
    return image

def output_saliency_map(saliency, image, scale_factor = 3, saliency_factor = 2, edges = True):
    ''' scales the image and adds the saliency map
    :param saliency:
    :param image:
    :param scale_factor: factor to scale height and width of the image
    :param saliency_factor:
    :param edges: if True, creates a edge version of the image first
    :return:
    '''
    image = np.squeeze(image)
    output_shape = (image.shape[0] * scale_factor, image.shape[1] * scale_factor)
    image = transform.resize(image, output_shape, order=0, mode='reflect')
    if edges:
        image = create_edge_image(image, output_shape)

    final_image = add_saliency_to_image(saliency, image, saliency_factor)

    return final_image

def saliency_in_channel(saliency, image, scale_factor = 3, saliency_brightness = 2, channel = 1):
    '''
    Ressizes image and adds saliency
    :param saliency:
    :param image:
    :param scale_factor:
    :param saliency_factor:
    :return:
    '''
    image = np.squeeze(image)
    output_shape = (image.shape[0] * scale_factor, image.shape[1] * scale_factor)
    image = transform.resize(image, output_shape, order=0, mode='reflect')
    saliency = transform.resize(saliency, output_shape, order=0, mode='reflect')
    saliency = saliency * saliency_brightness
    image[:,:,channel] = saliency
    image = np.clip(image, 0, 1)

    return image

def normalise_image(image):
    '''normalises image by forcing the min and max values to 0 and 1 respectively
     :param image: the input image
    :return: normalised image as numpy array
    '''
    try:
        image = np.asarray(image)
    except:
        print('Cannot convert image to array')
    image = image - image.min()
    if image.max() != 0:
        image = image / image.max()
    return image

def show_image(image):
    '''shortcut to show an image with matplotlib'''
    plt.imshow(image)
    plt.show()

def crop_image_button(image, part = 0.18 ):
    '''
    cuts the lower *part* from an image
    :param image: the image to be cropped
    :param part: part of the image to be cut (expects floats between 0 and 1)
    :return: the cropped image
    '''
    image_length = image.shape[0]
    pixels = int(image_length * part)
    crop = image[0:image_length - pixels,:, :]
    return crop


def generate_video(image_folder, out_path, name="video.mp4", image_indices=None, crop_images = True):
    ''' creates a video from images in a folder
    :param image_folder: folder containing the images
    :param out_path: output folder for the video
    :param name: name of the output video
    :param image_indices: states to be included in the summary video
    :return: nothing, but saves the video in the given path
    '''
    images = [img for img in os.listdir(image_folder)]
    images = natural_sort(images)
    fourcc = cv2.VideoWriter_fourcc(*'H264') #important for browser support, MP4V is not working with browsers
    fps = 30
    height, width, layers = 420,320,3
    if not (os.path.isdir(out_path)):
        os.makedirs(out_path)
    video = cv2.VideoWriter(out_path + name, fourcc, fps, (width,height))
    old_state_index = None
    black_frame = np.zeros((height, width, layers),np.uint8)
    black_frame_number = int(fps)

    for image in images:
        to_write = False
        try:
            image_str = image.split('_')
            state_index = int(image_str[1])
            if (state_index in image_indices) or (image_indices is None):

                #check if the states are successive and insert black frames, if the are not
                if old_state_index != None and state_index != old_state_index + 1 and state_index != old_state_index:
                    for n in range(black_frame_number):
                        video.write(black_frame)
                old_state_index = state_index

                i = cv2.imread(os.path.join(image_folder, image))
                if crop_images:
                    i = crop_image_button(i)
                i = cv2.resize(i, (width,height))
                to_write = True
        except Exception as e:
            print(e)
            print('Try next image.')
            continue
        if to_write:
            video.write(i)

    cv2.destroyAllWindows()
    video.release()

def natural_sort( l ):
    """ Sort the given list in natural sort (the way that humans expect).
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l


#test = cv2.imread('stream_1M/test/c0/c0_4_0.png')
#test = crop_image_button(test, 0.18)
#show_image(test)

#pass



