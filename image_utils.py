import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import filters, io, transform, exposure
import cv2
import os
import re

def add_saliency_to_image(saliency, image, scale_factor = 3, saliency_factor = 2):
    ''' creates a edge version of an image and adds the saliency map over this image
    :param saliency:
    :param image:
    :param scale_factor:
    :param saliency_factor:
    :return:
    '''
    image = np.squeeze(image)
    image = skimage.color.rgb2gray(image)
    output_shape = (image.shape[0] * scale_factor, image.shape[1] * scale_factor)
    image = transform.resize(image, output_shape, order=0, mode='reflect')
    image = filters.sobel(image)
    image = np.stack((image,image,image), axis= -1)

    saliency = transform.resize(saliency, output_shape, order=0, mode='reflect')
    zeros = np.zeros(output_shape)
    saliency = np.stack((zeros, saliency, zeros), axis=-1)

    final_image= image + (saliency * saliency_factor)
    final_image = np.clip(final_image, 0, 1)

    return  final_image

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

def generate_video(image_folder, out_path, name="video.mp4"):
    ''' creates a video from all images in a folder
    :param image_folder: folder containing the images
    :param out_path: output folder for the video
    :param name: name of the output video
    :return: nothing, but saves the video in the given path
    '''
    images = [img for img in os.listdir(image_folder)]
    images = natural_sort(images)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = 60
    height, width, layers = 420,320,3
    video = cv2.VideoWriter(out_path + name, fourcc, fps, (width,height))


    for image in images:
        try:
            i = cv2.imread(os.path.join(image_folder, image))
            i = cv2.resize(i, (width,height))
        except Exception as e:
            print(e)
            print('Try next image.')
            continue
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


