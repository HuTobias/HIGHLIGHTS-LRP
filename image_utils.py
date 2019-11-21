import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import filters, io, transform, exposure

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