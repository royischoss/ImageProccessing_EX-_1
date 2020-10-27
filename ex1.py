import numpy as np
from imageio import imread, imwrite
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :],
               np.array([255] * 6)[None, :]])
grad = np.tile(x, (256, 1))


def read_image(filename, representation):
    image = imread(filename)
    if representation == 1:
        image_float = rgb2gray(image)
    else:
        image_float = image.astype(np.float64)
    image_mat = np.array(image_float)
    image_mat /= 255
    my_vec = image_mat[:,:,0]
    return image_mat


def imdisplay(filename, representation):
    my_mat = read_image(filename, representation)
    if representation == 1:
        plt.imshow(my_mat, cmap=plt.cm.gray)
    else:
        plt.imshow(my_mat)
    plt.show()


# imdisplay("C:\\Users\\Roy\PycharmProjects\\ex1IP\\monkey.JPG", 2)

def rgb2yiq(imRGB):
    mat = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])


