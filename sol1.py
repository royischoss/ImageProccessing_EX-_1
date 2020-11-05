##############################################################################
# This python file is ex1 in image processing course.
# the next script is basic functions for reading and transforming images
# and editing their grayscale deviation for different needs.
##############################################################################

import numpy as np
from imageio import imread, imwrite
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


def read_image(filename, representation):
    """
    The next lines preform a image read to a matrix of numpy.float64 using
    imagio and numpy libraries.
    :param filename: a path to jpg image we would like to read.
    :param representation: 1 stands for grayscale , 2 for RGB.
    :return: image_mat - a numpy array represents the photo as described above.
    """
    image = imread(filename)
    if representation == 1:
        image_mat = np.array(rgb2gray(image))
    else:
        image_mat = np.array(image.astype(np.float64))
        image_mat /= 255
    return image_mat


def imdisplay(filename, representation):
    """
    The next lines preform image display using matplotlib.pyplot and read_image
     function from above.
    :param filename:a path to jpg image we would like to read.
    :param representation: 1 stands for grayscale , 2 for RGB.
    """
    my_mat = read_image(filename, representation)
    if representation == 1:
        plt.imshow(my_mat, cmap=plt.cm.gray)
    else:
        plt.imshow(my_mat)
    plt.show()


def rgb2yiq(imRGB):
    """
    The next lines preform transformation between a matrix represent an image
    in RGB format to a matrix represent the same image in YIQ format.
    :param imRGB: a numpy array represent an image in RGB format.
    :return: image_mat - a numpy array represent the same image in YIQ format.
    """
    r = imRGB[:, :, 0]
    g = imRGB[:, :, 1]
    b = imRGB[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    i = 0.596 * r + -0.275 * g + -0.321 * b
    q = 0.212 * r + -0.523 * g + 0.311 * b
    image_mat = np.array([y, i, q])
    image_mat = np.moveaxis(image_mat, 0, -1)
    return image_mat


def yiq2rgb(imYIQ):
    """
    The next lines preform transformation between a matrix represent an image
    in YIQ format to a matrix represent the same image in RGB format.
    :param imYIQ: a numpy array represent an image in YIQ format.
    :return: image_mat - a matrix represent the same image in RGB format.
    """
    y = imYIQ[:, :, 0]
    i = imYIQ[:, :, 1]
    q = imYIQ[:, :, 2]
    r = 1.000 * y + 0.956 * i + 0.620 * q
    g = 1.000 * y + -0.272 * i + -0.647 * q
    b = 1.000 * y + -1.108 * i + 1.705 * q
    image_mat = np.array([r, g, b])
    image_mat = np.moveaxis(image_mat, 0, -1)
    return image_mat


def histogram_equalize(im_orig):
    """
    The next lines preform histogram equalization on a given matrix represent
    of a photo.
    :param im_orig: a numpy array represent of a photo
    :return: [im_eq, hist_orig, hist_eq] : a list includes in the equalized
    photo , the original histogram and the new equalized histogram.
    """
    dim = len(im_orig.shape)
    # dim check for representation check.
    if dim == 3:
        yiq_im = rgb2yiq(im_orig)
        im_to_process = np.copy(yiq_im[:, :, 0])
    else:
        im_to_process = np.copy(im_orig)
    im_to_process *= 255  # multiplying by 255 for us not to round values to
    # zero in the look up table initializing.
    hist_orig, bounds2 = np.histogram(im_to_process, 256)
    hist_cs = np.cumsum(hist_orig)
    m = np.nonzero(hist_cs)[0][0]
    T = np.round(255 * (hist_cs - hist_cs[m]) /
                 (hist_cs[255] - hist_cs[m]))  # initializing the look-up table
    # in this step we take care we go from 0-255 ,we normalize and we
    # multiply by (z - 1) value.
    T = T.astype(int)
    if dim == 3:
        im_eq = np.copy(yiq_im)
        im_to_process = T[im_to_process.astype(int)]
        im_eq[:, :, 0] = np.copy(im_to_process)
        hist_eq, bounds = np.histogram(im_to_process, 256)
        im_eq = yiq2rgb(im_eq)
    else:
        im_eq = (T[im_to_process.astype(int)])
        hist_eq = np.histogram(im_eq, 256)  # taking the equalized histogram.
    im_eq = im_eq / 255  # dividing so we get [0,1] matrix.
    return [im_eq, hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):
    """
    The next lines preform image quantization by minimizing the error.
    :param im_orig: a numpy array represent of a photo
    :param n_quant: the number of new grayscale levels.
    :param n_iter: the number of max iterations for minimizing the error.
    :return: [im_quant, error] : a list holding the image matrix (numpy array)
     after quantization and a numpy.array of errors calculated.
    """
    dim = len(im_orig.shape)
    # dim check for representation check.
    if dim == 3:
        yiq_im = rgb2yiq(im_orig)
        im_to_process = yiq_im[:, :, 0]
    else:
        im_to_process = im_orig[:]
    im_to_process *= 255
    hist_orig, bounds = np.histogram(im_to_process, 256)
    hist_cum = np.cumsum(hist_orig)
    z = np.zeros(n_quant + 1)
    z[0] = -1
    z[n_quant] = 255
    q = np.zeros(n_quant)
    # initialization :
    for i in range(1, n_quant + 1):
        indices_array = np.where(hist_cum > i * (hist_cum[255] / n_quant))
        if i != n_quant:
            z[i] = indices_array[0][0]
        up_sum = np.sum(np.dot(hist_orig[int(z[i - 1]) + 1:int(z[i]) + 1],
                               np.arange(int(z[i - 1]) + 1, int(z[i]) + 1).T))
        dw_sum = np.sum(hist_orig[int(z[i - 1]) + 1:int(z[i] + 1)])
        q[i - 1] = up_sum / dw_sum
    k = 0
    error = np.zeros(0)
    z_new = np.zeros(n_quant + 1)
    z_new[0] = -1
    z_new[n_quant] = 255
    # start of iterations:
    while k < n_iter:
        error_sum = 0
        for i in range(1, n_quant + 1):
            if i != n_quant:
                z_new[i] = (q[i - 1] + q[i]) / 2
            else:
                z_new[i] = z[i]
            up_sum = np.sum(np.dot(hist_orig[int(z_new[i - 1]) + 1:int(z_new[i]) + 1],
                                   np.arange(int(z_new[i - 1]) + 1,
                                             int(z_new[i]) + 1).T))
            dw_sum = np.sum(hist_orig[int(z_new[i - 1]) + 1:int(z_new[i]) + 1])
            q[i - 1] = up_sum / dw_sum
            g = np.arange(int(z_new[i - 1] + 1), int(z_new[i] + 1)).astype(np.float64)
            q_values = np.full((1, int(z_new[i]) - int(z_new[i - 1])), q[i - 1])
            error_sum += np.dot((q_values - g) ** 2,
                                hist_orig[int(z_new[i - 1]) + 1:int(z_new[i]) + 1])
        error = np.append(error, error_sum)
        k += 1
        # stopping condition:
        if np.array_equal(z, z_new):
            break
        z = np.copy(z_new)
    # creating a look up table for the grayscale levels.
    map_vector = np.zeros(256)
    q = q.astype(np.int32)
    z = z.astype(np.int32)
    for i in range(0, n_quant):
        map_vector[z[i] + 1:z[i + 1] + 1] = q[i]
    # making the change in grayscale levels on the photo.
    im_to_process = map_vector[im_to_process.astype(np.int32)]
    im_to_process /= 255
    if dim == 3:
        yiq_im[:, :, 0] = im_to_process
        im_quant = yiq2rgb(yiq_im)
    else:
        im_orig /= 255
        im_quant = im_to_process
    return [im_quant, error]

