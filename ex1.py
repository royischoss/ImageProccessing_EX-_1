import numpy as np
from imageio import imread, imwrite
import matplotlib.pyplot as plt
from skimage.color import rgb2gray




def read_image(filename, representation):
    """"""
    image = imread(filename)
    if representation == 1:
        image_mat = np.array(rgb2gray(image))
    else:
        image_mat = np.array(image.astype(np.float64))
        image_mat /= 255
    return image_mat


def imdisplay(filename, representation):
    """"""
    my_mat = read_image(filename, representation)
    if representation == 1:
        plt.imshow(my_mat, cmap=plt.cm.gray)
    else:
        plt.imshow(my_mat)
    plt.show()


# imdisplay("C:\\Users\\Roy\PycharmProjects\\ex1IP\\monkey.JPG", 2)


def rgb2yiq(imRGB):
    """"""
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
    dim = len(im_orig.shape)
    if dim == 3:
        yiq_im = rgb2yiq(im_orig)
        im_to_process = yiq_im[:, :, 0]
    else:
        im_to_process = im_orig
    hist_orig, bounds = np.histogram(im_to_process, 256)
    im_orig *= 255 # shallow copy maybe needs change
    hist_mul, bounds2 = np.histogram(im_to_process, 256)
    hist_cs = np.cumsum(hist_mul)
    m = np.nonzero(hist_cs)[0][0]
    T = np.round(255 * (hist_cs - hist_cs[m]) /
                       (hist_cs[255] - hist_cs[m]))
    T = T.astype(int)
    if dim == 3:
        im_orig[:, :, 0] = T[im_orig[:, :, 0].astype(int)]
    else:
        im_orig = T[im_orig.astype(int)]
    im_orig = im_orig / 255
    hist_eq = np.histogram(im_orig, 256)
    return [im_orig, hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):
    dim = len(im_orig.shape)
    if dim == 3:
        yiq_im = rgb2yiq(im_orig)
        im_to_process = yiq_im[:, :, 0]
    else:
        im_to_process = im_orig[:]
    im_to_process *= 255
    hist_orig, bounds = np.histogram(im_to_process, 256)
    hist_orig = hist_orig
    hist_cum = np.cumsum(hist_orig)
    z = np.arange(n_quant + 1)
    z[0] = -1
    z[n_quant] = 255
    q = np.arange(n_quant)
    for i in range(1, n_quant + 1):
        up_sum = 0
        dw_sum = 0
        indices_array = np.where(hist_cum > i * (hist_cum[255] / n_quant))
        if i != n_quant:
            z[i] = indices_array[0][0]
        for j in range(z[i - 1] + 1, z[i] + 1):
            up_sum += hist_orig[j] * j
            dw_sum += hist_orig[j]
        q[i - 1] = up_sum / dw_sum
    k = 0
    error = np.zeros(0)
    z_new = np.arange(n_quant + 1)
    z_new[0] = -1
    z_new[n_quant] = 255
    while k < n_iter:
        error_sum = 0
        for i in range(1, n_quant + 1):
            up_sum = 0
            dw_sum = 0
            if i != n_quant:
                z_new[i] = (q[i - 1] + q[i]) / 2
            else:
                z_new[i] = z[i]
            for j in range(z_new[i - 1] + 1, z_new[i] + 1):
                up_sum += hist_orig[j] * j
                dw_sum += hist_orig[j]
            q[i - 1] = up_sum / dw_sum
            for j in range(z_new[i - 1] + 1, z_new[i] + 1):
                error_sum += ((q[i - 1] - j) ** 2) * hist_orig[j]
        error = np.append(error, error_sum)
        k += 1
        if np.array_equal(z, z_new):
            break
        z = np.copy(z_new)
    map_vector = np.zeros(256)
    for i in range(0, n_quant):
        map_vector[z[i] + 1:z[i + 1] + 1] = q[i]
    im_to_process = map_vector[im_to_process.astype(np.int32)]
    im_to_process /= 255
    if dim == 3:
        yiq_im[:, :, 0] = im_to_process
        im_quant = yiq2rgb(yiq_im)
    else:
        im_orig /= 255
        im_quant = im_to_process
    return [im_quant, error]







# mat = read_image("C:\\Users\\Roy\PycharmProjects\\ex1IP\\monkey.jpg", 1)
# image = histogram_equalize(mat)
# x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :],
#                np.array([255] * 6)[None, :]])
# grad = np.tile(x.astype(np.float64), (256, 1))
# grad /= 255
# y = x.shape
# z = histogram_equalize(grad)
# image = quantize(z[0], 5, 5)
#
# plt.imshow(image[0], cmap=plt.cm.gray)
#
# plt.show()




# imdisplay("C:\\Users\\Roy\PycharmProjects\\ex1IP\\jerusalem.jpg", 2)

# mat = [[52, 55, 61,  59,  79,  61, 76, 61],
#     [62, 59, 55,  104, 94,  85, 59, 71],
#     [63, 65, 66, 113, 144, 104, 63, 72],
#     [64, 70, 70, 126, 154, 109, 71, 69],
#     [67, 73, 68, 106, 122,  88, 68, 68],
#     [68, 79, 60,  70,  77,  66, 58, 75],
#     [69, 85, 64,  58,  55,  61, 65, 83],
#     [70, 87, 69,  68,  65,  73, 78, 90]]

# [[  0  12  53  32 190  53 174  53]
#  [ 57  32  12 227 219 202  32 154]
#  [ 65  85  93 239 251 227  65 158]
#  [ 73 146 146 247 255 235 154 130]
#  [ 97 166 117 231 243 210 117 117]
#  [117 190  36 146 178  93  20 170]
#  [130 202  73  20  12  53  85 194]
#  [146 206 130 117  85 166 182 215]]


# the_mat = yiq2rgb(rgb2yiq(read_image("C:\\Users\\Roy\PycharmProjects\\ex1IP\\monkey.JPG", 2)))
# plt.imshow(the_mat)
# plt.show()


import skimage.color

images = []
jer_bw = read_image("C:\\Users\\Roy\PycharmProjects\\ex1IP\\jerusalem.jpg", 1)
images.append((jer_bw, "jerusalem grayscale"))
jer_rgb = read_image("C:\\Users\\Roy\PycharmProjects\\ex1IP\\jerusalem.jpg", 2)
images.append((jer_rgb, "jerusalem RGB"))
low_bw = read_image("C:\\Users\\Roy\PycharmProjects\\ex1IP\\low_contrast.jpg", 1)
images.append((low_bw, "low_contrast grayscale"))
low_rgb = read_image("C:\\Users\\Roy\PycharmProjects\\ex1IP\\low_contrast.jpg", 2)
images.append((low_rgb, "low_contrast RGB"))
monkey_bw = read_image("C:\\Users\\Roy\PycharmProjects\\ex1IP\\monkey.jpg", 1)
images.append((monkey_bw, "monkey grayscale"))
monkey_rgb = read_image("C:\\Users\\Roy\PycharmProjects\\ex1IP\\monkey.jpg", 2)
images.append((monkey_rgb, "monkey RGB"))


def test_rgb2yiq_and_yiq2rgb(im, name):
    """
    Tests the rgb2yiq and yiq2rgb functions by comparing them to the built in ones in the skimage library.
    Allows error to magnitude of 1.e-3 (Difference from built in functions can't be bigger than 0.001).
    :param im: The image to test on.
    :param name: Name of image.
    :return: 1 on success, 0 on failure.
    """
    imp = rgb2yiq(im)
    off = skimage.color.rgb2yiq(im)

    if not np.allclose(imp, off, atol=1.e-3):
        print("ERROR: in rgb2yiq on image '%s'" % name)
        return 0
    imp2 = yiq2rgb(imp)
    off2 = skimage.color.yiq2rgb(off)
    if not np.allclose(imp2, off2, atol=1.e-3):
        print("ERROR: in yiq2rgb on image '%s'" % name)
        return 0
    print("passed conversion test on '%s'" % name)
    return 1


for im in images:
    if len(im[0].shape) == 3:
        result = test_rgb2yiq_and_yiq2rgb(im[0], im[1])
        if not result:
            print("=== Failed Conversion Test ===")
            break


def display_all(im, add_bonus):
    if len(im.shape) == 3 and add_bonus:
        fig, a = plt.subplots(nrows=3, ncols=2)
    else:
        fig, a = plt.subplots(nrows=2, ncols=2)

    # adds the regular image
    a[0][0].imshow(im, cmap=plt.cm.gray)
    a[0][0].set_title(r"original image")

    # adds the quantified image
    quant = quantize(im, 3, 10)[0]
    a[0][1].imshow(quant, cmap=plt.cm.gray)
    a[0][1].set_title(r"quantize to 3 levels, 10 iterations")

    # adds the histogram equalized image
    hist = histogram_equalize(im)[0]
    a[1][0].imshow(hist, cmap=plt.cm.gray)
    a[1][0].set_title("histogram equalization")

    # adds quantization on histogram equalized image
    hist_quant = quantize(hist, 6, 10)[0]
    a[1][1].imshow(hist_quant, cmap=plt.cm.gray)
    a[1][1].set_title("quantize on equalization")

    # adds the bonus image
    # if len(im.shape) == 3 and add_bonus:
        # a[2][0].imshow(quantize_rgb(im, 3))
        # a[2][0].set_title(r"bonus quantize_rgb")

    plt.show()


for im in images:
    # change "False" to "True" if you wish to add the bonus task to the print
    display_all(im[0], False)