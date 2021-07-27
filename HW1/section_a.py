import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d


# region Section A


# noinspection PyTypeChecker
def convolutionMaskA(img: np.ndarray) -> np.ndarray:
    """
    Notes
    -----
    To average 9 pixels in a row, we need to sum all of their value and divide by 9
    therefore we used a weight of 1/9 for each pixel
    """
    if img is None:
        return None

    mask = np.array([[1 / 9] * 9])

    res = convolve2d(img, mask, mode='same')

    return res


# noinspection PyTypeChecker
def convolutionMaskB(img: np.ndarray) -> np.ndarray:
    """
    Notes
    -----
    In a region of 5x5 pixels, we have 21 pixels with high values (close to 255)
    and another 4 pixels with low values (close to 0, our L shape we looking for)
    Let's suppose each of the 21 pixels is 255 so we give a weight of 1/21
    to each one of them in the mask so the maximum value can be 255.
    In the low values pixels we choose to use negative value -4/21 in the mask,
    so if there is high value in the L shape pixels they will balance the high pixels and reduce the sum.
    """
    if img is None:
        return None

    hi = 1 / 21
    lo = -4 / 21

    mask = np.array([[hi, hi, hi, hi, hi],
                     [hi, hi, lo, lo, hi],
                     [hi, hi, hi, lo, hi],
                     [hi, hi, hi, lo, hi],
                     [hi, hi, hi, hi, hi]])

    res = np.abs(convolve2d(img, mask, mode='same'))

    return res


# noinspection PyTypeChecker
def convolutionMaskC(img: np.ndarray) -> np.ndarray:
    """
    Notes
    -----
    Same as B with one difference, the middle pixel is 0.
    """
    if img is None:
        return None

    hi = 1 / 20
    lo = -4 / 21

    mask = np.array([[hi, hi, hi, hi, hi],
                     [hi, hi, lo, lo, hi],
                     [hi, hi, 0, lo, hi],
                     [hi, hi, hi, lo, hi],
                     [hi, hi, hi, hi, hi]])

    res = np.abs(convolve2d(img, mask, mode='same'))

    return res


def my_plot(image_original: np.ndarray, result: np.ndarray):
    f, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row')

    ax1.imshow(image_original, cmap='gray'), ax1.set_title('Original Image')
    ax2.imshow(result, cmap='gray'), ax2.set_title('Applying convolution mask')
    plt.show()


# noinspection PyPep8Naming
def test_A(imageName: str):
    img = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)

    convImageA = convolutionMaskA(img)
    convImageB = convolutionMaskB(img)
    convImageC = convolutionMaskC(img)

    my_plot(img, convImageA)
    my_plot(img, convImageB)
    my_plot(img, convImageC)


# endregion

def load_image(path: str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


if __name__ == '__main__':
    test_A("images/synthCheck.tif")

    print("a")
