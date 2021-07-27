from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d


def gaussian_dx(x: np.ndarray, y: np.ndarray, sigma: float):
    return (-x / (2 * np.pi * sigma ** 4)) * np.exp(-(np.square(x) + np.square(y)) / (2 * sigma ** 2))


def gaussian_dy(x: np.ndarray, y: np.ndarray, sigma: float):
    return (-y / (2 * np.pi * sigma ** 4)) * np.exp(-(np.square(x) + np.square(y)) / (2 * sigma ** 2))


def calc_mask_size(sigma: float) -> int:
    def gaussian(_x, _y, _sigma):
        return (1 / (2 * np.pi * _sigma ** 2)) * np.exp(-(np.square(_x) + np.square(_y)) / (2 * _sigma ** 2))

    size_mask = 1
    while True:
        if (size_mask % 2) == 0:
            size_mask += 1

        ax = np.linspace(-(size_mask - 1) / 2., (size_mask - 1) / 2., size_mask)

        x, y = np.meshgrid(ax, ax)

        if np.sum(gaussian(x, y, sigma)) >= 0.95:
            return size_mask
        size_mask += 1


def gaussian_derivative_xy(sigma: float) -> (np.ndarray, np.ndarray):
    mask_size = calc_mask_size(sigma)

    ax = np.linspace(-(mask_size - 1) / 2., (mask_size - 1) / 2., mask_size)

    x, y = np.meshgrid(ax, ax)
    g_dx = gaussian_dx(x, y, sigma)
    g_dy = gaussian_dy(x, y, sigma)

    return g_dx, g_dy


def get_angles(i_x: np.ndarray, i_y: np.ndarray) -> np.ndarray:
    # Set all zeros to value close to zero because we cannot divide by zero
    i_x[i_x == 0] = 0.0000001

    angles = np.degrees(np.arctan(i_y / i_x))

    # Each pixel have 8 neighbors in those angles.
    angles[(-90 <= angles) & (angles < -67.5)] = 90
    angles[(-67.5 <= angles) & (angles < -22.5)] = 135
    angles[(-22.5 <= angles) & (angles < 22.5)] = 0
    angles[(22.5 <= angles) & (angles < 67.5)] = 45
    angles[67.5 <= angles] = 90

    return angles


def thinning(i_magnitude: np.ndarray, angles: np.ndarray) -> np.ndarray:
    img = i_magnitude.copy()

    height, width = img.shape

    # Find pixel's neighbors by angle and coordinates.
    def get_neighbors(_x: int, _y: int, _angle: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if _angle == 0:
            return (_x - 1, _y), (_x + 1, _y)
        elif _angle == 45:
            return (_x - 1, _y - 1), (_x + 1, _y + 1)
        elif _angle == 90:
            return (_x, _y - 1), (_x, _y + 1)
        elif _angle == 135:
            return (_x - 1, _y + 1), (_x + 1, _y - 1)

    # For each pixel we check if their neighbors have bigger values, if so set the pixel to 0.
    for x in range(width):
        for y in range(height):
            neigh_1, neigh_2 = get_neighbors(x, y, angles[y, x])

            if width > neigh_1[0] >= 0 and height > neigh_1[1] >= 0:
                if img[neigh_1[1], neigh_1[0]] >= img[y, x]:
                    img[y, x] = 0

            if width > neigh_2[0] >= 0 and height > neigh_2[1] >= 0:
                if img[neigh_2[1], neigh_2[0]] >= img[y, x]:
                    img[y, x] = 0

    return img


def hysteresis(i_magnitude: np.ndarray, H_th: float, L_th: float) -> np.ndarray:
    img = i_magnitude.copy()

    # We mark all strong pixels (pixel > high threshold) to 2, weak pixels (low < pixel < high) to 1 and others to 0.
    strong = 2
    weak = 1
    img[img < L_th] = 0
    img[img >= H_th] = strong
    img[(img > L_th) & (img < H_th)] = weak

    mag_uint8 = np.array(img, dtype=np.uint8)
    labels, comps = cv2.connectedComponents(mag_uint8, connectivity=8)

    # For each label we going through it's component and if one of the pixels
    # in the component is strong then mark all component's pixels as strong
    for label in range(1, labels):
        comps_y, comps_x = np.where(comps == label)
        found_strong = False

        for idx in range(len(comps_y)):
            if img[comps_y[idx], comps_x[idx]] == strong:
                found_strong = True
                break

        if found_strong:
            img[comps_y, comps_x] = strong
        else:
            img[comps_y, comps_x] = 0

    img[img == weak] = 0
    img[img == strong] = 1

    return img


def plot_all(img_bytes, gx_result_conv, gy_result_conv, g_result, g_result_thinning, g_final, H_th, L_th):
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')
    ax1.imshow(img_bytes, cmap='gray'), ax1.set_title('Original')
    ax2.imshow(gx_result_conv, cmap='gray'), ax2.set_title('Image_x')
    ax3.imshow(gy_result_conv, cmap='gray'), ax3.set_title('Image_y')
    ax4.imshow(g_result, cmap='gray'), ax4.set_title('|Image|')
    ax5.imshow(g_result_thinning, cmap='gray'), ax5.set_title('Thinning')
    ax6.imshow(g_final, cmap='gray'), ax6.set_title('Hysteresis - H_th: {0}, L_th: {1}'.format(H_th, L_th))
    plt.show()


def canny(img: np.ndarray, sigma: float, L_th: float, H_th: float) -> np.ndarray:
    g_dx, g_dy = gaussian_derivative_xy(sigma)

    i_x = np.abs(convolve2d(img, g_dx, mode='same'))
    i_y = np.abs(convolve2d(img, g_dy, mode='same'))

    i_magnitude = np.sqrt(np.square(i_x) + np.square(i_y))
    i_magnitude = i_magnitude / np.max(i_magnitude)
    i_orientation = get_angles(i_x, i_y)
    i_thinning = thinning(i_magnitude, i_orientation)
    i_hysteresis = hysteresis(i_thinning, H_th, L_th)

    plot_all(img, i_x, i_y, i_magnitude, i_thinning, i_hysteresis, H_th, L_th)

    return i_hysteresis


def load_image(path: str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def test_B(imageName: str, sigma: float, L_th: float, H_th: float) -> np.ndarray:
    img = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)

    cannyResults = canny(img, sigma, L_th, H_th)

    return cannyResults


if __name__ == "__main__":
    imageName = './images/Church.jpg'
    # imageName = './images/cameraman.jpg'
    # imageName = './images/Nuns.jpg'
    sigma, L_th, H_th = 1.3, 0.1, 0.15
    #
    img = test_B(imageName, sigma, L_th, H_th)

    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    img_gt = load_image('./images/Church_GT.bmp')

    img_gt_strong_y, img_gt_strong_x = np.where(img_gt == 1)
    n = len(img_gt_strong_x)
    counter = 0
    for i in range(n):
        if img[img_gt_strong_y[i], img_gt_strong_x[i]] == 1:
            counter += 1

    R = counter / n
    P = counter / img[img == 1].sum()
    F = (2 * P * R) / (P + R)

    print("R {}, P {}, F {}".format(R, P, F))

    print("DFDSF")
