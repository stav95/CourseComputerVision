import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

import project.constants as cons
import project.frame_manager as fm
from project.BorderSide import BorderSide
from project.Line import Line, get_point_cross_lines
from project.Point import Point


def plott(f: np.ndarray):
    plt.imshow(f, cmap='gray')
    plt.show()


def get_frame_border(frame: np.ndarray, border_side: BorderSide) -> np.ndarray:
    border = np.zeros_like(frame)

    middle_x = border.shape[1] // 2
    middle_y = border.shape[0] // 2

    if border_side is BorderSide.Right:
        for _ in range(frame.shape[0]):
            nonzero_indices = np.where(frame[_, middle_x:] > 0)[0]
            if nonzero_indices.size > 0:
                border[_, middle_x + nonzero_indices[0]] = 1

    if border_side is BorderSide.Left:
        for _ in range(frame.shape[0]):
            nonzero_indices = np.where(frame[_, :middle_x] > 0)[0]
            if nonzero_indices.size > 0:
                border[_, nonzero_indices[-1]] = 1

    if border_side is BorderSide.Top:
        for _ in range(frame.shape[1]):
            nonzero_indices = np.where(frame[:middle_y, _] > 0)[0]
            if nonzero_indices.size > 0:
                border[nonzero_indices[-1], _] = 1

    if border_side is BorderSide.Bottom:
        for _ in range(frame.shape[1]):
            nonzero_indices = np.where(frame[middle_y:, _] > 0)[0]
            if nonzero_indices.size > 0:
                border[middle_y + nonzero_indices[0], _] = 1

    kernel = np.ones((2, 2), np.uint8)
    border = cv2.dilate(border, kernel=kernel, iterations=1)

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(border)

    label_line = np.flip(np.argsort(stats[:, 4]))[1:2]

    border[labels != label_line] = 0

    # plott(f=border)

    return border


def get_linear_line(border: np.ndarray) -> Line:
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(border)

    m = 0
    x_total = 0
    y_total = 0
    points = 0
    for label in range(1, retval, 1):
        _y, _x = np.where(labels == label)

        for p1_idx in range(_x.size):
            x_total += _x[p1_idx]
            y_total += _y[p1_idx]

            for p2_idx in range(p1_idx + 1, _x.size, 1):
                points += 1

                x_total += _x[p2_idx]
                y_total += _y[p2_idx]

                change_y = _y[p1_idx] - _y[p2_idx]
                change_x = _x[p1_idx] - _x[p2_idx]

                if change_x != 0 and change_y != 0:
                    m += change_y / change_x

        if points == 0:
            continue

    m /= points
    x_total /= points
    y_total /= points

    return Line(slope=m, y=y_total, x=x_total)


def get_minmax_points_on_line(line: Line, border_side: BorderSide, max_point: Point) -> (Point, Point):
    p1, p2 = None, None

    if border_side is BorderSide.Right or border_side is BorderSide.Left:
        _x = (0 - line.y + line.slope * line.x) / line.slope
        p1 = Point(y=0, x=_x)

        _x = (max_point.y - line.y + line.slope * line.x) / line.slope
        p2 = Point(y=max_point.y, x=_x)

    if border_side is BorderSide.Top or border_side is BorderSide.Bottom:
        _y = line.slope * (0 - line.x) + line.y
        p1 = Point(y=_y, x=0)

        _y = line.slope * (max_point.x - line.x) + line.y
        p2 = Point(y=_y, x=max_point.x)

    return p1, p2


# noinspection PyTypeChecker
def find_table_border(frame: np.ndarray) -> dict:
    frame = fm.canny(frame_gray=frame)

    # noinspection PyShadowingNames
    def get_border_line(border_side: BorderSide) -> (np.ndarray, Line):
        frame_border = get_frame_border(frame=frame, border_side=border_side)

        line = get_linear_line(border=frame_border)

        max_py = frame_border.shape[0]
        max_px = frame_border.shape[1]
        p1_min, p2_max = get_minmax_points_on_line(line=line,
                                                   border_side=border_side,
                                                   max_point=Point(y=max_py, x=max_px))
        c = (255, 255, 255)
        line_on_table = cv2.line(np.zeros_like(frame), (p1_min.x, p1_min.y), (p2_max.x, p2_max.y), c, 1)
        return line_on_table, line

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(frame)

    for _ in range(stats.shape[0]):
        if stats[_, 4] < 200:
            frame[labels == _] = 0

    frame_border_line_top, line_top = get_border_line(border_side=BorderSide.Top)
    frame_border_line_right, line_right = get_border_line(border_side=BorderSide.Right)
    frame_border_line_bottom, line_bottom = get_border_line(border_side=BorderSide.Bottom)
    frame_border_line_left, line_left = get_border_line(border_side=BorderSide.Left)

    frame_border = np.logical_or(frame_border_line_top > 0, frame_border_line_right > 0)
    frame_border = np.logical_or(frame_border, frame_border_line_bottom > 0)
    frame_border = np.logical_or(frame_border, frame_border_line_left > 0)

    cross_top_right = get_point_cross_lines(line_1=line_top, line_2=line_right)
    cross_top_left = get_point_cross_lines(line_1=line_top, line_2=line_left)
    cross_bottom_right = get_point_cross_lines(line_1=line_bottom, line_2=line_right)
    cross_bottom_left = get_point_cross_lines(line_1=line_bottom, line_2=line_left)

    for y in range(frame_border.shape[0]):
        indices = np.where(frame_border[y, :] > 0)[0]

        if len(indices) == 0:
            continue

        frame_border[y, :indices[0]] = 1

        if len(indices) > 1:
            frame_border[y, indices[-1]:] = 1

    y = 0
    while True:
        indices = np.where(frame_border[y, :] > 0)[0]

        if len(indices) > frame_border.shape[1] // 2:
            break

        frame_border[y, :] = 1
        y += 1

    y = frame_border.shape[0] - 1
    while True:
        indices = np.where(frame_border[y, :] > 0)[0]

        if len(indices) > frame_border.shape[1] // 2:
            break

        frame_border[y, :] = 1
        y -= 1

    fn = os.path.join(cons.CLIPS_PATH, 'board_points.npy')
    if not os.path.exists(fn):
        with open(fn, 'wb') as f:
            np.save(f, frame_border)

    # fig, axs = plt.subplots(2)
    #
    # axs[0].imshow(frame, cmap='gray')
    # axs[1].imshow(frame_border, cmap='gray')
    # plt.show()

    return {
        'top_right': cross_top_right,
        'top_left': cross_top_left,
        'bottom_right': cross_bottom_right,
        'bottom_left': cross_bottom_left,
        'board_indices': frame_border == 0
    }


# noinspection PyTypeChecker
class PointsBorder:
    # These functions created after we obtained the results from find_table_border function.
    def __init__(self, frame: np.ndarray):
        self.tl = None
        self.tr = None
        self.br = None
        self.bl = None

        if frame is not None:
            results = find_table_border(frame=frame)
            self.tl = results['top_left']
            self.tr = results['top_right']
            self.br = results['bottom_right']
            self.bl = results['bottom_left']

    def set_points(self, frame: np.ndarray):
        results = find_table_border(frame=frame)

        self.tl = results['top_left']
        self.tr = results['top_right']
        self.br = results['bottom_right']
        self.bl = results['bottom_left']

    def get_top_right(self, frame: np.ndarray = None) -> Point:
        if self.tr is not None:
            return self.tr

        if frame is not None:
            self.set_points(frame=frame)
            return self.tr

        return None
        # return Point(x=1005, y=70)

    def get_top_left(self, frame: np.ndarray = None) -> Point:
        if self.tl is not None:
            return self.tl

        if frame is not None:
            self.set_points(frame=frame)
            return self.tl

        return None
        # return Point(x=260, y=69)

    def get_bottom_right(self, frame: np.ndarray = None) -> Point:
        if self.br is not None:
            return self.br

        if frame is not None:
            self.set_points(frame=frame)
            return self.br

        return None
        # return Point(x=1143, y=653)

    def get_bottom_left(self, frame: np.ndarray = None) -> Point:
        if self.bl is not None:
            return self.bl

        if frame is not None:
            self.set_points(frame=frame)
            return self.bl

        return None
        # return Point(x=113, y=653)

    @staticmethod
    def get_outside_board_indices(frame: np.ndarray) -> np.ndarray:
        fn = os.path.join(cons.CLIPS_PATH, 'board_points.npy')

        if not os.path.exists(fn):
            return find_table_border(frame=frame)['board_indices']

        if os.path.exists(fn):
            with open(fn, 'rb') as f:
                return np.load(f)
