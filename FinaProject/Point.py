import numpy as np


class Point:
    def __init__(self, y: float, x: float):
        self.y = round(y)
        self.x = round(x)

    def get_tuple_y_x(self) -> (int, int):
        return self.y, self.x

    def get_tuple_x_y(self) -> (int, int):
        return self.x, self.y

    def get_tuple_x_y_with_offset(self, offset_x: int = 0, offset_y: int = 0) -> (int, int):
        return self.x + offset_x, self.y + offset_y


def get_distance_2_points(p1: Point, p2: Point) -> float:
    tmp = np.array([p1.x, p1.y])
    tmp2 = np.array([p2.x, p2.y])

    return np.linalg.norm(tmp - tmp2)
