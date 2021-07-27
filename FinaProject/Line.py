from typing import List

from project.Point import Point


class Line(Point):
    def __init__(self, slope: float, y: int, x: int):
        super().__init__(y=y, x=x)
        self.slope = slope


def get_point_cross_lines(line_1: Line, line_2: Line) -> Point:
    nu = line_2.y - line_1.y + (line_1.x * line_1.slope) - (line_2.x * line_2.slope)
    de = line_1.slope - line_2.slope

    _x = nu / de
    _y = line_1.slope * (_x - line_1.x) + line_1.y

    return Point(y=_y, x=_x)


# noinspection PyTypeChecker
def get_linear_line(p1: Point, p2: Point) -> Line:
    if p1.x == p2.x:
        return None

    slope = (p1.y - p2.y) / (p1.x - p2.x)

    return Line(slope=slope, y=p1.y, x=p1.x)


# noinspection PyTypeChecker
def get_linear_line_from_set_points(points: List[Point]) -> Line:
    m = 0
    x_total = 0
    y_total = 0
    total_points = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points), 1):
            p1 = points[i]
            p2 = points[j]

            if p1.x - p2.x == 0:
                continue

            total_points += 1

            x_total += p2.x + p1.x
            y_total += p2.y + p1.y

            m += (p2.y - p1.y) / (p2.x - p1.x)

    if total_points == 0:
        return None

    m /= total_points
    x_total /= total_points
    y_total /= total_points

    return Line(slope=m, y=y_total, x=x_total)
