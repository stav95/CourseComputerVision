from typing import List
import numpy as np

from project.Point import Point, get_distance_2_points
from project.ColorMatching import Color


class Ball:
    def __init__(self, _id: int, point: Point, color: Color):
        self.id = _id
        self.last_frame_seen = 0
        self.points: List[Point] = [point]
        self.colors: List[Color] = [color]

    def add_point(self, point: Point):
        threshold = 100
        if self.get_color() is Color.White:
            threshold = 300

        dis = get_distance_2_points(p1=point, p2=self.points[-1])
        if dis > threshold:
            p = self.points[-1]
            self.points.append(Point(y=p.y, x=p.x))
            return

        self.points.append(point)

    def add_color(self, color: Color):
        self.colors.append(color)

    def add_frame(self, point: Point, color: Color):
        self.points.append(point)
        self.colors.append(color)

    def get_color(self) -> Color:
        counters = np.zeros(9)

        for c in self.colors:
            idx = c.value
            counters[idx] += 1

        max_color = np.argmax(counters)

        for c in self.colors:
            if c.value == max_color:
                return c
