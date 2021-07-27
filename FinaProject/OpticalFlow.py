import numpy as np

from project.Ball import Ball
from project.Point import get_distance_2_points, Point


class OpticalFlow:
    def __init__(self, ball: Ball):
        self.ball = ball

        self.velocity = 0.0
        self.slope = 0.0

        self.avg_change_x = 0
        self.avg_change_y = 0

        self.start_point = None
        self.end_point = None

    def is_ball_moving(self) -> bool:
        if abs(self.avg_change_x) > 1 or abs(self.avg_change_y > 1):
            return True

        return False

    def calc_optical_flow_of_ball(self) -> bool:
        k = 4

        if len(self.ball.points) < k:
            return False

        points = self.ball.points[-k:]

        _x = []
        _y = []

        distances = []
        changes_x = []
        changes_y = []
        for i in range(len(points)):
            p1 = points[i]

            _x.append(p1.x)
            _y.append(p1.y)

            for j in range(i + 1, len(points), 1):
                p2 = points[j]

                changes_x.append(p2.x - p1.x)
                changes_y.append(p2.y - p1.y)

                dis = get_distance_2_points(p1=p1, p2=p2)
                distances.append(dis)

        self.avg_change_x = np.array(changes_x).mean()
        self.avg_change_y = np.array(changes_y).mean()

        self.start_point = Point(y=round(np.array(_y[:k // 2]).mean()),
                                 x=round(np.array(_x[:k // 2]).mean()))
        self.end_point = Point(y=round(np.array(_y[-k // 2:]).mean()),
                               x=round(np.array(_x[-k // 2:]).mean()))

        if get_distance_2_points(p1=self.start_point, p2=self.end_point) < 20:
            self.start_point = Point(y=round(_y[0]),
                                     x=round(_x[0]))
            self.end_point = Point(y=round(_y[-1]),
                                   x=round(_x[-1]))
