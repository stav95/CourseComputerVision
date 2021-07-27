from typing import List

import cv2
import numpy as np
from matplotlib import pyplot as plt

import project.frame_manager as fm
from project.Ball import Ball
from project.ColorMatching import Color, ColorMatching
from project.OpticalFlow import OpticalFlow
from project.Point import Point, get_distance_2_points
from project.PointsBorder import PointsBorder


# noinspection PyTypeChecker,PyUnusedLocal,PyMethodMayBeStatic
class Table:
    def __init__(self,
                 clip_mp4: str,
                 clip_bev_mp4: str,
                 table_width: int,
                 table_height: int):
        self.clip_mp4 = clip_mp4
        self.clip_bev_mp4 = clip_bev_mp4

        self.table_width = table_width
        self.table_height = table_height

        self.frames: dict = {}
        self.frames_bev: dict = {}

        self.balls: List[Ball] = []

        self.homography_matrix = None

    def clear_memory(self):
        self.clip_mp4 = None
        self.clip_bev_mp4 = None

        self.table_width = None
        self.table_height = None

        self.frames = None
        self.frames_bev = None

        self.balls = None

        self.homography_matrix = None

    def save_game_video(self, out_filename_video: str):
        # noinspection PyShadowingNames
        def load_frame_right(_key: int) -> np.ndarray:
            f_right = fm.load_frame(clip_mp4_path=self.clip_bev_mp4,
                                    frame_k=_key,
                                    load_gray_frame=False)

            f_right = fm.cut_table_from_frame(frame=f_right, is_bev=True)
            f_right = cv2.rotate(f_right, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            proportion = self.table_height / f_right.shape[0]
            dst_size = int(f_right.shape[1] * proportion), self.table_height
            f_right = cv2.resize(f_right, dst_size)

            return f_right

        key = 1

        if key not in self.frames:
            return

        spacer = 50
        video_width = self.frames[key].shape[1] + self.frames_bev[key].shape[1] + spacer * 3
        video_height = 700

        _ = load_frame_right(_key=key)

        y = video_width + _.shape[1]
        x = video_height
        out = cv2.VideoWriter(out_filename_video, cv2.VideoWriter_fourcc(*"MJPG"), 30, (y, x))

        while key in self.frames:
            f_left = self.frames[key]
            f_middle = self.frames_bev[key]
            f_right = load_frame_right(_key=key)

            frame = np.zeros((700, video_width + f_right.shape[1], 3))

            x_boundaries = [0, f_left.shape[1]]
            frame[:, x_boundaries[0]: x_boundaries[1], :] = f_left

            x_boundaries[0] = x_boundaries[1] + spacer
            x_boundaries[1] = x_boundaries[0] + f_middle.shape[1]

            frame[:, x_boundaries[0]: x_boundaries[1], :] = f_middle

            x_boundaries[0] = x_boundaries[1] + spacer
            x_boundaries[1] = x_boundaries[0] + f_right.shape[1]

            frame[:, x_boundaries[0]: x_boundaries[1], :] = f_right
            frame = frame.astype('uint8')

            print(f'Saving frame {key}')
            out.write(frame.astype('uint8'))
            key += 1

        out.release()

    # noinspection PyShadowingNames
    def detect_balls_on_frame(self, frame: np.ndarray, frame_k: int) -> np.ndarray:
        frame_gray = fm.frame_to_gray(frame=frame)
        frame_gray = cv2.Canny(frame_gray, 50, 150)

        indices = PointsBorder.get_outside_board_indices(frame=frame_gray)
        frame_gray[indices] = 255

        # self.plott(f=frame_gray)

        kernel = np.ones((2, 2), np.uint8)
        frame_gray_dilated = cv2.dilate(frame_gray, kernel=kernel, iterations=5)

        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(frame_gray_dilated)

        # self.plott(f=frame_gray_dilated)
        def get_horizontal_label_size(_labels: np.ndarray, _label: int) -> int:
            _, _indices_x, = np.where(_labels == _label)

            return np.max(_indices_x) - np.min(_indices_x)

        def get_vertical_label_size(_labels: np.ndarray, _label: int) -> int:
            _indices_y, _ = np.where(_labels == _label)

            return np.max(_indices_y) - np.min(_indices_y)

        for label in range(1, retval, 1):
            if stats[label, 4] < 200:
                frame_gray_dilated[labels == label] = 0
                continue

            horizontal_size = get_horizontal_label_size(_labels=labels, _label=label)
            vertical_size = get_vertical_label_size(_labels=labels, _label=label)

            if horizontal_size > 100 or vertical_size > 100:
                frame_gray_dilated[labels == label] = 0

        # self.plott(f=frame_gray_dilated)
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(frame_gray_dilated)

        balls_range = range(1, stats.shape[0], 1)
        for ball_label in balls_range:
            indices_y, _ = np.where(labels == ball_label)
            for y in indices_y:
                indices = np.where(labels[y, :] == ball_label)[0]

                if len(indices) > 1:
                    labels[y, indices[0]: indices[-1]] = ball_label

        # noinspection PyShadowingNames
        def get_ball_location(labels_frame: np.ndarray, _label: int) -> Point:
            indices_y, indices_x = np.where(labels_frame == _label)

            _y = np.sum(indices_y) // len(indices_y) - 5
            _x = np.sum(indices_x) // len(indices_x) - 2
            return Point(y=_y, x=_x)

        # noinspection PyShadowingNames
        def get_ball_color(color_frame: np.ndarray, _ball_location: Point):
            p = _ball_location
            colors = color_frame[p.y - 4: p.y + 5, p.x - 4: p.x + 5]

            r, g, b = 0, 0, 0
            for i in range(colors.shape[0]):
                for j in range(colors.shape[1]):
                    b += colors[i, j, 0]
                    g += colors[i, j, 1]
                    r += colors[i, j, 2]

            k = colors.shape[0] * colors.shape[1]
            return np.array([r // k, g // k, b // k])

        add_new_balls = len(self.balls) == 0
        ball_id = 1

        points: List[Point] = []
        colors: List[Color] = []
        for ball_label in balls_range:
            ball_location = get_ball_location(labels_frame=labels, _label=ball_label)
            ball_color_rgb = get_ball_color(color_frame=frame, _ball_location=ball_location)
            color, mse = ColorMatching.recognize_color(rgb=ball_color_rgb)

            ball = Ball(_id=ball_id, point=ball_location, color=color)
            if add_new_balls:
                self.add_new_ball(ball=ball)
                ball_id += 1
            else:
                if mse > 5000:
                    self.add_new_ball_by_distance(point=ball_location)
                    continue
                points.append(ball_location)
                colors.append(color)

            # frame[ball_location.y - 4:ball_location.y + 5, ball_location.x - 4:ball_location.x + 5] = 0

        self.add_all_balls(points=points, colors=colors, frame_k=frame_k)

        new_frame = self.add_text_balls_id(frame=frame)

        return new_frame

    def add_frame(self, frame_k: int):
        self.balls = [ball for ball in self.balls
                      if frame_k - ball.last_frame_seen < 15 or ball.get_color() is Color.White]

        print(f'Adding frame {frame_k}')
        frame = fm.load_frame(clip_mp4_path=self.clip_mp4, frame_k=frame_k, load_gray_frame=False)
        frame = fm.cut_table_from_frame(frame=frame)

        new_frame = self.detect_balls_on_frame(frame=frame, frame_k=frame_k)
        frame_bev = self.get_homography_frame(frame=new_frame)

        for ball in self.balls:
            optical_flow = OpticalFlow(ball=ball)
            optical_flow.calc_optical_flow_of_ball()

            if optical_flow.is_ball_moving():
                new_frame = cv2.arrowedLine(img=new_frame,
                                            pt1=optical_flow.start_point.get_tuple_x_y(),
                                            pt2=optical_flow.end_point.get_tuple_x_y(),
                                            color=(0, 0, 0),
                                            thickness=2,
                                            tipLength=0.5)

                new_frame = cv2.arrowedLine(img=new_frame,
                                            pt1=optical_flow.start_point.get_tuple_x_y_with_offset(offset_y=30),
                                            pt2=optical_flow.end_point.get_tuple_x_y_with_offset(offset_y=30),
                                            color=(0, 0, 0),
                                            thickness=2,
                                            tipLength=0.5)

                new_frame = cv2.arrowedLine(img=new_frame,
                                            pt1=optical_flow.start_point.get_tuple_x_y_with_offset(offset_y=-30),
                                            pt2=optical_flow.end_point.get_tuple_x_y_with_offset(offset_y=-30),
                                            color=(0, 0, 0),
                                            thickness=2,
                                            tipLength=0.5)

        for ball in self.balls:
            points_n = len(ball.points) - 1
            if points_n > 1:
                for i in range(points_n):
                    start_point = (ball.points[i].x, ball.points[i].y)
                    end_point = (ball.points[i + 1].x, ball.points[i + 1].y)

                    color_rgb = ColorMatching.color_to_rgb(color=ball.get_color())

                    new_frame = cv2.line(img=new_frame,
                                         pt1=start_point,
                                         pt2=end_point,
                                         color=np.flip(color_rgb).tolist(),
                                         thickness=5)

                _l = ball.points[-1]
                new_frame[_l.y - 4:_l.y + 5, _l.x - 4:_l.x + 5] = 0

        # if frame_k > 70:
        #     self.plott(f=new_frame)
        self.frames.update({frame_k: new_frame})
        self.frames_bev.update({frame_k: frame_bev})

    def get_closest_ball(self, point: Point) -> Ball:
        distances = np.zeros(len(self.balls))

        for i, ball in enumerate(self.balls):
            dis = get_distance_2_points(p1=point, p2=ball.points[-1])
            distances[i] = dis

        sorted_distances = np.argsort(distances)
        balls_by_distance: List[Ball] = []

        for _ in range(sorted_distances.shape[0]):
            balls_by_distance.append(self.balls[sorted_distances[_]])

        return balls_by_distance

    def add_new_ball(self, ball: Ball):
        self.balls.append(ball)

    def add_new_ball_by_distance(self, point: Point):
        balls_by_distance: List[Ball] = self.get_closest_ball(point=point)

        b = balls_by_distance[0]

        b.add_point(point=point)
        b.add_color(color=b.get_color())

    def add_data_to_ball(self, point: Point, color: Color, frame_k: int, ball_pointer: Ball = None) -> Ball:
        if ball_pointer is not None:
            ball_pointer.add_point(point=point)
            ball_pointer.add_color(color=color)
            ball_pointer.last_frame_seen = frame_k
            return ball_pointer

        balls_by_distance = self.get_closest_ball(point=point)

        for ball in balls_by_distance:
            if ball.get_color() is color:
                ball.add_point(point=point)
                ball.add_color(color=color)
                ball.last_frame_seen = frame_k
                return ball

        return None

    def add_text_balls_id(self, frame: np.ndarray) -> np.ndarray:
        frame = frame.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (255, 255, 255)
        for b in self.balls:
            p = b.points[-1].x, b.points[-1].y
            cv2.putText(img=frame,
                        text=str(b.id),
                        org=p,
                        fontFace=font,
                        fontScale=1,
                        color=font_color,
                        thickness=2)

        return frame

    def add_all_balls(self, points: List[Point], colors: List[Color], frame_k: int):
        error_points: List[Point] = []
        error_colors: List[Color] = []

        ball_pointer = None
        for point, color in zip(points, colors):
            ball = self.add_data_to_ball(point=point, color=color, frame_k=frame_k)

            if ball is None:
                error_points.append(point)
                error_colors.append(color)
            else:
                ball_pointer = ball

        if len(error_points) > 0:
            for error_point, error_color in zip(error_points, error_colors):
                for ball in self.balls:
                    if len(ball.points) != len(ball_pointer.points):
                        if ball.get_color() is Color.White:
                            self.add_data_to_ball(point=error_points[0],
                                                  color=Color.White,
                                                  frame_k=frame_k,
                                                  ball_pointer=ball)
                        break

    def find_homography_matrix(self, frame: np.ndarray):
        if self.homography_matrix is not None:
            return

        pb = PointsBorder(frame=frame)

        tl = pb.get_top_left()
        tr = pb.get_top_right()
        br = pb.get_bottom_right()
        bl = pb.get_bottom_left()

        src_points = np.array([
            [tl.x, tl.y],
            [tr.x, tr.y],
            [br.x, br.y],
            [bl.x, bl.y]
        ])

        dst_table_points = np.array([[0, 0, 1],
                                     [0, self.table_width, 1],
                                     [self.table_height, self.table_width, 1],
                                     [self.table_height, 0, 1]])

        homography_matrix, _ = cv2.findHomography(src_points, dst_table_points)

        self.homography_matrix = homography_matrix

    def get_homography_frame(self, frame: np.ndarray) -> np.ndarray:
        self.find_homography_matrix(frame=frame)

        frame_gray = fm.frame_to_gray(frame=frame)

        size = self.table_height, self.table_width
        frame_bev = cv2.warpPerspective(src=frame,
                                        M=self.homography_matrix,
                                        dsize=size)

        frame_bev = cv2.rotate(frame_bev, cv2.cv2.ROTATE_90_CLOCKWISE)
        frame_bev = cv2.flip(frame_bev, 1)

        return frame_bev

    def plott(self, f: np.ndarray):
        plt.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        plt.show()
