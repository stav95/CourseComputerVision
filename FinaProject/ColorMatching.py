from enum import Enum

import numpy as np


class Color(Enum):
    Yellow = 1,
    Brown = 2,
    Green = 3,
    Blue = 4,
    Pink = 5,
    Red = 6,
    Black = 7,
    White = 8


def mse_yellow(rgb: np.ndarray) -> float:
    arr = ColorMatching.color_to_rgb(color=Color.Yellow)
    return float(np.sum(np.power(rgb - arr, 2)))


def mse_brown(rgb: np.ndarray) -> float:
    arr = ColorMatching.color_to_rgb(color=Color.Brown)
    return float(np.sum(np.power(rgb - arr, 2)))


def mse_green(rgb: np.ndarray) -> float:
    arr = ColorMatching.color_to_rgb(color=Color.Green)
    return float(np.sum(np.power(rgb - arr, 2)))


def mse_blue(rgb: np.ndarray) -> float:
    arr = ColorMatching.color_to_rgb(color=Color.Blue)
    return float(np.sum(np.power(rgb - arr, 2)))


def mse_pink(rgb: np.ndarray) -> float:
    arr = ColorMatching.color_to_rgb(color=Color.Pink)
    return float(np.sum(np.power(rgb - arr, 2)))


def mse_red(rgb: np.ndarray) -> float:
    arr = ColorMatching.color_to_rgb(color=Color.Red)
    return float(np.sum(np.power(rgb - arr, 2)))


def mse_black(rgb: np.ndarray) -> float:
    arr = ColorMatching.color_to_rgb(color=Color.Black)
    return float(np.sum(np.power(rgb - arr, 2)))


def mse_white(rgb: np.ndarray) -> float:
    arr = ColorMatching.color_to_rgb(color=Color.White)
    return float(np.sum(np.power(rgb - arr, 2)))


class ColorMatching:
    @staticmethod
    def recognize_color(rgb: np.ndarray) -> (Color, float):
        min_color = Color.Yellow

        min_mse = mse_yellow(rgb=rgb)
        _mse_brown = mse_brown(rgb=rgb)
        _mse_green = mse_green(rgb=rgb)
        _mse_blue = mse_blue(rgb=rgb)
        _mse_pink = mse_pink(rgb=rgb)
        _mse_red = mse_red(rgb=rgb)
        _mse_black = mse_black(rgb=rgb)
        _mse_white = mse_white(rgb=rgb)

        if min_mse > _mse_brown:
            min_mse = _mse_brown
            min_color = Color.Brown

        if min_mse > _mse_green:
            min_mse = _mse_green
            min_color = Color.Green

        if min_mse > _mse_blue:
            min_mse = _mse_blue
            min_color = Color.Blue

        if min_mse > _mse_pink:
            min_mse = _mse_pink
            min_color = Color.Pink

        if min_mse > _mse_red:
            min_mse = _mse_red
            min_color = Color.Red

        if min_mse > _mse_black:
            min_mse = _mse_black
            min_color = Color.Black

        if min_mse > _mse_white:
            min_mse = _mse_white
            min_color = Color.White

        return min_color, min_mse

    @staticmethod
    def color_to_rgb(color: Color) -> np.ndarray:
        if color is Color.Yellow:
            return np.array([226, 212, 57])

        if color is Color.Brown:
            return np.array([85, 70, 21])

        if color is Color.Green:
            return np.array([32, 119, 74])

        if color is Color.Blue:
            return np.array([25, 84, 112])

        if color is Color.Pink:
            return np.array([169, 60, 83])

        if color is Color.Red:
            return np.array([112, 16, 18])

        if color is Color.Black:
            return np.array([10, 20, 13])

        if color is Color.White:
            return np.array([226, 247, 167])
