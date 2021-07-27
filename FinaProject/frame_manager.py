import cv2
import os
import numpy as np


def frame_to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def count_frames(clip_mp4_path: str) -> int:
    cap = cv2.VideoCapture(clip_mp4_path)

    frames_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames_counter += 1
        else:
            break

    cap.release()

    return frames_counter


# noinspection PyTypeChecker
def crop_frame(clip_mp4_path: str, frame_k: int):
    cap = cv2.VideoCapture(clip_mp4_path)

    frames_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames_counter += 1

            if frames_counter == frame_k:
                fn = f'{clip_mp4_path}_frame_{frame_k}.npy'

                if not os.path.exists(fn):
                    with open(fn, 'wb') as f:
                        np.save(f, frame)
                break
        else:
            break

    cap.release()


# noinspection PyTypeChecker
def load_frame(clip_mp4_path: str, frame_k: int, load_gray_frame: bool = True) -> np.ndarray:
    fn = f'{clip_mp4_path}_frame_{frame_k}.npy'

    if not os.path.exists(fn):
        crop_frame(clip_mp4_path=clip_mp4_path, frame_k=frame_k)
        return load_frame(clip_mp4_path=clip_mp4_path,
                          frame_k=frame_k,
                          load_gray_frame=load_gray_frame)

    with open(fn, 'rb') as f:
        frame = np.load(f)
        if load_gray_frame:
            return frame_to_gray(frame=frame)
        else:
            return frame


def cut_table_from_frame(frame: np.ndarray, is_bev: bool = False) -> np.ndarray:
    if not is_bev:
        return frame.copy()[250:950, 330:1600]
    else:
        return frame.copy()[25:1030, 90:1850]


def canny(frame_gray: np.ndarray, threshold_low: int = 50, threshold_high: int = 150) -> np.ndarray:
    return cv2.Canny(frame_gray, threshold_low, threshold_high)
