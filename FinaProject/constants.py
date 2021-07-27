from os.path import join as path_join

DATASET_PATH = 'dataset'
CLIPS_PATH = path_join(DATASET_PATH, 'clips')

# CLIP_3_PATH = path_join(CLIPS_PATH, 'clip_3')
# CLIP_3_MP4 = path_join(CLIP_3_PATH, 'clip_3.mp4')
#
# CLIP_3_BEV_PATH = path_join(CLIPS_PATH, 'clip_3_bev')
# CLIP_3_BEV_MP4 = path_join(CLIP_3_PATH, 'clip_3_bev.mp4')


def get_clip_mp4_path(clip_k: int, bev_video: bool) -> str:
    if bev_video:
        fn = f'clip_{clip_k}_bev.mp4'
    else:
        fn = f'clip_{clip_k}.mp4'

    return path_join(CLIPS_PATH, f'clip_{clip_k}', fn)


def get_output_video_path(clip_k: int):
    return path_join(CLIPS_PATH, f'final_video_{clip_k}.avi')
