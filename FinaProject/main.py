import project.constants as cons

from project.Table import Table
import project.frame_manager as fm


def analyze_clip(clip_k: int):
    table_height = 700
    table_width = table_height // 2
    table = Table(clip_mp4=cons.get_clip_mp4_path(clip_k=clip_k, bev_video=False),
                  clip_bev_mp4=cons.get_clip_mp4_path(clip_k=clip_k, bev_video=True),
                  table_height=table_height,
                  table_width=table_width)

    max_frame = fm.count_frames(clip_mp4_path=cons.get_clip_mp4_path(clip_k=clip_k, bev_video=False))
    max_frame = min(max_frame, fm.count_frames(clip_mp4_path=cons.get_clip_mp4_path(clip_k=clip_k, bev_video=True)))
    # max_frame = 100
    for i in range(1, max_frame, 1):
        table.add_frame(frame_k=i)

    table.save_game_video(out_filename_video=cons.get_output_video_path(clip_k=clip_k))
    table.clear_memory()


if __name__ == '__main__':
    # analyze_clip(clip_k=1)
    analyze_clip(clip_k=2)
    analyze_clip(clip_k=3)
