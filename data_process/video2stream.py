import cv2
import os
from argparse import ArgumentParser

# parser = ArgumentParser("Video converter")
# parser.add_argument("--source_path", "-s", required=True, type=str)
# args = parser.parse_args()

def extract_frames_to_folders(video_path, base_folder, video_label):
    """从视频中提取所有帧并根据帧号将它们保存到相应的文件夹中"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 为当前帧创建目录
        frame_folder = f"{base_folder}{frame_idx:06d}"
        os.makedirs(frame_folder, exist_ok=True)

        # 格式化文件名并保存图片
        output_filename = f"{video_label}.png"
        output_path = os.path.join(frame_folder, output_filename)
        cv2.imwrite(output_path, frame)
        print(f"Saved {output_path}")
        frame_idx += 1

    # 关闭视频文件
    cap.release()

# scenes = ["coffee_martini", "cook_spinach", "cut_roasted_beef", "flame_salmon_1", "flame_steak", "sear_steak"]
# scenes = ["coffee_martini", "cut_roasted_beef", "flame_salmon_1"]
scenes = ["Dance_Dunhuang_Pair_1080"]

base_path = "/home/ethan/Project/Python/I3DV/dataset"
source_path = "/home/ethan/Project/Python/I3DV/dataset"

for scene in scenes:
    # 循环处理每个视频文件
    # 先获取视频（相机）个数，即Video path的mp4文件个数
    video_path = os.path.join(source_path, scene)
    num_videos = len([name for name in os.listdir(video_path) if name.endswith(".mp4")])
    print(f"Processing {num_videos} videos in {scene}...")
    for i in range(num_videos):
        video_filename = os.path.join(source_path, scene, f"{i}.mp4")
        base_folder = os.path.join(base_path, scene) + "/frame"
        video_label = f"cam{i:02d}"
        extract_frames_to_folders(video_filename, base_folder, video_label)
