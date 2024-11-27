import os
import shutil

def organize_frames_to_images(parent_dir):
    # 确保输入的路径是有效的文件夹
    if not os.path.isdir(parent_dir):
        print(f"指定的路径 '{parent_dir}' 不是一个有效的文件夹。")
        return
    
    # 遍历父文件夹中的子文件夹
    for subfolder in os.listdir(parent_dir):
        subfolder_path = os.path.join(parent_dir, subfolder)
        
        # 仅处理以 'frames' 开头的子文件夹
        if os.path.isdir(subfolder_path) and subfolder.startswith("frame"):
            images_folder = os.path.join(subfolder_path, "images")
            
            # 创建 images 文件夹
            os.makedirs(images_folder, exist_ok=True)
            
            # 遍历子文件夹中的所有文件
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith(".png"):
                    file_path = os.path.join(subfolder_path, file_name)
                    
                    # 移动 png 文件到 images 文件夹
                    shutil.move(file_path, os.path.join(images_folder, file_name))
            
            print(f"已完成处理子文件夹: {subfolder}")
    print("所有文件已整理完成。")

# 示例调用
parent_directory = r"/home/ethan/Project/Python/I3DV/dataset/Dance_Dunhuang_Pair_1080"  # 替换为你的目标文件夹路径
organize_frames_to_images(parent_directory)
