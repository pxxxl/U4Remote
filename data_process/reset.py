import os
import shutil

def clean_folder(folder_path):
    """
    仅保留指定文件夹中的 .mp4 文件，删除其他文件和子文件夹。

    :param folder_path: 目标文件夹路径
    """
    # 确保目标路径存在
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    # 遍历文件夹中的内容
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        
        # 如果是子文件夹，删除
        if os.path.isdir(item_path):
            print(f"Removing folder: {item_path}")
            shutil.rmtree(item_path)
        
        # 如果是文件且不是 .mp4，删除
        elif os.path.isfile(item_path) and not item.lower().endswith('.mp4'):
            print(f"Removing file: {item_path}")
            os.remove(item_path)

    print("Cleanup complete. Only .mp4 files remain.")

# 指定目标文件夹路径
target_folder = "/home/ethan/Project/Python/I3DV/dataset/Dance_Dunhuang_Pair_1080"
clean_folder(target_folder)
