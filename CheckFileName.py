import os
import re

def correct_mp3_filenames(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            match = re.match(r"(\d+)\s+Track\s+(\d+)\.mp3", filename)
            if match:
                leading_num, track_num = match.groups()
                if leading_num != track_num:
                    correct_name = f"{track_num} Track {track_num}.mp3"
                    old_path = os.path.join(folder_path, filename)
                    new_path = os.path.join(folder_path, correct_name)
                    print(f"Renaming: {filename} -> {correct_name}")
                    os.rename(old_path, new_path)

# Example usage
folder_path = r"/Users/nguyenconghung/Music/Music/Media.localized/Music/Unknown Artist/AEF3CD1"
correct_mp3_filenames(folder_path)