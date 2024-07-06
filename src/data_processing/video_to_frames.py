import cv2
import os
import yaml
from tqdm import tqdm

def load_config():
    with open('configs/data_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def extract_frames(video_path, output_folder, frame_interval):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            success, frame = video.read()
            if not success:
                break

            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_folder, f"frame_{saved_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_count += 1

            frame_count += 1
            pbar.update(1)

    video.release()
    return saved_count

def main():
    config = load_config()
    video_path = config['video']['input_path']
    output_folder = config['video']['output_folder']
    frame_interval = config['video']['frame_interval']

    frames_extracted = extract_frames(video_path, output_folder, frame_interval)
    print(f"Extracted {frames_extracted} frames from the video.")

if __name__ == "__main__":
    main()