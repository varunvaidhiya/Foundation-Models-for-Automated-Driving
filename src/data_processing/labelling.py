import os
import json
import yaml
import base64
import openai
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_config():
    with open('configs/data_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def label_image(image_path, model):
    base64_image = encode_image(image_path)
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image for autonomous vehicle perception. Identify and describe key elements such as vehicles, pedestrians, road signs, and traffic conditions."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ],
            }
        ],
    )
    
    return response.choices[0].message['content']

def main():
    config = load_config()
    frames_folder = config['video']['output_folder']
    output_file = config['labeling']['output_file']
    model = config['labeling']['model']
    batch_size = config['labeling']['batch_size']

    labeled_data = []
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])

    for i in tqdm(range(0, len(frame_files), batch_size), desc="Labeling frames"):
        batch = frame_files[i:i+batch_size]
        for frame in batch:
            frame_path = os.path.join(frames_folder, frame)
            label = label_image(frame_path, model)
            labeled_data.append({"frame": frame, "label": label})

    with open(output_file, 'w') as f:
        json.dump(labeled_data, f, indent=2)

    print(f"Labeling complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()