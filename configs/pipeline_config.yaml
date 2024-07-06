import os
import yaml
from src.data_processing.video_to_frames import extract_frames
from src.data_processing.labeling import label_frames
from src.model_training.vision_model_finetuning import train_vision_model
from src.model_training.llm_finetuning import train_llm
from src.model_compression.quantization import quantize_model
from src.edge_deployment.inference import EdgeInference

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_pipeline(config_path):
    config = load_config(config_path)

    # Step 1: Extract frames from video
    print("Extracting frames from video...")
    frames_count = extract_frames(
        config['data']['video_path'],
        config['data']['frames_output_folder'],
        config['data']['frame_interval']
    )
    print(f"Extracted {frames_count} frames.")

    # Step 2: Label frames
    print("Labeling frames...")
    labeled_data = label_frames(
        config['data']['frames_output_folder'],
        config['data']['labeled_data_output'],
        config['labeling']['model'],
        config['labeling']['batch_size']
    )
    print(f"Labeled {len(labeled_data)} frames.")

    # Step 3: Train vision model
    print("Training vision model...")
    vision_model = train_vision_model(
        config['model']['vision_model_config'],
        config['data']['labeled_data_output'],
        config['data']['frames_output_folder']
    )
    print("Vision model training complete.")

    # Step 4: Train LLM
    print("Fine-tuning LLM...")
    llm = train_llm(
        config['model']['llm_config'],
        config['data']['labeled_data_output']
    )
    print("LLM fine-tuning complete.")

    # Step 5: Compress models
    print("Compressing models...")
    quantized_vision_model = quantize_model(vision_model, config['model']['vision_model_config'])
    quantized_llm = quantize_model(llm, config['model']['llm_config'])
    print("Model compression complete.")

    # Step 6: Set up edge inference
    print("Setting up edge inference...")
    edge_inference = EdgeInference(
        quantized_vision_model,
        quantized_llm,
        config['deployment']
    )

    # Step 7: Run inference on test data
    print("Running inference on test data...")
    test_image_path = config['data']['test_image_path']
    result = edge_inference.run_inference(test_image_path)
    print(f"Inference result: {result}")

    print("Pipeline execution complete.")

if __name__ == "__main__":
    run_pipeline('configs/pipeline_config.yaml')