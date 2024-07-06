import torch
from torchvision import models, transforms
from PIL import Image
import yaml
import cv2
import numpy as np

class PerceptionAgent:
    def __init__(self, model_path, config_path):
        self.config = self.load_config(config_path)
        self.model = self.load_model(model_path)
        self.transform = self.get_transform()
        self.object_detector = cv2.dnn.readNetFromCaffe(
            self.config['object_detection']['prototxt_path'],
            self.config['object_detection']['model_path']
        )

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_model(self, model_path):
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, self.config['model']['num_classes'])
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def process_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(image_tensor)
        
        return output, np.array(image)

    def detect_objects(self, image):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        
        self.object_detector.setInput(blob)
        detections = self.object_detector.forward()
        
        objects = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.config['object_detection']['confidence_threshold']:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                objects.append({
                    "class": self.config['object_detection']['classes'][idx],
                    "confidence": float(confidence),
                    "box": (startX, startY, endX, endY)
                })
        
        return objects

    def classify_scene(self, output):
        _, predicted = torch.max(output, 1)
        return self.config['scene_classes'][predicted.item()]

    def estimate_distances(self, objects, image_width):
        # This is a simplified distance estimation. In a real system, you'd use more sophisticated methods.
        for obj in objects:
            obj_width = obj['box'][2] - obj['box'][0]
            obj['distance'] = f"{int(image_width / obj_width * 10)}m"  # Rough estimate

    def perceive(self, image_path):
        output, image = self.process_image(image_path)
        scene_type = self.classify_scene(output)
        objects = self.detect_objects(image)
        self.estimate_distances(objects, image.shape[1])

        return {
            "scene_type": scene_type,
            "objects": objects
        }

def main():
    agent = PerceptionAgent('models/compressed/quantized_vision_model.pth', 'configs/agent_config.yaml')
    result = agent.perceive('data/test_image.jpg')
    print(f"Scene type: {result['scene_type']}")
    print("Detected objects:")
    for obj in result['objects']:
        print(f"  {obj['class']} (confidence: {obj['confidence']:.2f}) at distance {obj['distance']}")

if __name__ == "__main__":
    main()