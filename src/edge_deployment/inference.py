import torch
import time
from src.agent_framework.perception_agent import PerceptionAgent
from src.agent_framework.planning_agent import PlanningAgent

class EdgeInference:
    def __init__(self, perception_model_path, planning_model_path, config_path):
        self.perception_agent = PerceptionAgent(perception_model_path, config_path)
        self.planning_agent = PlanningAgent(planning_model_path, config_path)

    def run_inference(self, image_path):
        start_time = time.time()

        # Perception
        perception_result = self.perception_agent.perceive(image_path)

        # Planning
        plan = self.planning_agent.plan(perception_result)

        end_time = time.time()
        inference_time = end_time - start_time

        return {
            "perception": perception_result,
            "plan": plan,
            "inference_time": inference_time
        }

def main():
    edge_inference = EdgeInference(
        'models/compressed/quantized_vision_model.pth',
        'models/compressed/quantized_llm.pth',
        'configs/deployment_config.yaml'
    )

    result = edge_inference.run_inference('data/test_image.jpg')
    print(f"Perception result: {result['perception']}")
    print(f"Plan: {result['plan']}")
    print(f"Inference time: {result['inference_time']:.4f} seconds")

if __name__ == "__main__":
    main()