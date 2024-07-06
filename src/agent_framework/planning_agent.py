import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

class PlanningAgent:
    def __init__(self, model_path, config_path):
        self.config = self.load_config(config_path)
        self.model, self.tokenizer = self.load_model(model_path)

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_model(self, model_path):
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        return model, tokenizer

    def plan(self, perception_result):
        prompt = self.create_prompt(perception_result)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids, 
                max_length=150, 
                num_return_sequences=1, 
                no_repeat_ngram_size=2
            )
        
        plan = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return self.parse_plan(plan)

    def create_prompt(self, perception_result):
        return f"""
        Based on the following perception data, create a safe driving plan:
        {perception_result}
        
        Driving plan:
        1.
        """

    def parse_plan(self, plan):
        # Extract the numbered steps from the plan
        steps = [step.strip() for step in plan.split('\n') if step.strip().startswith(tuple('123456789'))]
        return steps

def main():
    agent = PlanningAgent('models/compressed/quantized_llm.pth', 'configs/agent_config.yaml')
    perception_result = {
        "vehicles": [{"type": "car", "distance": "10m", "direction": "ahead"}],
        "pedestrians": [{"distance": "5m", "direction": "left"}],
        "traffic_signs": [{"type": "stop_sign", "distance": "15m"}]
    }
    plan = agent.plan(perception_result)
    print("Driving Plan:")
    for step in plan:
        print(step)

if __name__ == "__main__":
    main()