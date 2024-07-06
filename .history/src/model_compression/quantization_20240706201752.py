import torch
import torch.quantization
from torchvision import models
import yaml

def load_config():
    with open('configs/model_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def quantize_model(model, backend='fbgemm'):
    model.eval()
    
    # Fuse Conv, BN and ReLU layers
    model = torch.quantization.fuse_modules(model, [['conv1', 'bn1', 'relu']], inplace=True)
    for module_name, module in model.named_children():
        if 'layer' in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2']], inplace=True)
    
    # Specify quantization configuration
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    
    # Prepare model for quantization
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate the model (you would normally do this with a representative dataset)
    # For this example, we're just using random data
    input_fp32 = torch.randn(5, 3, 224, 224)
    model(input_fp32)
    
    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    
    return model

def main():
    config = load_config()
    
    # Load the trained model
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, config['model']['num_classes'])
    model.load_state_dict(torch.load(config['model']['save_path']))
    
    # Quantize the model
    quantized_model = quantize_model(model)
    
    # Save the quantized model
    torch.save(quantized_model.state_dict(), config['model']['quantized_save_path'])
    print(f"Quantized model saved to {config['model']['quantized_save_path']}")
    
    # Compare model sizes
    torch.save(model.state_dict(), "temp_full.pth")
    torch.save(quantized_model.state_dict(), "temp_quantized.pth")
    
    full_size = os.path.getsize("temp_full.pth") / (1024 * 1024)
    quantized_size = os.path.getsize("temp_quantized.pth") / (1024 * 1024)
    
    print(f"Full model size: {full_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size/full_size)*100:.2f}%")
    
    os.remove("temp_full.pth")
    os.remove("temp_quantized.pth")

if __name__ == "__main__":
    main()