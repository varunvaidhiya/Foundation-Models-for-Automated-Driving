import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import json
import yaml
from PIL import Image
import os

def load_config():
    with open('configs/model_config.yaml', 'r') as f:
        return yaml.safe_load(f)

class LabeledImageDataset(Dataset):
    def __init__(self, labeled_data_file, img_dir, transform=None):
        with open(labeled_data_file, 'r') as f:
            self.labeled_data = json.load(f)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labeled_data)

    def __getitem__(self, idx):
        img_name = self.labeled_data[idx]['frame']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.labeled_data[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label

def train_model(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}')

def main():
    config = load_config()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = LabeledImageDataset(
        config['data']['labeled_data_file'],
        config['data']['img_dir'],
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config['model']['num_classes'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    train_model(model, dataloader, criterion, optimizer, config['training']['num_epochs'])

    torch.save(model.state_dict(), config['model']['save_path'])
    print(f"Model saved to {config['model']['save_path']}")

if __name__ == "__main__":
    main()