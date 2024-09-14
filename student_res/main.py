import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image

from src.utils import download_images, parse_string
from src.constants import entity_unit_map, allowed_units

class ProductImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_test=False):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, os.path.basename(self.data.iloc[idx]['image_link']))
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        entity_name = self.data.iloc[idx]['entity_name']
        entity_value = self.data.iloc[idx]['entity_value'] if not self.is_test else ""
        
        return image, entity_name, entity_value

class EntityExtractor(nn.Module):
    def __init__(self, num_entity_types, num_units):
        super(EntityExtractor, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        
        self.fc_value = nn.Linear(256, 1)
        self.fc_unit = nn.Linear(256, num_units)
        self.fc_entity = nn.Linear(256, num_entity_types)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        value = self.fc_value(x)
        unit = self.fc_unit(x)
        entity = self.fc_entity(x)
        
        return value, unit, entity

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, entity_names, entity_values in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            entity_names = torch.tensor([list(entity_unit_map.keys()).index(name) for name in entity_names]).to(device)
            numeric_values, units = parse_entity_values(entity_values)
            numeric_values = numeric_values.to(device)
            units = units.to(device)
            
            optimizer.zero_grad()
            pred_values, pred_units, pred_entities = model(images)
            loss = criterion(pred_values, numeric_values, pred_units, units, pred_entities, entity_names)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, entity_names, entity_values in val_loader:
            images = images.to(device)
            entity_names = torch.tensor([list(entity_unit_map.keys()).index(name) for name in entity_names]).to(device)
            numeric_values, units = parse_entity_values(entity_values)
            numeric_values = numeric_values.to(device)
            units = units.to(device)
            
            pred_values, pred_units, pred_entities = model(images)
            loss = criterion(pred_values, numeric_values, pred_units, units, pred_entities, entity_names)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def predict(model, test_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, entity_names, _ in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            pred_values, pred_units, pred_entities = model(images)
            batch_predictions = convert_predictions(pred_values, pred_units, pred_entities, entity_names)
            predictions.extend(batch_predictions)
    return predictions

def parse_entity_values(entity_values):
    numeric_values = []
    units = []
    for value in entity_values:
        parsed = parse_string(value)
        if parsed[0] is not None:
            numeric_values.append(parsed[0])
            units.append(list(allowed_units).index(parsed[1]))
        else:
            numeric_values.append(0.0)
            units.append(-1)
    return torch.tensor(numeric_values).float(), torch.tensor(units)

def convert_predictions(pred_values, pred_units, pred_entities, true_entities):
    predictions = []
    for value, unit, entity, true_entity in zip(pred_values, pred_units, pred_entities, true_entities):
        numeric_value = value.item()
        unit_idx = torch.argmax(unit).item()
        entity_idx = torch.argmax(entity).item()
        
        entity_name = list(entity_unit_map.keys())[entity_idx]
        if entity_name != true_entity:
            predictions.append("")
        else:
            unit_name = list(entity_unit_map[entity_name])[unit_idx]
            prediction = f"{numeric_value:.2f} {unit_name}"
            predictions.append(prediction)
    return predictions

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred_values, true_values, pred_units, true_units, pred_entities, true_entities):
        value_loss = self.mse_loss(pred_values.squeeze(), true_values)
        unit_loss = self.ce_loss(pred_units, true_units)
        entity_loss = self.ce_loss(pred_entities, true_entities)
        return value_loss + unit_loss + entity_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Download images
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    all_images = pd.concat([train_df, test_df])['image_link']
    download_images(all_images, 'images', allow_multiprocessing=True)

    # Prepare datasets and dataloaders
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ProductImageDataset('dataset/train.csv', 'images', transform=transform)
    test_dataset = ProductImageDataset('dataset/test.csv', 'images', transform=transform, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize model
    num_entity_types = len(entity_unit_map)
    num_units = sum(len(units) for units in entity_unit_map.values())
    model = EntityExtractor(num_entity_types, num_units).to(device)

    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, train_loader, train_loader, criterion, optimizer, device, num_epochs=20)

    # Load best model and make predictions
    model.load_state_dict(torch.load('best_model.pth'))
    predictions = predict(model, test_loader, device)

    # Generate output file
    output_df = pd.DataFrame({
        'index': test_df['index'],
        'prediction': predictions
    })
    output_df.to_csv('test_out.csv', index=False)

    # Run sanity check
    os.system('python src/sanity.py --test_filename dataset/test.csv --output_filename test_out.csv')

if __name__ == "__main__":
    main()