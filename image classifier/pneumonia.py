import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

import matplotlib.pyplot as plt

classes = ["No pneumonia", "pneumonia"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_layers = nn.Sequential(
      nn.Conv2d(1, 8, kernel_size=3, padding=1),  # [1, 224, 224] → [8, 224, 224]
      nn.BatchNorm2d(8),
      nn.ReLU(),
      nn.Dropout2d(0.2),
      nn.MaxPool2d(2, 2),                             # → [8, 112, 112]

      nn.Conv2d(8, 16, kernel_size=3, padding=1), # → [16, 112, 112]
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.Dropout2d(0.2),
      nn.MaxPool2d(2,2),                             # → [16, 56, 56]

      nn.Conv2d(16, 32, kernel_size=3, padding=1), # → [32, 56, 56]
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Dropout2d(0.2),
      nn.MaxPool2d(2,2),                           # → [32, 28, 28]
  )

    self.fc_layers = nn.Sequential(
      nn.AdaptiveAvgPool2d((8, 8)),  # [64, 8, 8]
      nn.Flatten(),                              # → [32 * 28 * 28]
      nn.Linear(32 * 8 * 8, 64),
      nn.ReLU(),
      nn.Dropout(.3),
      nn.Linear(64, 1)  # Binary classification
    )
    pos_weight = torch.tensor([1341 / 3875]).to(device)  # ≈ 0.346
    self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-3)
    self.accuracy = 0

  def forward(self, x):
    x = self.conv_layers(x)
    x = self.fc_layers(x)
    return x

  def train_model(self, data):
    self.train()
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0.0

    for images, labels in data:
      images, labels = images.to(device), labels.to(device).float().view(-1,1)
      self.optimizer.zero_grad()
      outputs = self(images)
      loss = self.criterion(outputs, labels)
      loss.backward()
      self.optimizer.step()

      total_loss += loss.item()
      probs = torch.sigmoid(outputs)
      preds = (probs > 0.5).float()

      # Compare predictions to labels
      correct = (preds == labels).sum().item()
      total_accuracy += correct
      total_samples += labels.size(0)

    avg_loss = total_loss / len(data)
    accuracy = total_accuracy / total_samples

    print(f'Train loss: {avg_loss}')
    print(f'Train accuracy: {accuracy}')
    print()
    
    return accuracy

  def test_model(self, data, verbose):
    self.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0.0
    
    with torch.no_grad():
      for images, labels in data:
        images, labels = images.to(device), labels.to(device).float().view(-1,1)
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        if verbose:
          img = images[0]  # select first image in batch
          img = torch.clamp(img, 0, 1)  # clamp values to [0,1]

          # Convert from [C, H, W] to [H, W, C] for plt.imshow
          plt.imshow(img.permute(1, 2, 0).cpu())  
          plt.axis('off')
          plt.show()

          print(f'Prediction: {classes[preds[0].int()]}') 
          print(f'Actual: {classes[labels[0].int()]}')
          print()

        # Compare predictions to labels
        correct = (preds == labels).sum().item()
        total_accuracy += correct
        total_samples += labels.size(0)

    avg_loss = total_loss / len(data)
    self.accuracy = total_accuracy / total_samples
    
    if not verbose:
      print(f'Test loss: {avg_loss}')
      print(f'Test accuracy: {self.accuracy}')
    else:
      print(f'Test accuracy: {100 * self.accuracy}%')
    print()

    return self.accuracy
    
