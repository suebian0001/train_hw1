import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import random

# Set the experiment name
_exp_name = "sample"

# Set seed for reproducibility
myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# Image transformations for training and testing
train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])



# Dataset class for food images
class FoodDataset(Dataset):
    def __init__(self, path, tfm=test_tfm, files=None):
        super(FoodDataset, self).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files is not None:
            self.files = files
        print(f"One {path} sample:", self.files[0])
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        try:
            im = Image.open(fname)

            if im.mode == 'RGBA':
                im = im.convert('RGB')

            im = self.transform(im)

            try:
                label = int(os.path.basename(fname).split("_")[0])
            except ValueError:
                label = -1
        except OSError:
            print(f"Skip error image: {fname}")
            return None

        return im, label

# # Simple classifier model (can be updated)
# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         pass

#     def forward(self, x):
#         pass

# updated 1st
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # Define convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 3 input channels (RGB), 16 output channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 16 input channels, 32 output channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        num_classes = 11
        # Define fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 32 * 32, 256),  # Adjust the input size based on the image dimensions
            nn.ReLU(),
            nn.Linear(256, num_classes)  # Assuming 10 output classes
        )
    
    def forward(self, x):
        x = self.conv_layers(x)  # Pass input through the conv layers
        x = x.view(x.size(0), -1)  # Flatten the tensor before passing to fully connected layers
        x = self.fc_layers(x)  # Pass through the fc layers
        return x

# Set device for model training/inference
device = "cuda" if torch.cuda.is_available() else "cpu"

# path is ".venv\training"
# 加載訓練和測試數據
train_data_path = "./training"  # 解壓後的訓練數據路徑
test_data_path = "./testing"    # 解壓後的測試數據路徑

train_set = FoodDataset(train_data_path, tfm=train_tfm)
test_set = FoodDataset(test_data_path, tfm=test_tfm)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)


# update 1st
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# print the model architecture
# print(model)  

# find total params
# # Optional: Print the number of parameters in the model
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params}")

# check labels range
# for images, labels in train_loader:
#     print(labels)  # Check if labels are within range
#     break



# 訓練模型 1st
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 儲存最佳模型
torch.save(model.state_dict(), f"{_exp_name}_best.ckpt")

# 評估模型
model.eval()
prediction = []
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        test_pred = model(data)
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()

def pad4(i):
    return "0" * (4 - len(str(i))) + str(i)

df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1, len(test_set)+1)]
df["Category"] = prediction
df.to_csv("submission.csv", index=False)


# # 2nd version
# import numpy as np
# import pandas as pd
# import torch
# import os
# import torch.nn as nn
# import torchvision.transforms as transforms
# from PIL import Image
# from torch.utils.data import DataLoader, Dataset, Subset
# from sklearn.model_selection import train_test_split
# from tqdm.auto import tqdm
# import random

# # Set the experiment name
# _exp_name = "sample"

# # Set seed for reproducibility
# myseed = 6666
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(myseed)
# torch.manual_seed(myseed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(myseed)

# # Image transformations for training and testing
# train_tfm = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.ToTensor(),
# ])

# test_tfm = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
# ])

# # Dataset class for food images
# class FoodDataset(Dataset):
#     def __init__(self, path, tfm=test_tfm, files=None):
#         super(FoodDataset, self).__init__()
#         self.path = path
#         self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
#         if files is not None:
#             self.files = files
#         print(f"One {path} sample:", self.files[0])
#         self.transform = tfm

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         fname = self.files[idx]
#         try:
#             im = Image.open(fname)

#             if im.mode == 'RGBA':
#                 im = im.convert('RGB')

#             im = self.transform(im)

#             try:
#                 label = int(os.path.basename(fname).split("_")[0])
#             except ValueError:
#                 label = -1
#         except OSError:
#             print(f"Skip error image: {fname}")
#             return None

#         return im, label

# # Improved Classifier model
# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#         )
#         self.fc_layers = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(128 * 16 * 16, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 11)
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc_layers(x)
#         return x

# # Set device for model training/inference
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Paths for train and test data
# train_data_path = "./training"  
# test_data_path = "./testing"    

# train_set = FoodDataset(train_data_path, tfm=train_tfm)
# test_set = FoodDataset(test_data_path, tfm=test_tfm)

# # Split train data into train/validation sets
# train_idx, val_idx = train_test_split(list(range(len(train_set))), test_size=0.2, random_state=myseed)
# train_subset = Subset(train_set, train_idx)
# val_subset = Subset(train_set, val_idx)

# train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# # Define the model, loss, and optimizer
# model = Classifier().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Learning rate scheduler
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# # Training loop with validation and early stopping
# num_epochs = 10
# best_loss = float('inf')
# early_stop_count = 0
# early_stop_patience = 3

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, labels in tqdm(train_loader):
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     epoch_loss = running_loss / len(train_loader)
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

#     # Validation step
#     model.eval()
#     val_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()

#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     val_loss = val_loss / len(val_loader)
#     val_acc = 100 * correct / total
#     print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')

#     # Save best model
#     if val_loss < best_loss:
#         best_loss = val_loss
#         torch.save(model.state_dict(), f"{_exp_name}_best.ckpt")
#         early_stop_count = 0
#     else:
#         early_stop_count += 1

#     # Early stopping
#     if early_stop_count >= early_stop_patience:
#         print("Early stopping triggered")
#         break

#     scheduler.step()

# # Load best model and generate test predictions
# model.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
# model.eval()
# prediction = []
# with torch.no_grad():
#     for data, _ in test_loader:
#         data = data.to(device)
#         test_pred = model(data)
#         test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
#         prediction += test_label.squeeze().tolist()

# # Prepare submission file
# def pad4(i):
#     return "0" * (4 - len(str(i))) + str(i)

# df = pd.DataFrame()
# df["Id"] = [pad4(i) for i in range(1, len(test_set)+1)]
# df["Category"] = prediction
# df.to_csv("submission.csv", index=False)
