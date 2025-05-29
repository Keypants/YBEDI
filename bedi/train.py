import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, random_split
from BEDI import bedi
from PIL import Image
import numpy as np
import os
import json
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from ptflops import get_model_complexity_info
import time
import random

# Set random seed for reproducibility
torch.manual_seed(0)

dataset_path = './Classifier_training_dataset'
unlabeled_male_path = os.path.join(dataset_path, 'male')
unlabeled_female_path = os.path.join(dataset_path, 'female')     # 未标注数据路径


# 图像大小
image_size = (256, 256)

# 数据预处理
def preprocess_image(image_path, image_size):
    image = Image.open(image_path)
    image = image.resize(image_size)
    image = np.array(image) / 255.0
    image = (image - 0.5) / 0.5  # 归一化到 [-1, 1]
    return image

# Load data from class folders
def load_unlabeled_dataset():
    images = []
    labels = []
    file_paths = []

    # 加载未标注的male数据
    unlabeled_male_files = os.listdir(unlabeled_male_path)
    for file in unlabeled_male_files:
        if file.endswith('.jpg'):
            image_path = os.path.join(unlabeled_male_path, file)
            images.append(preprocess_image(image_path, image_size))
            labels.append(1)  # 伪标签
            file_paths.append(image_path)

    # 加载未标注的female数据
    unlabeled_female_files = os.listdir(unlabeled_female_path)
    for file in unlabeled_female_files:
        if file.endswith('.jpg'):
            image_path = os.path.join(unlabeled_female_path, file)
            images.append(preprocess_image(image_path, image_size))
            labels.append(0)  # 伪标签
            file_paths.append(image_path)

    return images, labels, file_paths

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, images, labels, file_paths):
        self.images = images
        self.labels = labels
        self.file_paths = file_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        file_path = self.file_paths[idx]
        return image, label, file_path

# 加载有标注的数据集
images, labels, file_paths = load_unlabeled_dataset()

# 转换为张量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
images = torch.stack([torch.tensor(image).permute(2, 0, 1).to(device).float() for image in images])
labels = torch.tensor(labels).to(device)

# 创建数据集对象
dataset = CustomDataset(images, labels, file_paths)

# 划分训练集、验证集和测试集
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
batch_size = 8
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Build resnet model
model = bedi(pretrained=True)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Move model to the available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model.to(device)

# Train model
num_epochs = 30
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_val_accuracy = 0.0
best_model_wts = None
best_epoch = 0

# Record start time for training
train_start_time = time.time()

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    for images, labels, _ in train_data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)

    train_loss = train_loss / len(train_data_loader.dataset)
    train_accuracy = train_correct.double() / len(train_data_loader.dataset)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy.item())

    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, labels, _ in val_data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)

    val_loss = val_loss / len(val_data_loader.dataset)
    val_accuracy = val_correct.double() / len(val_data_loader.dataset)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy.item())

    # Save the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_wts = model.state_dict().copy()
        best_epoch = epoch


    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

# Record end time for training
train_end_time = time.time()

# Calculate and print the total training time
training_time = train_end_time - train_start_time
print(f'Total Training Time: {training_time:.2f} seconds')


# Function to evaluate the model
# Evaluate the best model
def evaluate_model(model_wts):
    model.load_state_dict(model_wts)
    model.eval()
    test_loss = 0.0
    test_correct = 0
    all_preds = []
    all_labels = []
    all_file_paths = []
    inference_times = []

    # Setup feature extraction with a hook
    features_list = []

    def hook_function(module, input, output):
        # Global average pooling output before the final FC layer
        features_list.append(output.cpu().numpy())

    # For ResNet-18, hook into the avgpool layer which comes before the final fc layer
    handle = model.avgpool.register_forward_hook(hook_function)

    with torch.no_grad():
        for images, labels, file_paths in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            inference_times.append(end_time - start_time)

            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            test_correct += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_file_paths.extend(file_paths)

    # Remove the hook when we're done
    handle.remove()

    # Process the collected features
    all_features = np.vstack([feat.mean(axis=(2, 3)) for feat in features_list])

    test_loss = test_loss / len(test_data_loader.dataset)
    test_accuracy = test_correct.double() / len(test_data_loader.dataset)
    test_f1_score = f1_score(all_labels, all_preds)
    test_recall = recall_score(all_labels, all_preds)
    test_conf_matrix = confusion_matrix(all_labels, all_preds)
    avg_inference_time = np.mean(inference_times)

    return test_loss, test_accuracy, test_f1_score, test_recall, test_conf_matrix, all_file_paths, all_preds, avg_inference_time, all_features, all_labels


best_model_path = './best_model_ours.pth'
torch.save(best_model_wts, best_model_path)

# Load and evaluate the best model
print("Evaluating the best model...")
best_test_loss, best_test_accuracy, best_test_f1_score, best_test_recall, best_test_conf_matrix, best_all_file_paths, best_all_preds, avg_inference_time, features, labels = evaluate_model(
    best_model_wts)
print(f'Best Epoch: {best_epoch + 1}')
print(f'Best Model Test Loss: {best_test_loss:.4f}')
print(f'Best Model Test Accuracy: {best_test_accuracy:.4f}')
print(f'Best Model Test F1 Score: {best_test_f1_score:.4f}')
print(f'Best Model Test Recall: {best_test_recall:.4f}')
print(f'Average Inference Time per Image: {avg_inference_time * 1000:.2f} ms')
print('Best Model Confusion Matrix:')
macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True,
                                         print_per_layer_stat=False, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
print(best_test_conf_matrix)

# 执行t-SNE降维
print("Performing t-SNE dimensionality reduction...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
features_tsne = tsne.fit_transform(features)

# 创建t-SNE散点图
plt.figure(figsize=(12, 10))
plt.scatter(features_tsne[np.array(labels) == 0, 0], features_tsne[np.array(labels) == 0, 1],
            color='#FF9999', marker='o', s=100, label='Female', alpha=0.7)
plt.scatter(features_tsne[np.array(labels) == 1, 0], features_tsne[np.array(labels) == 1, 1],
            color='#6666FF', marker='o', s=100, label='Male', alpha=0.7)

# plt.title('t-SNE Visualization of ResNet Features', fontsize=28)
plt.xlabel('t-SNE Dimension 1', fontsize=28)
plt.ylabel('t-SNE Dimension 2', fontsize=28)
plt.legend(fontsize=28)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 6))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', linewidth=4, linestyle='-')  # Solid line for training
plt.plot(val_losses, label='Val Loss', linewidth=4, linestyle='--')  # Dashed line for validation
plt.legend(fontsize=18)
plt.xlabel('Epoch', fontsize=30)
plt.ylabel('Loss', fontsize=30)
# plt.title('Training and Validation Loss', fontsize=26, fontname="Times New Roman")
plt.tick_params(axis='both', which='major', labelsize=26)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='green', linewidth=4, linestyle='-')  # Solid line for training
plt.plot(val_accuracies, label='Val Accuracy', color='brown', linewidth=4, linestyle='--')  # Dashed line for validation
plt.legend(fontsize=18)
plt.xlabel('Epoch', fontsize=30)
plt.ylabel('Accuracy', fontsize=30)
# plt.title('Training and Validation Accuracy', fontsize=26, fontname="Times New Roman")
plt.tick_params(axis='both', which='major', labelsize=26)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)

plt.tight_layout()

plt.show()

# Output file names and prediction results for the best model
print("Best Model Predictions:")
for file_path, pred in zip(best_all_file_paths, best_all_preds):
    print(f'File: {file_path}, Predicted Class: {pred}')

# Plot normalized confusion matrix for the best model (percentage form)
plt.figure(figsize=(10, 7))
best_test_conf_matrix_normalized = best_test_conf_matrix.astype('float') / best_test_conf_matrix.sum(axis=1)[:,
                                                                           np.newaxis] * 100
sns.heatmap(best_test_conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues',
            annot_kws={"size": 32})
plt.xlabel('Predicted', fontsize=28)
plt.ylabel('True', fontsize=28)
# plt.title("ResNet's Normalized Confusion Matrix", fontsize=28)

plt.xticks(fontsize=26)
plt.yticks(fontsize=26)

cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=26)
# for label in cbar.ax.get_yticklabels():
#     label.set_fontname()

plt.show()




