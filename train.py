import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

class EEGAugmentation:
    def __init__(self, p=0.5, device='cuda'):
        self.p = p
        self.device = device
        
    def add_gaussian_noise(self, data, std=0.1):
        if torch.rand(1).to(self.device) < self.p:
            return data + torch.randn_like(data).to(self.device) * std
        return data
    
    def time_shift(self, data, max_shift=10):
        if torch.rand(1).to(self.device) < self.p:
            shift = torch.randint(-max_shift, max_shift, (1,)).to(self.device)
            return torch.roll(data, shifts=shift.item(), dims=1)
        return data
    
    def amplitude_scale(self, data, range=(0.8, 1.2)):
        if torch.rand(1).to(self.device) < self.p:
            scale = (torch.rand(1) * (range[1] - range[0]) + range[0]).to(self.device)
            return data * scale
        return data

def sliding_window_preprocess(data, labels, window_size=370, overlap=0.5):
    """使用滑动窗口处理数据"""
    stride = int(window_size * (1 - overlap))
    windows = []
    window_labels = []
    
    for i in range(0, len(data) - window_size + 1, stride):
        window = data[i:i + window_size]
        # 使用众数作为窗口的标签
        label = np.bincount(labels[i:i + window_size]).argmax()
        windows.append(window)
        window_labels.append(label)
    
    return np.array(windows), np.array(window_labels)

def load_data(window_size=370, overlap=0.5):
    gestures = [f'gesture{i+1}' for i in range(10)]
    data = []
    labels = []

    for i, gesture in enumerate(gestures):
        gesture_dir = f"D:/TSD/emotive/emotiv_project/processed_dataset_v5/{gesture}"
        try:
            for group_file in os.listdir(gesture_dir):
                if group_file.endswith('.npy'):
                    file_path = os.path.join(gesture_dir, group_file)
                    eeg_data = np.load(file_path)
                    print(f"Loaded {file_path} with shape {eeg_data.shape}")
                    data.append(eeg_data)
                    labels.append(np.full(eeg_data.shape[0], i))
        except Exception as e:
            print(f"Error loading data for {gesture}: {e}")
            continue

    # 合并数据
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    # 应用滑动窗口预处理
    data, labels = sliding_window_preprocess(data, labels, window_size, overlap)

    # 划分数据集
    X_temp, X_test, y_temp, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train_shape = X_train.shape
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train_shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # 转换为PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 计算类别权重
    class_counts = np.bincount(y_train.numpy())
    total_samples = len(y_train)
    class_weights = torch.FloatTensor([total_samples / (len(class_counts) * count) 
                                     for count in class_counts])

    print("\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"X_test: {X_test.shape}")
    print("\nClass distribution:")
    for i in range(len(class_counts)):
        print(f"Class {i}: {class_counts[i]} samples")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, class_weights

class Improved3DEEGClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_classes=10):
        super().__init__()
        
        # 时间卷积层
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 空间注意力
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=8, dropout=0.1
        )
        
        # GRU层
        self.gru = nn.GRU(64, hidden_size, num_layers=2, 
                         batch_first=True, bidirectional=True,
                         dropout=0.3)
        
        # 自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size*2, num_heads=8, dropout=0.1
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, time_steps, channels)
        
        # 时间卷积
        x = x.permute(0, 2, 1)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1)
        
        # 空间注意力
        x_att, _ = self.spatial_attention(x, x, x)
        x = x + x_att
        
        # GRU处理
        gru_out, _ = self.gru(x)
        
        # 自注意力
        att_out, _ = self.self_attention(gru_out, gru_out, gru_out)
        x = gru_out + att_out
        
        # 全局池化
        x = torch.mean(x, dim=1)
        
        # 分类
        x = self.fc(x)
        return x

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, 
                epochs=80, batch_size=64, learning_rate=0.001, device='cuda'):
    
    # 初始化wandb
    wandb.init(
        project="EEG-Classification",
        config={
            "architecture": "Improved3DEEGClassifier",
            "dataset": "EEG-Gestures",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "AdamW",
            "scheduler": "OneCycleLR"
        }
    )
    
    # 记录模型架构
    wandb.watch(model)
    
    # 数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train.to(device), y_train.to(device))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(X_val.to(device), y_val.to(device))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 计算类别权重
    class_counts = np.bincount(y_train.cpu().numpy())
    weights = torch.FloatTensor([1.0 / count for count in class_counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                            epochs=epochs, steps_per_epoch=len(train_loader))
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_acc = 0
    best_model = None
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # 记录到wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss/len(train_loader),
            "train_acc": train_acc,
            "val_loss": val_loss/len(val_loader),
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        print(f'Epoch: {epoch+1}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print('Early stopping!')
            break
    
    # 加载最佳模型
    model.load_state_dict(best_model)
    
    # 结束wandb运行
    wandb.finish()
    
    return model, best_val_acc

def test_model(model, X_test, y_test, device):
    model.eval()
    test_dataset = torch.utils.data.TensorDataset(X_test.to(device), y_test.to(device))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_targets = []
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_total += targets.size(0)
            test_correct += (predicted == targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_acc = 100 * test_correct / test_total
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds))
    
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Final Test Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('final_confusion_matrix.png')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, class_weights = load_data()

    # 初始化模型
    input_size = X_train.shape[2]  # 通道数
    model = Improved3DEEGClassifier(input_size)
    model.to(device)

    # 训练模型
    model, best_val_acc = train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, device=device)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # 测试模型
    test_model(model, X_test, y_test, device)

if __name__ == "__main__":
    main()