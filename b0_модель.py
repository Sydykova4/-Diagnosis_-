import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from collections import defaultdict
import ast
from typing import Sized, List, Dict


class Config:
    def __init__(self):
        self.data_path = "odir5k"
        self.csv_path = os.path.join(self.data_path, "final_dataset.csv")
        self.image_dir = os.path.join(self.data_path, "cleaned_images")
        self.batch_size = 64
        self.num_epochs = 30
        self.lr = 1e-3
        self.num_classes = 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.early_stop_patience = 5
        self.plot_dir = "training_plots_b0"
        os.makedirs(self.plot_dir, exist_ok=True)

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")


config = Config()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            alpha = self.alpha[targets]
            focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


class EyeDataset(Dataset, Sized):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.classes: List[str] = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'S']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        try:
            img_path = os.path.join(self.img_dir, self.df['filename'].iloc[idx])
            with Image.open(img_path) as img:
                image = self.transform(img.convert("RGB")) if self.transform else img

            label_str = ast.literal_eval(self.df['labels'].iloc[idx])[0]
            label = self.class_to_idx[label_str]

            return image, label
        except Exception as e:
            print(f"Error loading {self.df['filename'].iloc[idx]}: {str(e)}")
            return self[np.random.randint(0, len(self))]


def get_transforms() -> tuple:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


class EyeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        self.classes = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'S']

        for param in self.base_model.blocks[-4:].parameters():
            param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model(x)
        return self.classifier(features)


def load_data() -> tuple:
    df = pd.read_csv(config.csv_path)

    df['label'] = df['labels'].apply(lambda x: ast.literal_eval(x)[0])
    print("\nClass distribution:")
    print(df['label'].value_counts())

    class_counts = df['label'].value_counts().sort_index().values
    class_weights = 1. / (class_counts ** 0.5)
    class_weights_dict = dict(zip(sorted(df['label'].unique()), class_weights))
    sample_weights = df['label'].map(class_weights_dict).values

    train_df, val_df = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        stratify=df['label']
    )

    print("\nTrain class distribution:")
    print(train_df['label'].value_counts())
    print("\nValidation class distribution:")
    print(val_df['label'].value_counts())

    train_transform, val_transform = get_transforms()
    train_dataset = EyeDataset(train_df, config.image_dir, train_transform)
    val_dataset = EyeDataset(val_df, config.image_dir, val_transform)

    sampler = WeightedRandomSampler(
        weights=sample_weights[train_df.index].astype(np.float32),
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, class_weights


def plot_training_history(history: dict, save_dir: str) -> None:
    plt.figure(figsize=(15, 10))

    metrics = [
        ('Loss', ['train_loss', 'val_loss']),
        ('Accuracy', ['train_acc', 'val_acc']),
        ('F1 Score', ['f1_weighted', 'f1_macro']),
        ('Learning Rate', ['lr'])
    ]

    for i, (title, keys) in enumerate(metrics, 1):
        plt.subplot(2, 2, int(i))
        for key in keys:
            if key in history:
                plt.plot(history['epoch'], history[key], label=key.replace('_', ' ').title())
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=300)
    plt.close()

    # Сохраняем историю обучения в CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)

    # Отдельно сохраняем метрики для каждой эпохи
    metrics_df = history_df[['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc',
                             'f1_weighted', 'f1_macro', 'lr']]
    metrics_df.to_csv(os.path.join(save_dir, 'epoch_metrics.csv'), index=False)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             save_cm: bool = False, epoch: int = None) -> Dict:
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(config.device)
            labels = labels.to(config.device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Конвертируем в numpy массивы
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Вычисляем метрики
    accuracy = correct / len(loader.dataset)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average=None)
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # Вычисляем accuracy для каждого класса
    class_accuracy = []
    for cls in range(len(model.classes)):
        cls_mask = (all_labels == cls)
        if np.sum(cls_mask) > 0:  # Используем np.sum вместо sum
            class_acc = np.mean(all_preds[cls_mask] == all_labels[cls_mask])
            class_accuracy.append(class_acc)
        else:
            class_accuracy.append(0.0)

    # Сохраняем confusion matrix
    if save_cm:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=model.classes,
                    yticklabels=model.classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix (Epoch {epoch})' if epoch else 'Confusion Matrix')
        plt.savefig(
            os.path.join(config.plot_dir, f'confusion_matrix_epoch{epoch}.png' if epoch else 'confusion_matrix.png'),
            dpi=300, bbox_inches='tight')
        plt.close()

    metrics = {
        'val_loss': total_loss / len(loader),
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'recall': dict(zip(model.classes, recall)),
        'precision': dict(zip(model.classes, precision)),
        'class_accuracy': dict(zip(model.classes, class_accuracy)),
        'confusion_matrix': cm
    }

    print("\nDetailed Metrics:")
    print(f"{'Class':<5} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 40)
    for cls, acc, prec, rec in zip(model.classes, class_accuracy, precision, recall):
        print(f"{cls:<5} {acc:.3f}{'':<7} {prec:.3f}{'':<7} {rec:.3f}")
    print("\nGlobal Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 weighted: {f1_weighted:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")

    return metrics


def train_model() -> None:
    train_loader, val_loader, class_weights = load_data()
    model = EyeModel().to(config.device)

    alpha = torch.tensor(class_weights, device=config.device)
    criterion = FocalLoss(alpha=alpha)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
    )

    history = defaultdict(list)
    best_f1 = 0.0
    best_metrics = {}
    no_improve = 0

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}"):
            images = images.to(config.device)
            labels = labels.to(config.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()

        train_acc = correct / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, criterion)
        scheduler.step(val_metrics['f1_weighted'])

        # Сохраняем метрики в историю
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['f1_weighted'].append(val_metrics['f1_weighted'])
        history['f1_macro'].append(val_metrics['f1_macro'])
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Обновляем графики обучения каждый эпоху
        plot_training_history(history, config.plot_dir)

        print(f"\nEpoch {epoch + 1} Results:")
        print(f"Train Loss: {history['train_loss'][-1]:.4f}, Acc: {train_acc:.4f}")
        print(f"Val Loss: {history['val_loss'][-1]:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val F1 (weighted): {val_metrics['f1_weighted']:.4f}, F1 (macro): {val_metrics['f1_macro']:.4f}")

        # Сохраняем только лучшую модель и ее confusion matrix
        if val_metrics['f1_weighted'] > best_f1:
            best_f1 = val_metrics['f1_weighted']
            best_metrics = val_metrics
            no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'metrics': best_metrics
            }, os.path.join(config.plot_dir, "training_plots_b0/best_model.pth"))

            # Сохраняем confusion matrix только для лучшей модели
            evaluate(model, val_loader, criterion, save_cm=True, epoch=epoch + 1)
            print(f"New best model saved with F1: {best_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= config.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        torch.cuda.empty_cache()

    # Сохраняем финальные метрики лучшей модели
    best_metrics_df = pd.DataFrame({
        'best_epoch': [best_metrics.get('epoch', epoch + 1)],
        'best_f1_weighted': [best_f1],
        'best_f1_macro': [best_metrics.get('f1_macro', 0)],
        **{f'recall_{cls}': [best_metrics.get('recall', {}).get(cls, 0)]
           for cls in model.classes}
    })
    best_metrics_df.to_csv(os.path.join(config.plot_dir, 'best_model_metrics.csv'), index=False)


if __name__ == "__main__":
    train_model()
