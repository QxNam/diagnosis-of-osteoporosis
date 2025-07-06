import os
import numpy as np
import matplotlib.pyplot as plt

# Hàm lấy và hiển thị một batch
def show_batch(loader):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    images = images[:16]
    labels = labels[:16]

    images = images.to('cpu')
    labels = labels.to('cpu')

    mean = np.array([0.485])
    std = np.array([0.229])
    images = images * std[:, None, None] + mean[:, None, None]
    images = images.numpy().transpose((0, 2, 3, 1))

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()

    class_names = loader.dataset.classes

    for idx in range(images.shape[0]):
        ax = axes[idx]
        img = images[idx].squeeze()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Class: {class_names[labels[idx]]}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Hàm trực quan hóa các chỉ số
def plot_metrics(train_losses, val_losses, train_accs, val_accs, epoch):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(range(1, epoch + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, epoch + 1), val_losses, label='Val Loss', color='orange')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(range(1, epoch + 1), train_accs, label='Train Accuracy', color='green')
    plt.plot(range(1, epoch + 1), val_accs, label='Val Accuracy', color='red')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.pause(0.1)  # Cập nhật đồ thị động

def plot_training_metrics(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    figsize = (10, 5)

    # 1. Loss plot
    plt.figure(figsize=figsize)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.close()

    # 2. Accuracy plot
    plt.figure(figsize=figsize)
    plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy')
    plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy.png"))
    plt.close()

    # 3. Precision, Recall, F1, AUC plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['val_precision'], label='Precision')
    plt.plot(df['epoch'], df['val_recall'], label='Recall')
    plt.plot(df['epoch'], df['val_f1'], label='F1 Score')
    plt.plot(df['epoch'], df['val_auc'], label='AUC')
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "validation_metrics.png"))
    plt.close()

    # 4. Learning rate plot (if applicable)
    if df['learning_rate'].nunique() > 1:
        plt.figure(figsize=figsize)
        plt.plot(df['epoch'], df['learning_rate'], label='Learning Rate')
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "learning_rate.png"))
        plt.close()
    