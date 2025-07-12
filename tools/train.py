import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from torch_optimizer import RAdam, Lookahead
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from collections import deque
from .losses import get_loss_function
from .utils import get_parameters, export_args
from .viz import plot_training_metrics

# Default DEVICE
DEVICE = "cpu"

# Check for CUDA
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    device_id = 0
    if device_count > 1:
        try:
            device_id = int(input("Chọn cuda: "))
        except:
            device_id = 0
            print("Không hợp lệ, mặc định device_id = 0")
    
    DEVICE = f"cuda:{device_id}"
    device_name = torch.cuda.get_device_name(device_id)
    current_device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_id)

    # Memory usage (in MB)
    total_memory_mb = props.total_memory / (1024 ** 2)
    allocated_memory_mb = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
    reserved_memory_mb = torch.cuda.memory_reserved(device_id) / (1024 ** 2)

    print("🚀 CUDA available")
    print(f"💻 CUDA DEVICE count: {device_count}")
    print(f"📦 Current DEVICE: {current_device}")
    print(f"🖥️ DEVICE name: {device_name}")
    print(f"💾 Total memory: {total_memory_mb:.2f} MB")
    print(f"🧠 Allocated memory: {allocated_memory_mb:.2f} MB")
    print(f"📚 Reserved memory: {reserved_memory_mb:.2f} MB")

# Check for MPS (Metal Performance Shaders - macOS only)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    import psutil
    total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"🧠 Total Unified Memory (RAM + GPU): {total_memory_gb:.2f} GB")

    DEVICE = "mps"
    print("🍏 MPS (Metal Performance Shaders) is available.")
    print("📱 Using MPS for acceleration on macOS.")

else:
    print("🧊 Using CPU. No GPU (CUDA or MPS) available.")

print("✅ Selected DEVICE:", DEVICE)

# Hàm đánh giá mỗi epoch train mô hình
def evaluate_predictions(y_true, y_pred):
    """
    Tính toán các chỉ số đánh giá mô hình phân loại.

    Params:
    - y_true: danh sách hoặc mảng numpy chứa ground truth labels
    - y_pred: danh sách hoặc mảng numpy chứa predicted labels

    Returns:
    - metrics: dict chứa accuracy, precision, recall, f1, auc
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'auc': roc_auc_score(y_true, y_pred, multi_class='ovr') if len(set(y_true)) > 2 else roc_auc_score(y_true, y_pred)
    }
    return metrics

# Hàm huấn luyện mô hình
def train_model(model, train_loader, val_loader, resume=False, **kwargs):
    '''
    Params:
    - model: Mô hình để huấn luyện
    - train_loader: Bộ dữ liệu đã được gom thành từng batch để train
    - val_loader: Bộ dữ liệu đã được gom thành từng batch để đánh giá trong lúc train
    **kwargs: các tham số khác
        + device = None
        + num_epochs: int = 100                     - số lượng epoch (mặc định 100)
        + lr: float = 1e-3                          - tốc độ học (mặc định 1e-3)
        + flag: str = model._get_name()             - tên file log quá trình training
        + early_stopping_patience: int = 10         - theo dõi 10 epoch gần nhất
        + early_stopping_delta: float = 1e-3        - nếu dao động nhỏ hơn 0.001 thì dừng
        + loss: str = "CrossEntropy"                - tên hàm loss sử dụng [CrossEntropy, Focal, LabelSmoothing]
        + optimizer: str = "Adam"                   - tên hàm tối ưu [Adam, Adamw, Radam, Lookahead, Sgd]
        + scheduler: str = "ReduceLROnPlateau"      - tên hàm cập nhật tốc độ học [ReduceLROnPlateau, Cosine, CosineRestart, Onecycle]
        + focal_alpha: list = [0.25, 0.75]          - tham số cho hàm loss là 'Focal'
        + focal_gamma: float = 2.0                  - tham số cho hàm loss là 'Focal'
        + smoothing: float = 0.1                    - tham số cho hàm loss là 'LabelSmoothing'
    '''
    metadata = {}

    num_epochs = kwargs.get("num_epochs", 200)
    lr = kwargs.get("lr", 1e-3)
    early_stopping_patience = kwargs.get("early_stopping_patience", 10) 
    early_stopping_delta = kwargs.get("early_stopping_delta", 1e-3)  
    recent_val_losses = deque(maxlen=early_stopping_patience)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # Chọn hàm tối ưu
    optimizer_name = kwargs.get("optimizer", "adam").lower()
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name == "radam":
        optimizer = RAdam(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name == "lookahead":
        base_opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        optimizer = Lookahead(base_opt)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
    # chọn hàm loss
    loss_name = kwargs.get("loss", "CrossEntropy").lower()
    criterion = get_loss_function(loss_name, kwargs)
    
    # chọn hàm cập nhật lr
    scheduler_name = kwargs.get("scheduler", "ReduceLROnPlateau").lower()
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", 20),
            eta_min=kwargs.get("eta_min", 1e-6)
        )
    elif scheduler_name == "cosinerestart":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get("T_0", 10),
            T_mult=kwargs.get("T_mult", 2)
        )
    elif scheduler_name == "onecycle":
        if train_loader is None:
            raise ValueError("train_loader is required for OneCycleLR")
        scheduler = OneCycleLR(
            optimizer,
            max_lr=kwargs.get("max_lr", 0.01),
            steps_per_epoch=len(train_loader),
            epochs=kwargs.get("num_epochs", 10),
            pct_start=kwargs.get("pct_start", 0.3)
        )

    flag = kwargs.get("flag", f"{model._get_name()}_{loss_name}_{optimizer_name}_{scheduler_name}") # {datetime.now().strftime('%Y%m%d_%H%M%S')}
    # Các đường dẫn thư mục
    train_dir = "training_seg"
    metadata_dir = os.path.join(train_dir, flag)
    os.makedirs(metadata_dir, exist_ok=True)

    csv_path = os.path.join(metadata_dir, "train_log.csv")
    best_model_path = os.path.join(metadata_dir, f"best.pth")
    last_model_path = os.path.join(metadata_dir, f"last.pth")
    
    # CSV file
    columns = [
        'epoch', 'train_loss', 'val_loss',
        'train_acc', 'val_acc',
        'val_precision', 'val_recall', 'val_f1', 'val_auc',
        'learning_rate'
    ]
    df = pd.DataFrame(columns=columns)
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False)
        
    # Load last model nếu ở chế độ tiếp tục train
    index_start = 0
    best_val_loss = float('inf')
    if resume and os.path.exists(last_model_path):
        print(f"🕹️ Continue training at {metadata_dir}")
        index_start = pd.read_csv(csv_path).shape[0]
        model.load_state_dict(torch.load(last_model_path))
        print(f"🔁 Loaded checkpoint from {last_model_path}")
        df = pd.read_csv(csv_path)
        lr = float(df.iloc[-1]['learning_rate'])
        # Load optimizer, best_val_loss, epoch nếu có
        opt_ckpt_path = os.path.join(metadata_dir, 'optimizer_last.pth')
        meta_ckpt_path = os.path.join(metadata_dir, 'train_meta_last.pth')
        if os.path.exists(opt_ckpt_path):
            optimizer.load_state_dict(torch.load(opt_ckpt_path))
        if os.path.exists(meta_ckpt_path):
            meta = torch.load(meta_ckpt_path)
            best_val_loss = meta.get('best_val_loss', best_val_loss)
            index_start = meta.get('epoch', index_start)
            lr = meta.get('lr', lr)
    else:
        print(f"🚀 New training saved at: {best_model_path}")

    # lưu args
    images, labels = next(iter(train_loader))
    metadata = {
        "Time": datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
        "model": model._get_name(),
        "parameters": get_parameters(model),
        "dir": metadata_dir,
        "num_epochs": index_start+num_epochs,
        "run_epoch": index_start,
        "learning_rate": lr,
        "optimizer_name": optimizer_name,
        "loss_name": loss_name,
        "scheduler": scheduler_name,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_delta": early_stopping_delta,
        "best_loss": None,
        "batch_size": train_loader.batch_size,
        "train_images": len(train_loader.dataset),
        "valid_images": len(val_loader.dataset),
        "image_shape": images[0].shape
    }
    metadata.update(kwargs)
    export_args(path=os.path.join(metadata_dir, "args.yml"), metadata=metadata)
    print(f"💾 Saved information at {os.path.join(metadata_dir, 'args.yml')}")
    
    # huấn luyện
    for epoch in range(index_start, index_start+num_epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        for inputs, labels in tqdm(train_loader, desc=f'➜ Epoch {epoch+1}', colour='green', ncols=100):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_metrics = evaluate_predictions(train_labels, train_preds)
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        
        # Valication
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_metrics = evaluate_predictions(val_labels, val_preds)
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | AUC: {val_metrics['auc']:.4f}")

        # Save to CSV
        row = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_metrics['accuracy'],
            'val_acc': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_auc': val_metrics['auc'],
            'learning_rate': current_lr
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        pd.DataFrame([row]).to_csv(csv_path, mode='a', header=False, index=False)
        
        # Save figure metrics
        plot_training_metrics(df, save_dir=metadata_dir)

        # Save weights
        torch.save(model.state_dict(), last_model_path)
        # Save optimizer, best_val_loss, epoch, lr
        torch.save(optimizer.state_dict(), os.path.join(metadata_dir, 'optimizer_last.pth'))
        torch.save({'best_val_loss': best_val_loss, 'epoch': epoch+1, 'lr': lr}, os.path.join(metadata_dir, 'train_meta_last.pth'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"\t💽 Saved best model at epoch {epoch+1}")
            
        # check early stop
        recent_val_losses.append(val_loss)
        cur_epoch = epoch + 1
        if len(recent_val_losses) == early_stopping_patience:
            max_loss = max(recent_val_losses)
            min_loss = min(recent_val_losses)
            if max_loss - min_loss < early_stopping_delta:
                print(f"🛑 Early stopping: val_loss variation < {early_stopping_delta} over last {early_stopping_patience} epochs.")
                break
        if scheduler_name=="onecycle".lower():
            scheduler.step()
        elif scheduler_name=="ReduceLROnPlateau".lower():
            scheduler.step(val_loss)
    print(f"✅ Training complete")
    # cập nhật và lưu lại metadata
    metadata = {
        "Time": datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
        "model": model._get_name(),
        "parameters": get_parameters(model),
        "dir": metadata_dir,
        "num_epochs": index_start+num_epochs,
        "run_epoch": cur_epoch,
        "learning_rate": lr,
        "optimizer_name": optimizer_name,
        "loss_name": loss_name,
        "scheduler": scheduler_name,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_delta": early_stopping_delta,
        "best_loss": best_val_loss,
        "batch_size": train_loader.batch_size,
        "train_images": len(train_loader.dataset),
        "valid_images": len(val_loader.dataset),
        "image_shape": images[0].shape
    }
    metadata.update(kwargs)
    export_args(path=os.path.join(metadata_dir, "args.yml"), metadata=metadata)
    print(f"💾 Saved information at {os.path.join(metadata_dir, 'args.yml')}")
    return metadata_dir

# Hàm đánh giá mô hình
def evaluate_model(model, test_loader, model_path, class_names=None, save_dir=None):
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_preds)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test AUC: {auc:.4f}")
    images, labels = next(iter(test_loader))
    metadata = {
        "Time": datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
        "model": model._get_name(),
        "parameters": get_parameters(model),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1_Score": f1,
        "AUC": auc,
        "test_images": len(test_loader.dataset),
        "image_shape": images[0].shape
    }
    export_args(path=os.path.join(save_dir, "test_metrics.yml"), metadata=metadata)
    print(f"💾 Saved information at {os.path.join(save_dir, 'test_metrics.yml')}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    labels = class_names if class_names else ['Class 0', 'Class 1']

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    if save_dir:
        save_fig_path = os.path.join(save_dir, "confusion_matix")
        plt.savefig(save_fig_path, bbox_inches='tight')
        print(f"✅ Confusion matrix saved at {save_dir}")
    else:
        plt.show()
