from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import shutil, os

def split_and_move_data(data_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'normal'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'osteoporosis'), exist_ok=True)

    # Lấy danh sách file từ hai lớp
    normal_files = [os.path.join(data_dir, 'normal', f) for f in os.listdir(os.path.join(data_dir, 'normal')) 
        if f.endswith(('.jpg', '.png', '.jpeg'))]
    osteoporosis_files = [os.path.join(data_dir, 'osteoporosis', f) for f in os.listdir(os.path.join(data_dir, 'osteoporosis')) 
        if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Chia tập normal
    normal_train, temp = train_test_split(normal_files, train_size=train_ratio, random_state=42)
    normal_val, normal_test = train_test_split(temp, train_size=val_ratio/(val_ratio+test_ratio), random_state=42)

    # Chia tập osteoporosis
    osteoporosis_train, temp = train_test_split(osteoporosis_files, train_size=train_ratio, random_state=42)
    osteoporosis_val, osteoporosis_test = train_test_split(temp, train_size=val_ratio/(val_ratio+test_ratio), random_state=42)

    # Di chuyển file
    for files, split in [(normal_train, 'train'), (normal_val, 'val'), (normal_test, 'test'),
                        (osteoporosis_train, 'train'), (osteoporosis_val, 'val'), (osteoporosis_test, 'test')]:
        for src_file in files:
            class_name = 'normal' if 'normal' in src_file else 'osteoporosis'
            dest_dir = os.path.join(output_dir, split, class_name)
            shutil.copy(src_file, dest_dir)

    print(f"Chia dữ liệu thành công:")
    print(f"Train: {len(normal_train) + len(osteoporosis_train)} ảnh")
    print(f"Validation: {len(normal_val) + len(osteoporosis_val)} ảnh")
    print(f"Test: {len(normal_test) + len(osteoporosis_test)} ảnh")

# Hàm tạo data loaders với augmentation
def get_data_loaders(data_dir, batch_size=32, is_transform=False):
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    val_test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    if is_transform is True:
        train_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        val_test_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_test_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader, train_dataset.classes
