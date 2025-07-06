# 🩻 Chuẩn đoán bệnh loãng xương ảnh X-ray

Dự án xây dựng hệ thống phát hiện và phân loại bất thường trong ảnh X-quang sử dụng học sâu. Ứng dụng bao gồm các bước: 
- tăng số lượng ảnh (x3)
- phát hiện vùng ROI (Region of Interest), 
- tăng cường chất lượng ảnh
- phân loại bằng mô hình ResNet50 hoặc VGG16. 

Giao diện tương tác được triển khai bằng `Gradio` và hỗ trợ chạy thông qua `Docker`.
---

## 🚀 Tính năng chính
- ✅ Phát hiện vùng ROI trên ảnh X-quang bằng mô hình YOLOv8n.
- ✅ Tăng cường ảnh (CLAHE + Gamma Correction).
- ✅ Phân loại ảnh ROI bằng mô hình ResNet50 hoặc VGG16 (tự huấn luyện).
- ✅ Giao diện người dùng thân thiện với Gradio.
- ✅ Dễ dàng triển khai bằng Docker.

---

## 🧠 Huấn luyện mô hình
### 1. Huấn luyện mô hình phân loại ResNet50 / VGG16
- Để tạo ra data x3 và xử lý, đọc qua `notebook/process.ipynb`.
- Notebook: `notebook\train_3_ROI_enhance.ipynb`
- Dataset: Ảnh đã cắt sẵn ROI tự động từ YOLO và xử lý tăng cường.
- Output: checkpoint `.pth` được lưu trong [Drive](https://drive.google.com/drive/u/0/folders/13ytPRe5ovAv3Tm4Bz0cCIDNbKF8Fh7WC)

### 2. Huấn luyện mô hình phát hiện ROI (YOLOv8n)

- Kết quả model `best.pt` nằm trong [Drive](https://drive.google.com/drive/u/0/folders/1sKplzI4UfngL6dDQzxp2pBI0XbsC2W4P)
---

## 🖥️ Cách sử dụng
- Tạo `venv`, sử dụng python >=10:
```bash
➜ python -m venv venv
```
- Khởi động venv:
```bash
# đối với linux/mac
➜ source venv/bin/activate

# đối với window
➜ venv\Script\activate
```
Sau khi chạy lệnh hiển thị venv sau mũi tên là được:
```bash
(venv) ➜
```
- Cài đặt các thư viện cần thiết
```bash
(venv) ➜ pip install -r requirements.txt 
```

- Sử dụng jupyter notebook server làm kernel
```bash
(venv) ➜ jupyter notebook
```


**Sử dụng app**:
Để chạy app có thể

Cách 1: chạy bằng python
```bash
(venv) ➜ cd app
(venv) ➜ python app.py
```

Cách 2: sử dụng docker
```bash
(venv) ➜ docker compose up -d
```

truy cập đường dẫn: [localhost:7860](http://localhost:7860/)
