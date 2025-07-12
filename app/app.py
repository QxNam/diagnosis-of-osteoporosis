import gradio as gr
import numpy as np
from utils import enhance_xray, extract_roi, predict_cls, segment_roi
import cv2

def process_and_classify(raw_image: np.ndarray, processing_mode: str, model_choice: str):
    if raw_image is None:
        return None, None, None, "Vui lòng tải ảnh lên để bắt đầu."

    # if processing_mode != "ROI Extract":
    #     return raw_image, raw_image, raw_image, "Chỉ hỗ trợ Simple ROI trong phiên bản này."

    # Trích xuất ROI
    rois_draw, rois = extract_roi(image=raw_image)
    if not rois:
        return raw_image, rois_draw, None, "Không tìm thấy ROI nào."
    
    # Tăng cường ảnh
    enhanceds = []
    for roi in rois:
        if processing_mode.lower() == "segment":
            roi = segment_roi(roi)
        if not isinstance(roi, np.ndarray):
            roi = np.array(roi)
        if len(roi.shape) == 3 and roi.shape[2] == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        enhanced = enhance_xray(image=gray)
        enhanceds.append(enhanced)
    # Dự đoán kết quả
    predictions, transformed_images = predict_cls(model_choice.lower(), enhanceds)

    # Dự đoán kết quả
    results = []
    for i, pred in enumerate(predictions):
        print(f'🟢 ROI {i}, predict: {pred}')
        result_str = {
            pred[0]: pred[1]
        }
        results.append(result_str)
    if len(enhanceds)==1:
        return rois_draw, transformed_images[0], None, results[0], None
    return rois_draw, transformed_images[0], transformed_images[1], results[0], results[1]

# def process_and_classify(raw_image: np.ndarray, processing_mode: str, model_choice: str):
#     return None, None, None, None, None

# --- XÂY DỰNG GIAO DIỆN GRADIO ---

with gr.Blocks(theme=gr.themes.Soft(), title="Phân loại ảnh X-quang") as app:
    gr.Markdown(
        """
        # Ứng dụng Phân loại Ảnh Y tế X-quang
        Tải lên một ảnh X-quang, chọn chế độ xử lý, chọn mô hình và xem kết quả.
        """
    )

    with gr.Row():
        # Cột bên trái cho input
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="Ảnh X-quang đầu vào")
            
            mode_selector = gr.Dropdown(
                choices=["ROI Extract", "Segment"],
                value="ROI Extract",
                label="Chọn Chế độ Xử lý"
            )
            
            model_selector = gr.Dropdown(
                choices=["ResNet50", "VGG16"],
                value="ResNet50",
                label="Chọn Mô hình Phân loại"
            )
            
            submit_btn = gr.Button("Chạy Phân Tích", variant="primary")

        # Cột bên phải cho output
        with gr.Column(scale=2):
            gr.Markdown("### Kết quả xử lý")
            with gr.Row():
                # output_raw = gr.Image(label="Ảnh Gốc (Raw)")
                # output_processed = gr.Image(label="Ảnh đã xử lý (ROI/Detect/Segment)")
                # output_for_classify = gr.Image(label="Ảnh đưa vào phân loại")
                rois_draw = gr.Image(label="Ảnh sau khi ROI")
                
            with gr.Row():
                with gr.Column(scale=1):
                    transformed_image_1 = gr.Image(label="Ảnh sau khi xử lý")
                with gr.Column(scale=2):
                    result_1 = gr.Label(label="Kết quả Phân loại")
                    
            with gr.Row():
                with gr.Column(scale=1):
                    transformed_image_2 = gr.Image(label="Ảnh sau khi xử lý")
                with gr.Column(scale=2):
                    result_2 = gr.Label(label="Kết quả Phân loại")

    # Kết nối hành động click nút với hàm xử lý
    submit_btn.click(
        fn=process_and_classify,
        inputs=[input_image, mode_selector, model_selector],
        outputs=[rois_draw, transformed_image_1, transformed_image_2, result_1, result_2]
    )

    # Thêm một vài ví dụ để người dùng dễ dàng thử nghiệm
    gr.Examples(
        examples=[
            ["examples/normal.png", "normal"],
            ["examples/osteoporosis.png", "osteoporosis"],
            ["examples/normal_2.png", "normal"],
            ["examples/osteoporosis_2.png", "osteoporosis"],
        ],
        inputs=[input_image, gr.Textbox(label="Nhãn", show_label=False)],
        outputs=[rois_draw, transformed_image_1, transformed_image_2, result_1, result_2],
        fn=process_and_classify,
        cache_examples=False,
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, debug=True, share=True)
    # share=True : thêm param này nếu muốn chạy https người khác có thể truy cập trong 1 tuần
