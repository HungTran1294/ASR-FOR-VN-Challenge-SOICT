# ASR-FOR-VN-Challenge-SOICT
This project is build for SOICT Automatic Speech Recognition Challenge.
Include:

1. Colab Project 
2. Project Report
3. Project Presentation slide
4. Output: Checkpoint - infer Result

# Thành Viên :
1. Dương Tiến Hoàng - 20240888e
2. Nguyễn Tuấn Anh - 20241171e
3. Nguyễn Duy Khánh Linh - 20240821e
4. Trần Quốc Hùng - 20240486e

# Phân Công Công Việc: 
1. Trần Quốc Hùng
- Nghiên cứu lựa chọn kiến trúc 
- Triển khai Finetune pipeline 
- Phân công công việc
- Chạy Pretrain model
- implement upload dataset lên driver
- Implement luồng training trên colap

2. Nguyễn Duy Khánh Linh
    •	Tổng hợp và viết báo cáo
	•	Tìm hiểu các mô hình tiềm năng
	•	Đánh giá bộ dữ liệu
	•	Đánh giá mô hình và đưa ra kết quả
3. Dương Tiến Hoàng
    •	Chuẩn bị slide
	•	Tìm hiểu các mô hình tiềm năng
	•	Tiền xử lý dữ liệu
4. Nguyễn Tuấn Anh
	•	Viết báo cáo
    •	Tìm hiểu các mô hình tiềm năng
	•	Tiền xử lý dữ liệu
	•	Đánh giá mô hình và đưa ra kết quả

# Fine-tuning Wav2Vec2 cho Bài toán Nhận dạng Giọng nói (ASR) Tiếng Việt

Project trong khuôn khổ BTL môn học Trí tuệ nhân tạo tạo sinh Audio. Nhóm sử dụng Google Colab để fine-tune mô hình Wav2Vec2 cho bài toán Nhận dạng Giọng nói Tự động (ASR) trên dữ liệu Tiếng Việt. Mô hình được fine-tune dựa trên các checkpoint đã được huấn luyện trước và sử dụng tập dữ liệu tùy chỉnh. Dataset và các checkpoint được lưu trên Google Drive.

## Giới thiệu

Wav2Vec2 là mô hình ASR của Facebook AI, giúp học cách biểu diễn giọng nói từ dữ liệu không nhãn. Mô hình này có thể được fine-tune trên một lượng nhỏ dữ liệu có nhãn để đạt được kết quả ấn tượng. 

Mô hình được fine-tune sử dụng Connectionist Temporal Classification (CTC) [1], một thuật toán phổ biến cho các bài toán sequence-to-sequence như ASR.

## Cài đặt và Chạy

Dự án này được thiết kế để chạy trực tiếp trên Google Colab. Các bước cần thiết để chuẩn bị và chạy notebook được mô tả dưới đây:

1.  **Mở Notebook:** Mở file notebook Google Colab này.
2.  **Kết nối Google Drive:** Đảm bảo bạn đã mount Google Drive trong Colab. Notebook có các cell code để thực hiện việc này. Các dataset đã tiền xử lý và checkpoint mô hình sẽ được lưu trữ tại các đường dẫn trên Drive của bạn.
3.  **Cập nhật Đường dẫn:** Trong phần đầu của notebook, bạn cần cập nhật các biến sau để phù hợp với cấu trúc thư mục trên Google Drive của bạn:
    *   `processed_dataset_path`: Đường dẫn đến thư mục trên Google Drive để lưu trữ dataset sau khi tiền xử lý và tải dataset đã lưu.
    *   `data_dir`: Đường dẫn đến thư mục chứa dataset gốc Tiếng Việt của bạn.
    *   `output_dir`: Đường dẫn đến thư mục trên Google Drive để lưu trữ các checkpoint mô hình trong quá trình huấn luyện.
    *   `repo_name`: Tên của repository trên Hugging Face Hub (nếu bạn muốn push tokenizer và mô hình lên đó).
4.  **Chuẩn bị Tokens:** Nếu bạn muốn push tokenizer và mô hình lên Hugging Face Hub hoặc sử dụng Weights & Biases (WandB), bạn cần chuẩn bị và cung cấp các API token tương ứng. Notebook bao gồm các cell để đăng nhập vào Hugging Face Hub.
5.  **Chạy từng Cell:** Chạy từng cell trong notebook theo thứ tự. Notebook sẽ tự động:
    *   Cài đặt các thư viện cần thiết (`datasets`, `transformers`, `jiwer`).
    *   Kiểm tra và tải dataset đã được tiền xử lý từ Drive (nếu tồn tại) hoặc tiến hành tiền xử lý dataset gốc (trích xuất audio và transcript, chuẩn hóa văn bản, tách train/test).
    *   Tạo vocabulary và tokenizer dựa trên dataset.
    *   Tạo feature extractor và processor.
    *   Tải mô hình Wav2Vec2 pretrained (base) và cấu hình cho bài toán ASR.
    *   Định nghĩa Data Collator và hàm tính metrics (WER).
    *   Thiết lập các tham số huấn luyện và khởi tạo `Trainer`.
    *   Bắt đầu quá trình fine-tuning.

## Cấu trúc Dự án

*   `your_notebook_name.ipynb`: File notebook Google Colab chứa toàn bộ mã nguồn cho việc tiền xử lý dữ liệu, thiết lập mô hình và huấn luyện.
*   `/content/drive/MyDrive/.../processed_dataset_20k`: Thư mục trên Google Drive chứa dataset đã được tiền xử lý (`train` và `test` subdirectories).
*   `/content/drive/MyDrive/.../ASR-model-20k-04`: Thư mục trên Google Drive chứa các checkpoint mô hình đã fine-tune.
*   `vocab.json`: File vocabulary được tạo ra từ dataset, sử dụng bởi tokenizer.

## Dataset

Dự án sử dụng một tập dữ liệu ASR Tiếng Việt tùy chỉnh. Quá trình tiền xử lý dataset trong notebook bao gồm:

*   Quét các file `.wav` và `.txt` trong thư mục dữ liệu gốc.
*   Ghép nối các cặp audio và transcript.
*   Load nội dung transcript và dữ liệu audio (sử dụng `librosa`).
*   Chuẩn hóa văn bản (chuyển sang chữ thường, loại bỏ ký tự đặc biệt).
*   Tách dataset thành tập train và test (tỷ lệ 90/10).
*   Lưu dataset đã xử lý lên Google Drive để sử dụng lại.

## Mô hình

Mô hình cơ sở được sử dụng là `facebook/wav2vec2-base` pretrained trên dữ liệu Tiếng Anh. Mô hình này sau đó được fine-tune trên tập dữ liệu Tiếng Việt.

*   **Mô hình:** `Wav2Vec2ForCTC` từ thư viện `transformers`.
*   **Tokenizer:** `Wav2Vec2CTCTokenizer` được tạo tùy chỉnh dựa trên vocabulary từ dataset Tiếng Việt.
*   **Feature Extractor:** `Wav2Vec2FeatureExtractor`.
*   **Processor:** `Wav2Vec2Processor` kết hợp feature extractor và tokenizer.

Quá trình fine-tuning bao gồm đóng băng các tham số của phần feature encoder để chỉ huấn luyện lớp đầu ra (linear layer) cho bài toán CTC.

## Kết quả và Đánh giá

Hiệu suất của mô hình được đánh giá bằng Word Error Rate (WER) trên tập test. Trong quá trình huấn luyện, WER được tính toán và log sau mỗi `eval_steps`.

*(Sau khi huấn luyện hoàn thành, bạn có thể thêm phần này để trình bày kết quả WER cuối cùng và bất kỳ quan sát nào khác.)*


## Tham khảo

[1] Awni Hannun. Sequence Modeling with CTC (2017). https://distill.pub/2017/ctc/

*Lưu ý: File README này được tạo dựa trên nội dung và cấu trúc của notebook Google Colab của bạn. Bạn có thể cần điều chỉnh các đường dẫn, tên file và chi tiết khác cho phù hợp với dự án cụ thể của mình.*