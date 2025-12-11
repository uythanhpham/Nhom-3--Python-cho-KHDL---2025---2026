---
------------------------------ PHƯỚNG DẪN CÀI ĐẶT THƯ VIỆN ------------------------------

---

# **Chuẩn bị môi trường (chạy 1 lần duy nhất)**
  Mở terminal tại thư mục:
  ```bash
  pima-diabetes-project/
  ````
  ---

### **Tạo môi trường ảo (khuyến nghị)**
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### **Cài dependencies**
```bash
pip install -r requirements.txt
```

---
---
---
---
---
---

---
------------------------------ HƯỚNG DẪN CHẠY DỰ ÁN ------------------------------
---

## **       Giới thiệu Cấu hình Hệ thống (Configuration)           **
Dự án sử dụng Configuration Object Pattern để quản lý tham số, chia làm 2 tầng:
  1. **Cấu hình Mặc định (Default Settings)**:
    * Nằm trong `src/preprocessing/data_preprocessor.py` và `src/modeling/model_trainer.py`.
    * Đây là các giá trị chuẩn (factory defaults) giúp code chạy ổn định ngay cả khi không truyền tham số.
        ? - [Giải thích THAM SỐ MẶC ĐỊNH DataPreprocessor](./reports/THAM%20SỐ%20MẶC%20ĐỊNH%20DataPreprocessor.pdf)
        ? - [Giải thích THAM SỐ MẶC ĐỊNH ModelTrainer](./reports/THAM%20SỐ%20MẶC%20ĐỊNH%20ModelTrainer.pdf)
  2. **Cấu hình Người dùng (User Options)**:
    * Nằm trong `configs/config.yaml` được truyền vào `src/main.py` khi chạy.
        ? - [Giải thích THAM SỐ TRUYỀN VÀO](./reports/THAM%20SỐ%20TRUYỀN%20VÀO.pdf)
    * Đây là nơi ghi đè (override) các cài đặt mặc định để tùy chỉnh thí nghiệm. Việc hiển thị rõ các tham số tại đây giúp người dùng dễ dàng theo dõi và thay đổi chiến lược xử lý mà không cần can thiệp vào code lõi.

---

# **        1. (Tuỳ chọn) Chạy Notebook 01 — EDA        **
```text
notebooks/01_eda_pima.ipynb
```
Notebook này dùng để:
  * Khám phá dữ liệu (EDA)
  * Xem missing, hidden missing (Glucose/BMI/Insulin, …)
  * Phân phối các biến
  * Quan sát imbalance của Outcome
  * Lấy hình minh họa đưa vào báo cáo.
---
**        Notebook 1 không bắt buộc        **.
Nó chỉ giúp hiểu dữ liệu trước khi tiền hành tiền xử lý và modeling.

---
# **        2. Chạy Pipeline Chính — BẮT BUỘC        **
Pipeline nằm trong:
```text
src/main.py
```
Chạy bằng một lệnh duy nhất:

```bash
python -m src.main
```
    Hoặc:

```bash
python src/main.py
```
### Bên trong pipeline sẽ tự động:
1. Xác định `project_root`
2. Tạo `PreprocessorConfig`
3. Tạo `TrainerConfig`
4. Chạy **tiền xử lý full**:
   * Load `data/raw/diabetes.csv`
   * Hidden missing → NaN
   * Feature engineering (BMI_category, Age_group, …)
   * Chia train/test (stratify)
   * Fit tiền xử lý trên Train → Transform Test
   * Lưu dữ liệu processed vào:
     ```text
     data/processed/
     ```
5. Huấn luyện mô hình
6. Tìm best model
7. In đầy đủ metrics
8. In Feature Importance + SHAP
9. Lưu model vào:
   ```text
   models/
   ```

---
# **        5. Chạy Notebook 02 — Evaluation, Visualization        **
Sau khi pipeline chạy xong ít nhất 1 lần, thư mục sau sẽ xuất hiện:
  * `data/processed/*.parquet` -> cleaned data
  * `models/*_best.joblib` -> the best model
  * `results/*.csv` -> best model's metrics
Sau đó mở:
```text
notebooks/02_model_evaluation.ipynb
```
Notebook này dùng để:
  * Load lại X_train/X_test đã xử lý
  * Load best model đã lưu
  * Vẽ thêm biểu đồ đẹp:

      * Confusion Matrix
      * ROC Curve
      * Precision–Recall Curve
      * SHAP Summary / Force Plot
      * Feature Importance cho báo cáo
  * Thử nghiệm ngưỡng phân loại (threshold tuning)
**Notebook 2 chỉ chạy SAU khi pipeline đã chạy.**
Đây là nơi lấy hình & phân tích sâu để làm báo cáo.

---

# **        6. Chú ý quan trọng        **
### Không cần chạy trực tiếp:
* `data_preprocessor.py`
* `model_trainer.py`
Hai file này chức năng duy nhất là **chứa định nghĩa class**, được gọi tự động trong `main.py`,
nên không phải luồng chạy.
### Lệnh duy nhất để chạy toàn bộ project:
```bash
python -m src.main
```
```
```
