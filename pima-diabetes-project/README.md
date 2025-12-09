---
# **PIMA DIABETES PROJECT — HƯỚNG DẪN CHẠY**
Dự án này xây dựng một **pipeline Machine Learning hoàn chỉnh** để dự đoán bệnh tiểu đường từ dataset Pima Indians Diabetes.
---

---
# **1. Chuẩn bị môi trường (chạy 1 lần duy nhất)**
Mở terminal tại thư mục:
```
pima-diabetes-project/
```
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
# **2. (Tuỳ chọn) Chạy Notebook 01 — EDA**
File:
```
notebooks/01_eda_pima.ipynb
```
Notebook này dùng để:
* Khám phá dữ liệu (EDA)
* Xem missing, hidden missing (Glucose/BMI/Insulin,…)
* Phân phối các biến
* Quan sát imbalance của Outcome
* Lấy hình minh họa đưa vào báo cáo.

**Notebook 1 không bắt buộc**.
Nó chỉ giúp hiểu dữ liệu trước khi modeling.
---

---
# **3. Chạy Pipeline Chính — BẮT BUỘC**
Pipeline nằm trong:
```
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
   * Feature engineering (BMI_category, Age_group,…)
   * Chia train/test (stratify)
   * Fit tiền xử lý trên Train → Transform Test
   * Lưu dữ liệu processed vào:
     ```
     data/processed/
     ```
5. Huấn luyện mô hình
6. Tìm best model
7. In đầy đủ metrics
8. In Feature Importance + SHAP
9. Lưu model vào:
   ```
   models/
   ```
---

---
# **5. Chạy Notebook 02 — Evaluation, Visualization**
Sau khi pipeline chạy xong ít nhất 1 lần, thư mục sau sẽ xuất hiện:
* `data/processed/*.parquet`
* `models/*_best.joblib`
* `results/*.csv`
Giờ mở:
```
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
  * Feature Importance đẹp cho slide
* Thử nghiệm ngưỡng phân loại (threshold tuning)

**Notebook 2 chỉ chạy SAU khi pipeline đã chạy.**
Đây là nơi lấy hình & phân tích sâu để làm báo cáo.
---

---
# **6. Ghi chú quan trọng**
### Không chạy trực tiếp:
* `data_preprocessor.py`
* `model_trainer.py`
Hai file này **chỉ chứa class**, được gọi tự động trong `main.py`.
### Chỉ cần một lệnh duy nhất để chạy toàn bộ project:
```bash
python -m src.main
```
---