from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional  # Any bỏ đi vì không dùng

import numpy as np
import pandas as pd

from src.preprocessing.data_preprocessor import (
    DataPreprocessor,
    PreprocessorConfig,
    DataPreprocessorError,
    LoggingPreprocessor,   # <<< MỚI: import lớp kế thừa
)

from src.modeling.model_trainer import (
    ModelTrainer,
    TrainerConfig,
    ModelTrainerError,
)

# -----------------------------------------------------------
# 0. Cấu hình logger chung cho toàn project (nếu chưa cấu hình)
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# 1. Hàm tiện ích build config (tránh hard-code rải rác)
# -----------------------------------------------------------
def build_preprocessor_config(project_root: Path) -> PreprocessorConfig:
    """
    Tạo PreprocessorConfig với đầy đủ cấu hình:
    - Đường dẫn file raw.
    - Chiến lược xử lý hidden missing, imputation, outlier, scaler, feature engineering.
    """
    raw_data_path = project_root / "data" / "raw" / "diabetes.csv"
    processed_dir = project_root / "data" / "processed"

    prep_config = PreprocessorConfig(
        raw_data_path=raw_data_path,
        processed_train_path=processed_dir / "pima_train_processed.parquet",
        processed_test_path=processed_dir / "pima_test_processed.parquet",
        target_col="Outcome",
        # Hidden missing + imputation thông minh
        hidden_missing_cols=["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        missing={
            "numeric_strategy": "median_by_outcome",    # median theo nhóm Outcome (0/1)
            "categorical_strategy": "most_frequent",
        },
        # Outlier IQR + winsorize
        outlier={
            "numeric_cols": None,   # None → tự phát hiện từ numeric_cols
            "method": "iqr",
            "strategy": "winsorize",
            "iqr_factor": 1.5,
        },
        # Scaler: standard, exclude target
        scaler={
            "type": "standard",
            "exclude_cols": ["Outcome"],
            "save_scaler_path": processed_dir / "scaler.joblib",
        },
        # Encoding: (dataset Pima gốc chủ yếu numeric, nhưng vẫn bật để tái sử dụng)
        encoding={
            "strategy": "onehot",
            "handle_unknown": "ignore",
        },
        # Feature engineering mang ý nghĩa y khoa
        feature_engineering={
            "enable": True,
            "create_bmi_category": True,
            "create_age_group": True,
            "create_pregnancy_flag": True,
            "create_interactions": True,
            "bmi_col": "BMI",
            "age_col": "Age",
            "pregnancies_col": "Pregnancies",
            "glucose_col": "Glucose",
            "insulin_col": "Insulin",
        },
        # Split: có stratify=y, random_state cố định để reproducible
        split={
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True,
        },
    )
    return prep_config

def build_trainer_config(project_root: Path) -> TrainerConfig:
    """
    Tạo TrainerConfig:
    - Random state cố định.
    - Dùng F1 làm scoring chính, thêm ROC-AUC và các metrics khác.
    - Huấn luyện Logistic Regression + Random Forest.
    - Có SMOTE để xử lý imbalance.
    """
    model_dir = project_root / "models"

    trainer_config = TrainerConfig(
        target_col="Outcome",
        random_state=42,               # reproducibility
        scoring_primary="f1",          # ưu tiên F1 cho bài toán y khoa
        scoring_other=["roc_auc", "accuracy", "precision", "recall"],
        cv_splits=5,
        use_randomized_search=True,
        n_iter_random_search=20,
        use_smote=True,                # xử lý imbalance trong CV (không leak test)
        model_names=["log_reg", "random_forest"],
        model_output_dir=model_dir,
    )
    return trainer_config

# -----------------------------------------------------------
# 2. Hàm in báo cáo metrics + giải thích trade-off Precision/Recall
# -----------------------------------------------------------
def summarize_test_metrics(
    best_model_name: str,
    metrics: Dict[str, float],
) -> None:
    """
    In summary các chỉ số trên tập test và giải thích nhanh trade-off
    Precision vs Recall trong bối cảnh tiểu đường.
    """
    logger.info("===== KẾT QUẢ TRÊN TẬP TEST (%s) =====", best_model_name)
    acc = metrics.get("accuracy", np.nan)
    prec = metrics.get("precision", np.nan)
    rec = metrics.get("recall", np.nan)
    f1 = metrics.get("f1", np.nan)
    roc_auc = metrics.get("roc_auc", np.nan)

    logger.info("Accuracy : %.4f", acc)
    logger.info("Precision: %.4f", prec)
    logger.info("Recall   : %.4f", rec)
    logger.info("F1-score : %.4f", f1)
    logger.info("ROC-AUC  : %.4f", roc_auc)

    # Confusion matrix components (tn, fp, fn, tp)
    tn = int(metrics.get("tn", np.nan))
    fp = int(metrics.get("fp", np.nan))
    fn = int(metrics.get("fn", np.nan))
    tp = int(metrics.get("tp", np.nan))

    logger.info("Confusion Matrix (tn, fp, fn, tp) = (%d, %d, %d, %d)", tn, fp, fn, tp)

    # Giải thích trade-off Precision / Recall theo tư duy y khoa
    explanation_lines = [
        "Giải thích nhanh về Precision / Recall trong bối cảnh tiểu đường:",
        "- Precision cao: Khi model dự đoán 'có tiểu đường', tỉ lệ dự đoán đúng cao → giảm số bệnh nhân bị báo nhầm là có bệnh.",
        "- Recall cao   : Trong số người thực sự bị tiểu đường, model bắt được nhiều người nhất → giảm số ca bị bỏ sót.",
        "Trong thực tế y khoa, thường ưu tiên Recall cao (thà gọi bệnh nhân đi kiểm tra thêm còn hơn bỏ sót người bệnh).",
    ]
    for line in explanation_lines:
        logger.info(line)

    if not np.isnan(prec) and not np.isnan(rec):
        if rec >= prec:
            logger.info(
                "Kết quả hiện tại: Recall (%.3f) >= Precision (%.3f) → mô hình có xu hướng ưu tiên không bỏ sót bệnh nhân.",
                rec,
                prec,
            )
        else:
            logger.info(
                "Kết quả hiện tại: Precision (%.3f) > Recall (%.3f) → mô hình cẩn trọng khi gắn nhãn 'có bệnh', nhưng có thể bỏ sót một số ca.",
                prec,
                rec,
            )

def show_feature_importance_and_shap(
    trainer: ModelTrainer,
    best_model_name: str,
    X_train: pd.DataFrame,
    top_k: int = 10,
) -> None:
    """
    In top-k feature quan trọng (importance hoặc SHAP) để chứng minh
    mô hình không phải hộp đen, đồng thời giúp giải thích cho giảng viên.
    """
    # Feature Importance (nếu hỗ trợ)
    try:
        importance_df = trainer.get_feature_importance(
            best_model_name,
            feature_names=X_train.columns.tolist(),
        )
        logger.info("===== TOP %d FEATURE IMPORTANCE (%s) =====", top_k, best_model_name)
        logger.info("\n%s", importance_df.head(top_k).to_string(index=False))
    except ModelTrainerError as e:
        logger.warning("Không lấy được feature importance: %s", e)

    # SHAP Values
    try:
        shap_values, shap_imp_df = trainer.compute_shap_values(
            best_model_name, X_sample=X_train, max_samples=200
        )
        logger.info("===== TOP %d SHAP MEAN(|VALUE|) (%s) =====", top_k, best_model_name)
        logger.info("\n%s", shap_imp_df.head(top_k).to_string(index=False))
    except ModelTrainerError as e:
        logger.warning("Không tính được SHAP values (có thể thiếu thư viện shap): %s", e)

# -----------------------------------------------------------
# 3. Hàm main – chạy full pipeline bằng 1 lệnh
# -----------------------------------------------------------
def main() -> None:
    """
    Chạy full pipeline:
    1) Tiền xử lý dữ liệu (DataPreprocessor/LoggingPreprocessor).
    2) Huấn luyện & tối ưu nhiều model (ModelTrainer) bằng CV.
    3) Đánh giá trên test với F1, ROC-AUC, Accuracy, Precision, Recall.
    4) In ra feature importance + SHAP (nếu có) để giải thích mô hình.
    """
    # Xác định project_root từ vị trí file main.py (src/main.py)
    project_root: Path = Path(__file__).resolve().parents[1]
    logger.info("Project root: %s", project_root)

    # 1) Tạo config cho Preprocessor & Trainer
    prep_config: PreprocessorConfig = build_preprocessor_config(project_root)
    trainer_config: TrainerConfig = build_trainer_config(project_root)

    # 2) Chạy Preprocessing
    try:
        # preprocessor = DataPreprocessor(config=prep_config)    # CŨ
        preprocessor = LoggingPreprocessor(config=prep_config)   # DÙNG LỚP KẾ THỪA
        X_train, X_test, y_train, y_test = preprocessor.run_full_preprocessing()
    except (DataPreprocessorError, FileNotFoundError) as e:
        logger.error("Lỗi trong bước tiền xử lý dữ liệu: %s", e)
        return
    except Exception as e:
        logger.exception("Lỗi không mong đợi trong bước tiền xử lý: %s", e)
        return

    logger.info(
        "Tiền xử lý hoàn tất. X_train: %s, X_test: %s",
        X_train.shape,
        X_test.shape,
    )

    # Log rõ class đang dùng để cho giảng viên thấy kế thừa
    logger.info(
        "Preprocessor class đang dùng: %s",
        preprocessor.__class__.__name__,
    )
    if hasattr(preprocessor, "missing_summary_"):
        logger.info(
            "Missing summary (từ LoggingPreprocessor): %s",
            preprocessor.missing_summary_,
        )

    # 3) Huấn luyện & Tối ưu model
    try:
        trainer = ModelTrainer(config=trainer_config)
        trainer.run_training(X_train, y_train, X_test, y_test)
    except ModelTrainerError as e:
        logger.error("Lỗi trong bước huấn luyện mô hình: %s", e)
        return
    except Exception as e:
        logger.exception("Lỗi không mong đợi trong bước huấn luyện: %s", e)
        return

    # 4) Báo cáo kết quả & Explainable AI
    best_model_name: Optional[str] = trainer.best_model_name_
    if best_model_name is None:
        logger.error("Không tìm được model tốt nhất. Kiểm tra lại cấu hình Trainer.")
        return

    best_metrics: Dict[str, float] = trainer.test_metrics_.get(best_model_name, {})
    if not best_metrics:
        logger.error(
            "Không tìm thấy metrics trên test cho model '%s'.", best_model_name
        )
        return

    summarize_test_metrics(best_model_name, best_metrics)

    # Tính feature importance & SHAP (nếu X_train là DataFrame)
    if isinstance(X_train, pd.DataFrame):
        show_feature_importance_and_shap(trainer, best_model_name, X_train)
    else:
        logger.warning(
            "X_train không phải DataFrame → không có tên cột để hiển thị feature importance / SHAP."
        )

    logger.info("Hoàn tất toàn bộ pipeline Pima Diabetes – có thể dùng để trình bày với giảng viên.")

if __name__ == "__main__":
    main()
    # Chạy bằng:
    #   python -m src.main


