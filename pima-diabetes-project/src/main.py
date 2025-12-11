from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

from src.preprocessing.data_preprocessor import (
    DataPreprocessor,
    PreprocessorConfig,
    DataPreprocessorError,
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
# 1. Hàm tiện ích load YAML + build config
# -----------------------------------------------------------
def load_config(project_root: Path) -> Dict[str, Any]:
    """
    Đọc file configs/config.yaml và trả về dict cấu hình.
    """
    config_path = project_root / "configs" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file config: {config_path}")

    logger.info("Đọc cấu hình từ %s", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    return cfg


def build_preprocessor_config(project_root: Path, cfg: Dict[str, Any]) -> PreprocessorConfig:
    """
    Tạo PreprocessorConfig từ dict cấu hình (section 'preprocessor' trong YAML).
    - Đường dẫn trong YAML là relative path, được nối với project_root.
    """
    p = cfg["preprocessor"]

    raw_data_path = project_root / p["raw_data_path"]
    processed_train_path = project_root / p["processed_train_path"]
    processed_test_path = project_root / p["processed_test_path"]

    # Scaler: chỉnh lại đường dẫn save_scaler_path sang Path
    scaler_cfg = dict(p["scaler"])
    if "save_scaler_path" in scaler_cfg and scaler_cfg["save_scaler_path"] is not None:
        scaler_cfg["save_scaler_path"] = project_root / scaler_cfg["save_scaler_path"]

    prep_config = PreprocessorConfig(
        raw_data_path=raw_data_path,
        processed_train_path=processed_train_path,
        processed_test_path=processed_test_path,
        # target_col có thể để default trong PreprocessorConfig, hoặc lấy từ YAML nếu có
        target_col=p.get("target_col", "Outcome"),
        hidden_missing_cols=p["hidden_missing_cols"],
        missing=p["missing"],
        outlier=p["outlier"],
        scaler=scaler_cfg,
        encoding=p["encoding"],
        feature_engineering=p["feature_engineering"],
        split=p["split"],
    )
    return prep_config


def build_trainer_config(project_root: Path, cfg: Dict[str, Any]) -> TrainerConfig:
    """
    Tạo TrainerConfig từ dict cấu hình (section 'trainer' trong YAML).
    """
    t = cfg["trainer"]

    model_output_dir = project_root / t["model_output_dir"]

    trainer_config = TrainerConfig(
        target_col=t["target_col"],
        random_state=t["random_state"],
        scoring_primary=t["scoring_primary"],
        scoring_other=t["scoring_other"],
        cv_splits=t["cv_splits"],
        use_randomized_search=t["use_randomized_search"],
        n_iter_random_search=t["n_iter_random_search"],
        use_smote=t["use_smote"],
        model_names=t["model_names"],
        model_output_dir=model_output_dir,
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
    In ra summary các chỉ số trên tập test và giải thích nhanh trade-off
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
        logger.info(
            "===== TOP %d FEATURE IMPORTANCE (%s) =====",
            top_k,
            best_model_name,
        )
        logger.info("\n%s", importance_df.head(top_k).to_string(index=False))
    except ModelTrainerError as e:
        logger.warning("Không lấy được feature importance: %s", e)

    # SHAP Values
    try:
        shap_values, shap_imp_df = trainer.compute_shap_values(
            best_model_name,
            X_sample=X_train,
            max_samples=200,
        )
        logger.info(
            "===== TOP %d SHAP MEAN(|VALUE|) (%s) =====",
            top_k,
            best_model_name,
        )
        logger.info("\n%s", shap_imp_df.head(top_k).to_string(index=False))
    except ModelTrainerError as e:
        logger.warning(
            "Không tính được SHAP values (có thể thiếu thư viện shap): %s",
            e,
        )


# -----------------------------------------------------------
# 3. Hàm main – chạy full pipeline bằng 1 lệnh
# -----------------------------------------------------------
def main() -> None:
    """
    Chạy full pipeline:
    1) Tiền xử lý dữ liệu (DataPreprocessor) – fit trên train, transform trên test.
    2) Huấn luyện & tối ưu nhiều model (ModelTrainer) bằng CV.
    3) Đánh giá trên test với F1, ROC-AUC, Accuracy, Precision, Recall.
    4) In ra feature importance + SHAP (nếu có) để giải thích mô hình.
    """
    # Xác định project_root từ vị trí file src/main.py
    project_root: Path = Path(__file__).resolve().parents[1]
    logger.info("Project root: %s", project_root)

    # 1) Đọc YAML config và tạo config cho Preprocessor & Trainer
    try:
        cfg = load_config(project_root)
    except FileNotFoundError as e:
        logger.error("Không đọc được file cấu hình: %s", e)
        return
    except Exception as e:
        logger.exception("Lỗi không mong đợi khi đọc config: %s", e)
        return

    prep_config: PreprocessorConfig = build_preprocessor_config(project_root, cfg)
    trainer_config: TrainerConfig = build_trainer_config(project_root, cfg)

    # 2) Chạy Preprocessing
    try:
        preprocessor = DataPreprocessor(config=prep_config)
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
            "Không tìm thấy metrics trên test cho model '%s'.",
            best_model_name,
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

    logger.info(
        "Hoàn tất toàn bộ pipeline Pima Diabetes – có thể dùng để trình bày với giảng viên."
    )


if __name__ == "__main__":
    main()
# python -m src.main