from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)

from src.utils.config import RESULTS_DIR


class ModelTrainerError(Exception):
    """Custom exception cho các lỗi liên quan đến ModelTrainer."""
    pass


def _get_project_root() -> Path:
    """
    Suy ra project root theo cấu trúc:
    pima-diabetes-project/
      └─ src/
          └─ modeling/
              └─ model_trainer.py
    """
    return Path(__file__).resolve().parents[2]


# Cố gắng dùng logger chung của project nếu có
try:
    from src.utils.logger import get_logger  # type: ignore

    logger = get_logger(__name__)
except Exception:  # pragma: no cover - fallback đơn giản
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """
    Cấu hình cho ModelTrainer.

    Tất cả hyper-parameters, random_state, đường dẫn lưu model, v.v.
    đều đi qua đây để tránh hard-code trong code.
    """
    target_col: str = "Outcome"
    random_state: int = 42

    # Mục tiêu tối ưu chính trong cross-validation
    scoring_primary: str = "f1"

    # Các metrics khác để log và phân tích
    scoring_other: List[str] = field(
        default_factory=lambda: ["roc_auc", "accuracy", "precision", "recall"]
    )

    # Cấu hình cross-validation
    cv_splits: int = 5
    use_randomized_search: bool = True
    n_iter_random_search: int = 20

    # Xử lý imbalance
    use_smote: bool = True

    # Danh sách model cần train & tune
    model_names: List[str] = field(
        default_factory=lambda: ["log_reg", "random_forest"]
    )

    # Thư mục lưu model (nếu None thì không lưu)
    model_output_dir: Optional[Union[str, Path]] = None


class ModelTrainer:
    """
    ModelTrainer:
    - Nhận dữ liệu sau khi đã được DataPreprocessor xử lý (X_train, X_test, y_train, y_test).
    - Huấn luyện nhiều model (ít nhất 2), dùng GridSearchCV / RandomizedSearchCV.
    - Xử lý imbalance bằng class_weight='balanced' và/hoặc SMOTE (trong Pipeline).
    - Đảm bảo reproducibility với random_state cố định.
    - Đánh giá bằng Accuracy, Precision, Recall, F1, ROC-AUC.
    - Cung cấp feature importance / SHAP cho Explainable AI.
    """

    def __init__(self, config: Optional[TrainerConfig] = None) -> None:
        self.config: TrainerConfig = config or TrainerConfig()

        self.project_root: Path = _get_project_root()

        if self.config.model_output_dir is not None:
            out_dir = Path(self.config.model_output_dir)
            if not out_dir.is_absolute():
                out_dir = self.project_root / out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            self.model_output_dir: Optional[Path] = out_dir
        else:
            self.model_output_dir = None

        # Lưu best model cho từng mô hình
        self.best_models_: Dict[str, BaseEstimator] = {}
        # Lưu toàn bộ kết quả cross-validation
        self.cv_results_: Dict[str, pd.DataFrame] = {}
        # Lưu metrics trên tập test
        self.test_metrics_: Dict[str, Dict[str, float]] = {}
        # Tên model tốt nhất theo scoring_primary
        self.best_model_name_: Optional[str] = None

        logger.info("Khởi tạo ModelTrainer với config: %s", self)

    def __repr__(self) -> str:
        return (
            f"ModelTrainer("
            f"models={self.config.model_names}, "
            f"scoring_primary={self.config.scoring_primary}, "
            f"cv_splits={self.config.cv_splits}, "
            f"use_smote={self.config.use_smote}"
            f")"
        )

    # ------------------------------------------------------------------
    # API chính
    # ------------------------------------------------------------------
    def run_training(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> None:
        """
        Chạy toàn bộ pipeline huấn luyện & đánh giá:

        1) Cho từng model trong config.model_names:
           - Xây dựng estimator (có/không dùng SMOTE Pipeline).
           - Chạy GridSearchCV / RandomizedSearchCV với CV stratified.
           - Lưu lại best_estimator_ và bảng cv_results_.

        2) Nếu X_test, y_test không None:
           - Đánh giá best model trên tập test.
           - Lưu lại metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
        """
        logger.info(
            "Bắt đầu run_training. X_train shape: %s",
            getattr(X_train, "shape", None),
        )

        for model_name in self.config.model_names:
            logger.info("Huấn luyện model: %s", model_name)
            estimator, param_grid = self._get_model_and_param_grid(model_name)

            # Scoring cho CV
            scoring_dict: Dict[str, str] = {
                self.config.scoring_primary: self.config.scoring_primary
            }
            for s in self.config.scoring_other:
                if s not in scoring_dict:
                    scoring_dict[s] = s

            cv = StratifiedKFold(
                n_splits=self.config.cv_splits,
                shuffle=True,
                random_state=self.config.random_state,
            )

            if self.config.use_randomized_search:
                search = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=param_grid,
                    n_iter=self.config.n_iter_random_search,
                    scoring=scoring_dict,
                    refit=self.config.scoring_primary,
                    cv=cv,
                    n_jobs=1,  # tránh lỗi joblib/loky trên một số môi trường
                    verbose=1,
                    random_state=self.config.random_state,
                )
            else:
                search = GridSearchCV(
                    estimator=estimator,
                    param_grid=param_grid,
                    scoring=scoring_dict,
                    refit=self.config.scoring_primary,
                    cv=cv,
                    n_jobs=1,
                    verbose=1,
                )

            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_

            # Lưu best estimator và cv_results_
            self.best_models_[model_name] = best_estimator
            self.cv_results_[model_name] = pd.DataFrame(search.cv_results_)

            logger.info(
                "Hoàn tất CV cho %s. Best params: %s, best %s: %.4f",
                model_name,
                search.best_params_,
                self.config.scoring_primary,
                search.best_score_,
            )

            # Lưu model nếu cấu hình
            self._maybe_save_model(model_name, best_estimator)

            # Đánh giá trên tập test nếu có
            if X_test is not None and y_test is not None:
                metrics = self.evaluate_on_test(best_estimator, X_test, y_test)
                self.test_metrics_[model_name] = metrics
                logger.info("Test metrics cho %s: %s", model_name, metrics)

        # Chọn model tốt nhất theo scoring_primary trên test (nếu có)
        if self.test_metrics_:
            primary = self.config.scoring_primary
            best_name: Optional[str] = None
            best_score = -np.inf
            for name, m in self.test_metrics_.items():
                score = m.get(primary, np.nan)
                if not np.isnan(score) and score > best_score:
                    best_score = score
                    best_name = name
            self.best_model_name_ = best_name
            logger.info(
                "Model tốt nhất trên test theo %s: %s (score=%.4f)",
                primary,
                self.best_model_name_,
                best_score,
            )

            # Sau khi xác định best_model_name_ → lưu ra file
            self._save_experiments_and_best_model()

    # ------------------------------------------------------------------
    # Ghi experiments.csv và best_model.txt
    # ------------------------------------------------------------------
    def _save_experiments_and_best_model(self) -> None:
        """
        Ghi kết quả thí nghiệm ra:
        - results/experiments.csv: metrics test của từng model.
        - results/best_model.txt: model tốt nhất + metrics chi tiết.
        """
        if not self.test_metrics_:
            logger.warning(
                "Không có test_metrics_ → không ghi được experiments.csv / best_model.txt."
            )
            return

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # 1) experiments.csv
        rows: List[Dict[str, Any]] = []
        for model_name, m in self.test_metrics_.items():
            row: Dict[str, Any] = {"model_name": model_name}
            for k, v in m.items():
                if isinstance(v, (np.floating, np.integer)):
                    row[k] = float(v)
                else:
                    row[k] = v
            rows.append(row)

        experiments_df = pd.DataFrame(rows)
        exp_path = RESULTS_DIR / "experiments.csv"
        experiments_df.to_csv(exp_path, index=False)
        logger.info("Đã ghi toàn bộ test metrics vào: %s", exp_path)

        # 2) best_model.txt
        if self.best_model_name_ is None:
            logger.warning(
                "best_model_name_ = None → không ghi được best_model.txt."
            )
            return

        best_metrics = self.test_metrics_.get(self.best_model_name_, {})
        best_path = RESULTS_DIR / "best_model.txt"
        with open(best_path, "w", encoding="utf-8") as f:
            f.write(f"Best model: {self.best_model_name_}\n")
            for k, v in best_metrics.items():
                if isinstance(v, (np.floating, np.integer)):
                    v = float(v)
                f.write(f"{k}: {v}\n")

        logger.info("Đã ghi model tốt nhất và metrics vào: %s", best_path)

    # ------------------------------------------------------------------
    # Xây dựng model & param grid
    # ------------------------------------------------------------------
    def _build_pipeline_with_optional_smote(
        self, base_estimator: BaseEstimator
    ) -> BaseEstimator:
        """
        Nếu config.use_smote=True thì quấn estimator vào Pipeline(SMOTE + model).
        Nếu False thì trả về estimator gốc.
        """
        if not self.config.use_smote:
            return base_estimator

        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline
        except ImportError as e:  # pragma: no cover - phụ thuộc imblearn
            logger.warning(
                "Không import được imblearn. Sẽ không dùng SMOTE. Lỗi: %s", e
            )
            return base_estimator

        smote = SMOTE(random_state=self.config.random_state)
        pipeline = Pipeline(steps=[("smote", smote), ("model", base_estimator)])
        return pipeline

    def _get_model_and_param_grid(
        self, model_name: str
    ) -> Tuple[BaseEstimator, Dict[str, List[Any]]]:
        """
        Trả về (estimator, param_grid) cho từng model_name.
        Các tham số đều gắn random_state & class_weight='balanced' (nếu hỗ trợ)
        để đảm bảo reproducibility & xử lý imbalance.
        """
        rs = self.config.random_state

        if model_name == "log_reg":
            base_estimator = LogisticRegression(
                solver="liblinear",
                max_iter=1000,
                class_weight="balanced",
                random_state=rs,
            )
            core_param_grid: Dict[str, List[Any]] = {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l1", "l2"],
            }

        elif model_name == "random_forest":
            base_estimator = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                n_jobs=1,
                random_state=rs,
            )
            core_param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5],
            }

        else:
            raise ModelTrainerError(f"Model không được hỗ trợ: {model_name}")

        # Nếu cấu hình dùng SMOTE, thử quấn vào Pipeline(SMOTE + model)
        use_smote_cfg = self.config.use_smote

        if use_smote_cfg:
            estimator_with_smote = self._build_pipeline_with_optional_smote(
                base_estimator
            )

            # Nếu imblearn có, estimator_with_smote sẽ là Pipeline có named_steps["model"]
            if hasattr(estimator_with_smote, "named_steps") and "model" in getattr(
                estimator_with_smote, "named_steps"
            ):
                estimator = estimator_with_smote
                # Khi đó tham số của model nằm dưới prefix "model__"
                param_grid = {
                    f"model__{k}": v for k, v in core_param_grid.items()
                }
            else:
                # Không tạo được Pipeline(SMOTE+model) → dùng base_estimator
                logger.warning(
                    "use_smote=True nhưng không tạo được Pipeline(SMOTE+model). "
                    "Sẽ dùng base_estimator mà không có SMOTE."
                )
                estimator = base_estimator
                param_grid = core_param_grid
        else:
            estimator = base_estimator
            param_grid = core_param_grid

        return estimator, param_grid

    # ------------------------------------------------------------------
    # Đánh giá & Explainability
    # ------------------------------------------------------------------
    def evaluate_on_test(
        self,
        model: BaseEstimator,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
    ) -> Dict[str, float]:
        """
        Tính các metrics trên tập test: Accuracy, Precision, Recall, F1, ROC-AUC.
        Đồng thời in classification_report để dễ xem khi debug.
        """
        y_pred = model.predict(X_test)

        metrics: Dict[str, float] = {}
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        metrics["precision"] = precision_score(y_test, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_test, y_pred, zero_division=0)
        metrics["f1"] = f1_score(y_test, y_pred, zero_division=0)

        # ROC-AUC cần xác suất hoặc decision_function
        y_score: Optional[np.ndarray] = None
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)

        if y_score is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_test, y_score)
            except ValueError:
                metrics["roc_auc"] = float("nan")
        else:
            metrics["roc_auc"] = float("nan")

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics["tn"] = float(tn)
        metrics["fp"] = float(fp)
        metrics["fn"] = float(fn)
        metrics["tp"] = float(tp)

        logger.info(
            "Classification report trên test:\n%s",
            classification_report(y_test, y_pred, zero_division=0),
        )

        return metrics

    def get_feature_importance(
        self,
        model_name: str,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """
        Trả về bảng feature importance (hoặc hệ số) cho model đã train.

        - Với RandomForest: dùng feature_importances_.
        - Với LogisticRegression: dùng |coef_| để đo độ quan trọng tuyệt đối.
        - Nếu model là Pipeline(SMOTE+model), sẽ lấy bước 'model' bên trong.
        """
        if model_name not in self.best_models_:
            raise ModelTrainerError(
                f"Chưa tìm thấy model '{model_name}' trong best_models_. "
                f"Hãy chạy run_training trước."
            )

        model = self.best_models_[model_name]

        underlying_model = model
        if hasattr(model, "named_steps") and "model" in getattr(
            model, "named_steps"
        ):
            underlying_model = model.named_steps["model"]

        importances: Optional[np.ndarray] = None

        if hasattr(underlying_model, "feature_importances_"):
            importances = np.asarray(
                underlying_model.feature_importances_, dtype=float
            )
        elif hasattr(underlying_model, "coef_"):
            coef = getattr(underlying_model, "coef_")
            importances = np.abs(coef).reshape(-1)
        else:
            raise ModelTrainerError(
                f"Model '{model_name}' không hỗ trợ feature_importances_ hoặc coef_."
            )

        if len(importances) != len(feature_names):
            raise ModelTrainerError(
                "Số lượng feature_importances không khớp với số feature_names. "
                f"len(importances)={len(importances)}, len(feature_names)={len(feature_names)}"
            )

        df_imp = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)

        return df_imp

    def compute_shap_values(
        self,
        model_name: str,
        X_sample: Union[pd.DataFrame, np.ndarray],
        max_samples: int = 200,
    ) -> Tuple[Any, pd.DataFrame]:
        """
        Tính SHAP values cho model để giải thích dự đoán.

        - Dùng shap.Explainer (TreeExplainer cho mô hình cây).
        - Giới hạn max_samples để tránh quá nặng.
        - Trả về (shap_values, shap_importance_df) với:
            + shap_values: object SHAP gốc.
            + shap_importance_df: bảng mean(|SHAP|) theo từng feature, sort giảm dần.
        """
        try:
            import shap  # type: ignore
        except ImportError as e:
            raise ModelTrainerError(
                "Thư viện 'shap' chưa được cài đặt. "
                "Cài bằng: pip install shap"
            ) from e

        if model_name not in self.best_models_:
            raise ModelTrainerError(
                f"Chưa có model '{model_name}' trong best_models_. "
                "Hãy chạy run_training trước."
            )

        model = self.best_models_[model_name]

        underlying_model = model
        if hasattr(model, "named_steps") and "model" in getattr(
            model, "named_steps"
        ):
            underlying_model = model.named_steps["model"]

        # 1) Chuẩn bị dữ liệu X_used + feature_names
        if isinstance(X_sample, pd.DataFrame):
            X_used_df = X_sample.copy()
            if len(X_used_df) > max_samples:
                X_used_df = X_used_df.sample(
                    n=max_samples,
                    random_state=self.config.random_state,
                )
            feature_names = X_used_df.columns.tolist()
            X_numeric = X_used_df.to_numpy(dtype=float, copy=True)
        else:
            X_arr = np.asarray(X_sample)
            if X_arr.shape[0] > max_samples:
                rng = np.random.default_rng(self.config.random_state)
                idx = rng.choice(X_arr.shape[0], size=max_samples, replace=False)
                X_arr = X_arr[idx]
            X_numeric = np.asarray(X_arr, dtype=float)
            feature_names = [f"f{i}" for i in range(X_numeric.shape[1])]

        explainer = shap.Explainer(underlying_model, X_numeric)
        shap_values = explainer(
            X_numeric,
            check_additivity=False,
        )

        sv = np.asarray(shap_values.values)

        # sv có thể là (n_samples, n_classes, n_features)
        if sv.ndim == 3:
            if sv.shape[1] == 2:
                sv = sv[:, 1, :]  # lớp positive
            else:
                sv = sv.mean(axis=1)

        mean_abs_shap = np.abs(sv).mean(axis=0)

        if mean_abs_shap.ndim > 1:
            mean_abs_shap = mean_abs_shap.ravel()

        n_feat = min(len(feature_names), len(mean_abs_shap))
        feature_names = feature_names[:n_feat]
        mean_abs_shap = mean_abs_shap[:n_feat]

        shap_imp_df = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs_shap,
            }
        ).sort_values("mean_abs_shap", ascending=False)

        return shap_values, shap_imp_df

    # ------------------------------------------------------------------
    # Lưu / tải model
    # ------------------------------------------------------------------
    def _maybe_save_model(self, model_name: str, model: BaseEstimator) -> None:
        """
        Lưu model bằng joblib nếu self.model_output_dir không None.
        """
        if self.model_output_dir is None:
            return

        path = self.model_output_dir / f"{model_name}_best.joblib"
        try:
            joblib.dump(model, path)
            logger.info("Đã lưu model '%s' vào: %s", model_name, path)
        except Exception as e:  # pragma: no cover - I/O
            logger.error(
                "Lỗi khi lưu model '%s' vào %s: %s", model_name, path, e
            )

    def load_saved_model(
        self,
        model_name: str,
        path: Optional[Union[str, Path]] = None,
    ) -> BaseEstimator:
        """
        Tải model đã lưu từ ổ đĩa.

        Parameters
        ----------
        model_name : str
            Tên model (log_reg, random_forest, ...).
        path : str | Path | None
            Đường dẫn file .joblib. Nếu None → lấy từ model_output_dir.

        Returns
        -------
        BaseEstimator
        """
        if path is None:
            if self.model_output_dir is None:
                raise ModelTrainerError(
                    "Không có model_output_dir và cũng không truyền path cụ thể."
                )
            path = self.model_output_dir / f"{model_name}_best.joblib"

        path = Path(path)
        if not path.exists():
            raise ModelTrainerError(f"Không tìm thấy file model: {path}")

        try:
            model = joblib.load(path)
            logger.info("Đã load model '%s' từ: %s", model_name, path)
        except Exception as e:  # pragma: no cover - I/O
            raise ModelTrainerError(f"Lỗi khi load model từ {path}: {e}") from e

        self.best_models_[model_name] = model
        return model

# python -m src.modeling.model_trainer