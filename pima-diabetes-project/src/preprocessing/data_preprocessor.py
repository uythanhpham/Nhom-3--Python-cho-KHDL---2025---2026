from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder


class DataPreprocessorError(Exception):
    """Custom exception cho các lỗi liên quan đến DataPreprocessor."""
    pass


def _get_project_root() -> Path:
    """
    Suy ra project root theo cấu trúc:
    pima-diabetes-project/
      └─ src/
          └─ preprocessing/
              └─ data_preprocessor.py
    """
    return Path(__file__).resolve().parents[2]


# Thử dùng logger chung của project; nếu không có thì fallback.
try:
    from src.utils.logger import get_logger  # type: ignore

    logger = get_logger(__name__)
except Exception:  # pragma: no cover - fallback
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)


@dataclass
class PreprocessorConfig:
    """
    Cấu hình cho DataPreprocessor.

    Tất cả tên cột, chiến lược, đường dẫn... đều đi qua đây để tránh hard-code.
    """
    # Đường dẫn file gốc & thư mục processed (có thể override khi gọi load/save)
    raw_data_path: Optional[Union[str, Path]] = None
    processed_train_path: Optional[Union[str, Path]] = None
    processed_test_path: Optional[Union[str, Path]] = None

    # Cột target
    target_col: str = "Outcome"

    # Các cột có giá trị 0 là "missing ẩn"
    hidden_missing_cols: List[str] = field(
        default_factory=lambda: [
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
        ]
    )

    # Chiến lược xử lý missing
    # numeric_strategy: median_overall | median_by_outcome
    # categorical_strategy: most_frequent
    missing: Dict[str, Any] = field(
        default_factory=lambda: {
            "numeric_strategy": "median_by_outcome",
            "categorical_strategy": "most_frequent",
        }
    )

    # Cấu hình outlier
    # strategy: winsorize | flag | none
    outlier: Dict[str, Any] = field(
        default_factory=lambda: {
            "numeric_cols": None,  # None → tự động lấy tất cả numeric trừ target
            "method": "iqr",
            "strategy": "winsorize",
            "iqr_factor": 1.5,
        }
    )

    # Cấu hình scaler
    # type: standard | minmax | none
    scaler: Dict[str, Any] = field(
        default_factory=lambda: {
            "type": "standard",
            "exclude_cols": [],  # thêm target, id, v.v. khi cần
            "save_scaler_path": None,  # nếu None → không lưu
        }
    )

    # Cấu hình encoding
    # strategy: onehot | label | none
    encoding: Dict[str, Any] = field(
        default_factory=lambda: {
            "strategy": "onehot",
            "handle_unknown": "ignore",  # dùng ý tưởng, tự kiểm soát bằng align cột
        }
    )

    # Feature engineering (domain knowledge)
    feature_engineering: Dict[str, Any] = field(
        default_factory=lambda: {
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
        }
    )

    # Tham số chia train/test
    split: Dict[str, Any] = field(
        default_factory=lambda: {
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True,
        }
    )


class DataPreprocessor:
    """
    DataPreprocessor:
    - Xử lý toàn bộ bước tiền xử lý dữ liệu Pima Diabetes (và có thể tái sử dụng cho dataset khác).
    - Thiết kế chống data leakage: CHỈ fit trên Train, transform trên Test.
    - Gồm: load, hidden missing, missing, outlier, feature engineering, encode, scale, lưu.
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None) -> None:
        self.config: PreprocessorConfig = config or PreprocessorConfig()

        self.project_root: Path = _get_project_root()
        self.data_raw_dir: Path = self.project_root / "data" / "raw"
        self.data_processed_dir: Path = self.project_root / "data" / "processed"

        # Loại cột
        self._numeric_cols: List[str] = []
        self._categorical_cols: List[str] = []
        self._datetime_cols: List[str] = []

        # Thông tin imputation
        self._numeric_impute_overall_: Dict[str, float] = {}
        # mapping col -> Series index=Outcome, value=median
        self._numeric_impute_by_outcome_: Dict[str, pd.Series] = {}
        self._categorical_impute_: Dict[str, Any] = {}

        # Outlier bounds
        self._outlier_bounds_: Dict[str, Tuple[float, float]] = {}

        # Encoding
        self._encoding_strategy: str = self.config.encoding.get("strategy", "onehot")
        self._label_encoders_: Dict[str, LabelEncoder] = {}
        self._onehot_columns_: List[str] = []
        self._categorical_cols_fitted_: List[str] = []

        # Scaler
        self.scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self._scaled_feature_names_: List[str] = []

        # Cờ đã fit
        self._fitted: bool = False

        logger.info("Khởi tạo DataPreprocessor: %s", self)

    def __repr__(self) -> str:
        return (
            f"DataPreprocessor("
            f"target_col={self.config.target_col}, "
            f"missing_numeric_strategy={self.config.missing.get('numeric_strategy')}, "
            f"outlier_strategy={self.config.outlier.get('strategy')}, "
            f"scaler={self.config.scaler.get('type')}, "
            f"encoding={self.config.encoding.get('strategy')}"
            f")"
        )

    # ------------------------------------------------------------------
    # Public API chính
    # ------------------------------------------------------------------
    def load_data(self, path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Đọc dữ liệu từ nhiều định dạng (csv, excel, json).
        Có try/except + custom exception, tránh hard-code đường dẫn.

        Parameters
        ----------
        path : str | Path | None
            Đường dẫn file dữ liệu. Nếu None → dùng config.raw_data_path
            hoặc data/raw/diabetes.csv.

        Returns
        -------
        pd.DataFrame
        """
        if path is None:
            if self.config.raw_data_path is not None:
                path = self.config.raw_data_path
            else:
                path = self.data_raw_dir / "diabetes.csv"

        path = Path(path)

        if not path.exists():
            raise DataPreprocessorError(f"Không tìm thấy file dữ liệu: {path}")

        logger.info("Đang load dữ liệu từ: %s", path)

        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(path)
            elif path.suffix.lower() == ".json":
                df = pd.read_json(path)
            else:
                raise DataPreprocessorError(
                    f"Định dạng file không được hỗ trợ: {path.suffix}"
                )
        except Exception as e:
            raise DataPreprocessorError(f"Lỗi khi đọc file dữ liệu: {e}") from e

        logger.info("Dữ liệu load thành công. Shape: %s", df.shape)
        self._infer_column_types(df)

        return df

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit toàn bộ pipeline trên tập Train:
        - KHÔNG dùng dữ liệu Test (chống data leakage).
        - Học các thống kê: median, IQR, encoder, scaler.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame chứa cả features + target.

        Returns
        -------
        DataPreprocessor
        """
        if self.config.target_col not in df.columns:
            raise DataPreprocessorError(
                f"Tập dữ liệu truyền vào fit không chứa cột target: {self.config.target_col}"
            )

        df = df.copy()
        logger.info("Bắt đầu fit DataPreprocessor. Shape: %s", df.shape)

        # 1) Phân loại cột
        self._infer_column_types(df)

        # 2) Hidden missing (không cần học thống kê)
        df = self.detect_hidden_missing(df)

        # 3) Feature engineering (chỉ dùng thông tin của từng dòng)
        df = self.feature_engineering(df)

        # 4) Missing (fit imputer)
        df = self.handle_missing(df, is_train=True)

        # 5) Outliers (fit bounds)
        df = self.detect_outliers(df, is_train=True)

        # 6) Encoding (fit encoder)
        df = self.encode_categoricals(df, is_train=True)

        # 7) Scaling (fit scaler)
        df = self.scale_features(df, is_train=True)

        self._fitted = True
        logger.info("Fit DataPreprocessor hoàn tất.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Áp dụng pipeline đã học cho tập Test hoặc dữ liệu mới.
        CHỈ dùng các thống kê đã học từ Train (không học lại).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame chứa features (+ có thể có target nếu là tập test có nhãn).

        Returns
        -------
        pd.DataFrame
        """
        if not self._fitted:
            raise DataPreprocessorError(
                "DataPreprocessor chưa được fit. Hãy gọi .fit(train_df) trước."
            )

        df = df.copy()
        logger.info("Bắt đầu transform dữ liệu. Shape: %s", df.shape)

        # Không cập nhật self._numeric_cols/... nữa để tránh lệch giữa Train/Test
        df = self.detect_hidden_missing(df)
        df = self.feature_engineering(df)
        df = self.handle_missing(df, is_train=False)
        df = self.detect_outliers(df, is_train=False)
        df = self.encode_categoricals(df, is_train=False)
        df = self.scale_features(df, is_train=False)

        logger.info("Transform dữ liệu hoàn tất. Shape mới: %s", df.shape)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tiện ích: fit + transform trên cùng tập (thường dùng cho Train).
        """
        return self.fit(df).transform(df)

    def run_full_preprocessing(
        self,
        df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Chạy full pipeline:
        1) Load dữ liệu nếu df=None.
        2) Hidden missing + feature engineering.
        3) Chia Train/Test (stratify theo Outcome).
        4) Fit trên Train, transform Test (CHỐNG DATA LEAKAGE).
        5) Trả về (X_train_processed, X_test_processed, y_train, y_test).

        Returns
        -------
        X_train_proc, X_test_proc, y_train, y_test
        """
        if df is None:
            df = self.load_data()

        if self.config.target_col not in df.columns:
            raise DataPreprocessorError(
                f"Dữ liệu đầu vào không chứa cột target: {self.config.target_col}"
            )

        logger.info("Bắt đầu run_full_preprocessing. Tổng shape: %s", df.shape)

        # Hidden missing + feature engineering có thể làm trên toàn bộ
        # vì KHÔNG dùng thống kê toàn bộ để "học" (chỉ thay đổi từng dòng).
        df = self.detect_hidden_missing(df)
        df = self.feature_engineering(df)

        target_col = self.config.target_col
        X = df.drop(columns=[target_col])
        y = df[target_col]

        split_cfg = self.config.split
        stratify = y if split_cfg.get("stratify", True) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=split_cfg.get("test_size", 0.2),
            random_state=split_cfg.get("random_state", 42),
            stratify=stratify if stratify is not None else None,
        )

        logger.info(
            "Chia train/test xong. X_train: %s, X_test: %s",
            X_train.shape,
            X_test.shape,
        )

        # Gộp lại với target để dùng cho handle_missing (median_by_outcome)
        train_df = X_train.copy()
        train_df[target_col] = y_train

        test_df = X_test.copy()
        test_df[target_col] = y_test

        # Fit trên train, transform test
        train_processed = self.fit_transform(train_df)
        test_processed = self.transform(test_df)

        # Tách lại X, y
        X_train_proc = train_processed.drop(columns=[target_col])
        X_test_proc = test_processed.drop(columns=[target_col])

        # Lưu processed nếu có cấu hình
        if self.config.processed_train_path is not None:
            self.save_processed(
                train_processed, self.config.processed_train_path
            )
        if self.config.processed_test_path is not None:
            self.save_processed(
                test_processed, self.config.processed_test_path
            )

        return X_train_proc, X_test_proc, y_train, y_test

    # ------------------------------------------------------------------
    # Các bước con trong pipeline
    # ------------------------------------------------------------------
    def _infer_column_types(self, df: pd.DataFrame) -> None:
        """
        Tự động phân loại cột numeric / categorical / datetime.
        """
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

        target_col = self.config.target_col

        # Không xem target là categorical dù là 0/1 (đề phòng encode nhầm)
        if target_col in cat_cols:
            cat_cols.remove(target_col)

        self._numeric_cols = num_cols
        self._categorical_cols = cat_cols
        self._datetime_cols = dt_cols

        logger.info(
            "Phân loại cột: numeric=%s, categorical=%s, datetime=%s",
            self._numeric_cols,
            self._categorical_cols,
            self._datetime_cols,
        )

    def detect_hidden_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Thay thế các giá trị 0 "vô lý y khoa" thành NaN cho các cột cấu hình.
        (Hidden missing).
        """
        df = df.copy()
        cols = self.config.hidden_missing_cols

        for col in cols:
            if col not in df.columns:
                logger.warning(
                    "Cột %s không tồn tại trong dữ liệu, bỏ qua hidden missing.",
                    col,
                )
                continue
            zeros_before = (df[col] == 0).sum()
            if zeros_before > 0:
                logger.info(
                    "Cột %s có %d giá trị 0 (coi là missing ẩn). Thay bằng NaN.",
                    col,
                    zeros_before,
                )
                df.loc[df[col] == 0, col] = np.nan

        return df

    def handle_missing(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """
        Xử lý missing cho numeric & categorical.

        Numeric:
            - median_overall: median toàn tập Train
            - median_by_outcome: median theo Outcome (config.target_col)

        Categorical:
            - Điền mode (most frequent).

        CHỈ fit thống kê khi is_train=True, lưu lại để dùng cho Test.
        """
        df = df.copy()
        target_col = self.config.target_col
        num_strategy = self.config.missing.get("numeric_strategy", "median_overall")
        cat_strategy = self.config.missing.get("categorical_strategy", "most_frequent")

        # Numeric
        numeric_cols = [c for c in self._numeric_cols if c in df.columns and c != target_col]

        if is_train:
            # Tính median overall cho tất cả numeric
            self._numeric_impute_overall_ = {
                col: float(df[col].median()) for col in numeric_cols
            }

            # Nếu median_by_outcome, tính thêm median theo nhóm Outcome
            if num_strategy == "median_by_outcome" and target_col in df.columns:
                grouped = df.groupby(target_col)
                for col in numeric_cols:
                    self._numeric_impute_by_outcome_[col] = grouped[col].median()
                    logger.info(
                        "Imputer numeric (median_by_outcome) cột %s: %s",
                        col,
                        self._numeric_impute_by_outcome_[col].to_dict(),
                    )

        # Áp dụng imputer numeric
        for col in numeric_cols:
            if df[col].isna().any():

                # Nếu dùng median_by_outcome và có target_col → ưu tiên median theo outcome
                if num_strategy == "median_by_outcome" and target_col in df.columns:

                    med_by_outcome = self._numeric_impute_by_outcome_.get(col)
                    overall = self._numeric_impute_overall_.get(col)

                    def _fill_row(row: pd.Series) -> Any:
                        if pd.isna(row[col]):
                            outcome = row.get(target_col, None)

                            # Nếu hàng có outcome và median_by_outcome có giá trị → dùng
                            if (
                                med_by_outcome is not None
                                and outcome in med_by_outcome.index
                                and not np.isnan(med_by_outcome.loc[outcome])
                            ):
                                return med_by_outcome.loc[outcome]

                            # fallback: median overall
                            return overall

                        return row[col]

                    df[col] = df.apply(_fill_row, axis=1)
                    logger.info(
                        "Impute missing numeric (median_by_outcome) cho cột %s.",
                        col,
                    )

                else:
                    # median_overall
                    value = self._numeric_impute_overall_.get(col)
                    df[col] = df[col].fillna(value)
                    logger.info(
                        "Impute missing numeric (median_overall) cho cột %s: %.4f",
                        col,
                        value,
                    )

        # Categorical
        cat_cols = [c for c in self._categorical_cols if c in df.columns]

        if is_train:
            self._categorical_impute_ = {}
            if cat_strategy == "most_frequent":
                for col in cat_cols:
                    mode_val = df[col].mode(dropna=True)
                    if len(mode_val) > 0:
                        self._categorical_impute_[col] = mode_val.iloc[0]
                    else:
                        self._categorical_impute_[col] = "missing"
                    logger.info(
                        "Imputer categorical (most_frequent) cột %s: %s",
                        col,
                        self._categorical_impute_[col],
                    )

        for col in cat_cols:
            if df[col].isna().any():
                fill_val = self._categorical_impute_.get(col, "missing")
                df[col] = df[col].fillna(fill_val)
                logger.info("Impute missing categorical cho cột %s.", col)

        return df

    @staticmethod
    def _compute_iqr_bounds(
        series: pd.Series, factor: float
    ) -> Tuple[float, float]:
        """
        Tính ngưỡng outlier theo IQR cho một Series.
        Dùng @staticmethod để thể hiện kỹ thuật Python chuẩn.
        """
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        return lower, upper

    def detect_outliers(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """
        Phát hiện & xử lý outlier bằng IQR.

        - method: iqr (Q1 - 1.5 IQR, Q3 + 1.5 IQR)
        - strategy:
            + winsorize: kẹp giá trị về [lower, upper]
            + flag: thêm cột is_outlier_<col> (0/1)
            + none: bỏ qua
        """
        df = df.copy()
        out_cfg = self.config.outlier
        method = out_cfg.get("method", "iqr")
        strategy = out_cfg.get("strategy", "winsorize")
        factor = float(out_cfg.get("iqr_factor", 1.5))
        target_col = self.config.target_col

        if method != "iqr" or strategy == "none":
            logger.info(
                "Outlier method=%s hoặc strategy=none, bỏ qua detect_outliers.",
                method,
            )
            return df

        if is_train:
            self._outlier_bounds_ = {}

        # Cột numeric áp dụng outlier
        if out_cfg.get("numeric_cols") is not None:
            numeric_cols = [
                c
                for c in out_cfg["numeric_cols"]
                if c in df.columns and c != target_col
            ]
        else:
            numeric_cols = [
                c for c in self._numeric_cols if c in df.columns and c != target_col
            ]

        for col in numeric_cols:
            if is_train:
                lower, upper = self._compute_iqr_bounds(df[col], factor)
                self._outlier_bounds_[col] = (lower, upper)
                logger.info(
                    "IQR bounds cho cột %s: lower=%.4f, upper=%.4f",
                    col,
                    lower,
                    upper,
                )
            else:
                if col not in self._outlier_bounds_:
                    # chưa fit bounds cho cột này (có thể do cấu hình)
                    continue
                lower, upper = self._outlier_bounds_[col]

            if strategy == "winsorize":
                df[col] = df[col].clip(lower, upper)
            elif strategy == "flag":
                flag_col = f"is_outlier_{col}"
                df[flag_col] = ((df[col] < lower) | (df[col] > upper)).astype(int)

        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo feature mới mang ý nghĩa y khoa:
        - BMI_category: Underweight / Normal / Overweight / Obese
        - Age_group: 21–30 / 31–40 / 41–50 / >50
        - Pregnancy_high: 1 nếu Pregnancies >= 3
        - Interaction: Glucose * BMI, Insulin / Glucose
        """
        cfg = self.config.feature_engineering
        if not cfg.get("enable", True):
            return df

        df = df.copy()

        bmi_col = cfg.get("bmi_col", "BMI")
        age_col = cfg.get("age_col", "Age")
        preg_col = cfg.get("pregnancies_col", "Pregnancies")
        glu_col = cfg.get("glucose_col", "Glucose")
        ins_col = cfg.get("insulin_col", "Insulin")

        # BMI category
        if cfg.get("create_bmi_category", True) and bmi_col in df.columns:
            df["BMI_category"] = pd.cut(
                df[bmi_col],
                bins=[0, 18.5, 25, 30, np.inf],
                labels=["Underweight", "Normal", "Overweight", "Obese"],
                include_lowest=True,
            )

        # Age group
        if cfg.get("create_age_group", True) and age_col in df.columns:
            df["Age_group"] = pd.cut(
                df[age_col],
                bins=[20, 30, 40, 50, 120],
                labels=["21-30", "31-40", "41-50", ">50"],
                include_lowest=True,
            )

        # Pregnancy_high
        if cfg.get("create_pregnancy_flag", True) and preg_col in df.columns:
            df["Pregnancy_high"] = (df[preg_col] >= 3).astype(int)

        # Interactions
        if cfg.get("create_interactions", True):
            if glu_col in df.columns and bmi_col in df.columns:
                df["Glucose_x_BMI"] = df[glu_col] * df[bmi_col]
            if ins_col in df.columns and glu_col in df.columns:
                # tránh chia cho 0
                df["Insulin_div_Glucose"] = df[ins_col] / (df[glu_col] + 1e-6)

        logger.info("Feature engineering hoàn tất. Shape mới: %s", df.shape)
        return df

    @staticmethod
    def encode_column(series: pd.Series) -> Tuple[pd.Series, Dict[Any, int]]:
        """
        Ví dụ dùng @staticmethod + LabelEncoder cho 1 cột.
        Trả về Series đã encode và mapping category → integer.
        """
        le = LabelEncoder()
        encoded = le.fit_transform(series.astype(str))
        mapping = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
        return pd.Series(encoded, index=series.index), mapping

    def encode_categoricals(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """
        Mã hóa các biến categorical theo config.encoding.strategy:
        - onehot: dùng pd.get_dummies, align cột giữa Train/Test.
        - label: dùng LabelEncoder cho từng cột.
        - none: bỏ qua.

        Trong Pima dataset gốc gần như không có categorical, nhưng
        class được viết tổng quát để tái sử dụng.
        """
        df = df.copy()
        target_col = self.config.target_col
        strategy = self._encoding_strategy

        cat_cols = [c for c in self._categorical_cols if c in df.columns]

        if not cat_cols or strategy == "none":
            return df

        # Không encode target nếu lỡ là categorical
        if target_col in cat_cols:
            cat_cols.remove(target_col)

        if not cat_cols:
            return df

        if strategy == "label":
            # LabelEncoder cho từng cột
            if is_train:
                self._label_encoders_ = {}
                self._categorical_cols_fitted_ = cat_cols.copy()

                for col in cat_cols:
                    le = LabelEncoder()
                    # Fill NaN tạm để encode, missing đã xử lý ở handle_missing
                    df[col] = df[col].astype(str)
                    df[col] = le.fit_transform(df[col])
                    self._label_encoders_[col] = le
                    logger.info(
                        "LabelEncoder fit cho cột %s. Classes: %s",
                        col,
                        list(le.classes_),
                    )
            else:
                # Dùng encoder đã fit
                for col in self._categorical_cols_fitted_:
                    if col not in df.columns:
                        continue
                    le = self._label_encoders_.get(col)
                    if le is None:
                        continue
                    # Xử lý unknown bằng cách map sang -1
                    df[col] = df[col].astype(str)
                    known_classes = set(le.classes_)
                    df[col] = df[col].apply(
                        lambda x: x
                        if x in known_classes
                        else "__unknown__"
                    )
                    if "__unknown__" not in le.classes_:
                        le.classes_ = np.append(le.classes_, "__unknown__")
                    df[col] = le.transform(df[col])

            return df

        # One-hot encoding
        if strategy == "onehot":
            if is_train:
                self._categorical_cols_fitted_ = cat_cols.copy()
                dummies = pd.get_dummies(
                    df[cat_cols],
                    prefix=cat_cols,
                    drop_first=False,
                )

                self._onehot_columns_ = dummies.columns.tolist()

                # Bỏ cột cat gốc, ghép dummies
                df = df.drop(columns=cat_cols)
                df = pd.concat([df, dummies], axis=1)
            else:
                # Encode bằng get_dummies rồi align cột
                dummies = pd.get_dummies(
                    df[cat_cols],
                    prefix=cat_cols,
                    drop_first=False,
                )

                # Align với train: thêm cột thiếu, bỏ cột thừa
                for col in self._onehot_columns_:
                    if col not in dummies.columns:
                        dummies[col] = 0

                extra_cols = [
                    c for c in dummies.columns if c not in self._onehot_columns_
                ]
                if extra_cols:
                    dummies = dummies.drop(columns=extra_cols)

                dummies = dummies[self._onehot_columns_]

                df = df.drop(columns=cat_cols)
                df = pd.concat([df, dummies], axis=1)

            logger.info("One-hot encoding hoàn tất. Shape mới: %s", df.shape)
            return df

        return df

    def scale_features(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """
        Chuẩn hóa dữ liệu numeric theo config.scaler:
        - type: standard (StandardScaler) / minmax (MinMaxScaler) / none
        - exclude_cols: danh sách cột KHÔNG scale (vd: target, id...)

        Chỉ fit scaler trên Train, sau đó dùng cho Test.
        """
        df = df.copy()
        scaler_cfg = self.config.scaler
        scaler_type = scaler_cfg.get("type", "standard")
        exclude_cols = set(scaler_cfg.get("exclude_cols", []))
        exclude_cols.add(self.config.target_col)

        if scaler_type == "none":
            return df

        # Cột numeric hiện tại (sau feature engineering, impute, v.v.)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols_to_scale = [
            c for c in numeric_cols if c not in exclude_cols
        ]

        if not numeric_cols_to_scale:
            logger.warning(
                "Không có cột numeric nào để scale (sau khi exclude), bỏ qua scale_features."
            )
            return df

        if is_train:
            if scaler_type == "standard":
                self.scaler = StandardScaler()
            elif scaler_type == "minmax":
                self.scaler = MinMaxScaler()
            else:
                raise DataPreprocessorError(
                    f"Loại scaler không được hỗ trợ: {scaler_type}"
                )

            self._scaled_feature_names_ = numeric_cols_to_scale.copy()
            self.scaler.fit(df[self._scaled_feature_names_])
            logger.info(
                "Fit scaler (%s) cho các cột: %s",
                scaler_type,
                self._scaled_feature_names_,
            )

            # Lưu scaler nếu có đường dẫn
            save_path = scaler_cfg.get("save_scaler_path", None)
            if save_path is not None:
                save_path = Path(save_path)
                if not save_path.is_absolute():
                    save_path = self.data_processed_dir / save_path
                save_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(self.scaler, save_path)
                logger.info("Đã lưu scaler vào: %s", save_path)
        else:
            if self.scaler is None:
                raise DataPreprocessorError(
                    "Scaler chưa được fit nhưng đang gọi scale_features với is_train=False."
                )

        # Áp dụng scaler cho cả train lẫn test
        df[self._scaled_feature_names_] = self.scaler.transform(
            df[self._scaled_feature_names_]
        )

        return df

    def save_processed(self, df: pd.DataFrame, path: Union[str, Path]) -> None:
        """
        Lưu dữ liệu đã tiền xử lý vào data/processed (hoặc đường dẫn tùy chọn).
        Hỗ trợ csv/parquet theo đuôi file.
        """
        path = Path(path)
        if not path.is_absolute():
            path = self.data_processed_dir / path

        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix.lower() == ".csv":
            df.to_csv(path, index=False)
        elif path.suffix.lower() in {".parquet", ".pq"}:
            df.to_parquet(path, index=False)
        else:
            # mặc định dùng csv
            path = path.with_suffix(".csv")
            df.to_csv(path, index=False)

        logger.info("Đã lưu dữ liệu processed vào: %s", path)


class LoggingPreprocessor(DataPreprocessor):
    """
    Kế thừa DataPreprocessor:
    - Giữ nguyên toàn bộ logic fit() của class cha.
    - Ghi thêm log thống kê missing để minh họa kế thừa + mở rộng hành vi.
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None) -> None:  # <<< MỚI
        # Gọi lại __init__ của class cha
        super().__init__(config=config)
        logger.info(
            ">>> Đang dùng LoggingPreprocessor (kế thừa DataPreprocessor)"  # <<< MỚI
        )

    def fit(self, df: pd.DataFrame) -> "LoggingPreprocessor":  # <<< MỚI
        """
        Override fit:
        - Trước tiên gọi logic fit() gốc của DataPreprocessor (super().fit).
        - Sau đó log thêm thống kê missing để minh họa mở rộng.
        """
        # Thống kê missing TRƯỚC khi fit (minh họa thêm hành vi)
        missing_before = df.isna().sum().to_dict()
        logger.info(
            "[LoggingPreprocessor] Missing BEFORE fit: %s",
            missing_before,
        )

        # Gọi logic gốc của DataPreprocessor
        super().fit(df)

        # Lưu lại summary (trên df gốc) để trình bày với cô
        self.missing_summary_ = df.isna().sum().to_dict()
        logger.info(
            "[LoggingPreprocessor] Missing summary sau fit (trên df gốc): %s",
            self.missing_summary_,
        )

        return self
#