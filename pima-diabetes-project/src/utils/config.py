"""
Config chung cho project Pima Diabetes.
"""

from pathlib import Path
from typing import Final

# PROJECT_ROOT: thư mục gốc của project pima-diabetes-project
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

# ==== Đường dẫn dữ liệu ====
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_PATH: Final[Path] = DATA_DIR / "raw" / "diabetes.csv"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"

# ==== Đường dẫn models / results / reports ====
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"

# ==== Tham số chung cho thí nghiệm ====
RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.2
