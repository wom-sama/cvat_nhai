from pathlib import Path

APP_NAME = "CVAT Nhai"
ORG_NAME = "wom-sama"
SCHEMA_VERSION = 1

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)
SPLITS = ("train", "val", "test")

CLASS_NAMES = (
    "Xoai_Song_Chua_KhoDap",
    "Xoai_Song_ChuaNhe_CoNguyCo",
    "Xoai_Chin_NgotThanh_DeDap",
    "Xoai_ChinGia_NgotGat_KhongVanChuyen",
    "Xoai_Hu_KhongAnDuoc",
)

CLASS_LABELS = (
    "Xoai song - chua, kho dap",
    "Xoai song - chua nhe, co nguy co",
    "Xoai chin - ngot thanh, de dap",
    "Xoai chin gia - ngot gat, khong van chuyen",
    "Xoai hu - khong an duoc",
)

CLASS_COLORS = (
    "#22C55E",
    "#84CC16",
    "#F59E0B",
    "#F97316",
    "#EF4444",
)

DEFAULT_DETECTION_ROOT = Path(r"D:\DataAI\AIEx\dataset")
DEFAULT_CLASSIFICATION_ROOT = Path(
    r"D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops"
)
