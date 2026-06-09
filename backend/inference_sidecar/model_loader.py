import os

# Import torch BEFORE paddleocr — paddleocr's albumentations dep loads torch
# in a way that crashes if Paddle gets its DLLs in first.
import torch
from paddleocr import PaddleOCR
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

_HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.join(_HERE, "..", "..", "model", "cord_layoutlmv3(v1)"),
)
FORCE_CPU = os.environ.get("FORCE_CPU", "false").lower() == "true"
OCR_LANG = os.environ.get("OCR_LANG", "en")
OCR_MIN_CONFIDENCE = float(os.environ.get("OCR_MIN_CONFIDENCE", "0.5"))
OCR_UPSCALE_TO = int(os.environ.get("OCR_UPSCALE_TO", "2000"))


class ModelBundle:
    def __init__(self):
        self.model = None
        self.processor = None
        self.ocr = None
        self.device = None
        self.is_loaded = False
        self.ocr_min_confidence = OCR_MIN_CONFIDENCE
        self.ocr_upscale_to = OCR_UPSCALE_TO

    def load(self):
        use_gpu = not FORCE_CPU and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_gpu else "cpu")
        print(f"Loading model from {MODEL_DIR} on {self.device} ...")
        self.processor = LayoutLMv3Processor.from_pretrained(MODEL_DIR, apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR)
        self.model.to(self.device)
        self.model.eval()
        print(f"Loading PaddleOCR (lang={OCR_LANG}, gpu={use_gpu}) ...")
        # Tuned for receipt OCR — accuracy over speed.
        #   ocr_version='PP-OCRv4'  — latest, most accurate
        #   det_limit_side_len=2400 — detect at higher res (default 960 misses small text)
        #   det_db_box_thresh=0.4   — keep more boxes (default 0.5)
        #   det_db_unclip_ratio=2.0 — expand boxes more (default ~1.6); helps tight receipts
        #   det_db_score_mode=slow  — accurate scoring (default 'fast')
        #   use_dilation=True       — better recall on thin/small chars
        #   drop_score=0.3          — let weak recs through; we re-filter with OCR_MIN_CONFIDENCE
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=OCR_LANG,
            use_gpu=use_gpu,
            show_log=False,
            ocr_version="PP-OCRv4",
            det_limit_side_len=2400,
            det_limit_type="max",
            det_db_thresh=0.3,
            det_db_box_thresh=0.4,
            det_db_unclip_ratio=2.0,
            det_db_score_mode="slow",
            use_dilation=True,
            drop_score=0.3,
            use_space_char=True,
        )
        self.is_loaded = True
        print("Model ready.")

    @property
    def device_str(self):
        return str(self.device) if self.device else "unknown"


bundle = ModelBundle()
