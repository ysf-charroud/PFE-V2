import os
import easyocr
import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

_HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.join(_HERE, "..", "..", "model", "cord_layoutlmv3(v1)"),
)
FORCE_CPU = os.environ.get("FORCE_CPU", "false").lower() == "true"
# EasyOCR expects a list of language codes e.g. ["en"] or ["en", "fr"]
OCR_LANGS = [l.strip() for l in os.environ.get("OCR_LANG", "en").split(",")]
OCR_MIN_CONFIDENCE = float(os.environ.get("OCR_MIN_CONFIDENCE", "0.5"))
OCR_UPSCALE_TO = int(os.environ.get("OCR_UPSCALE_TO", "1600"))


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
        self.ocr = easyocr.Reader(OCR_LANGS, gpu=use_gpu)
        self.is_loaded = True
        print("Model ready.")

    @property
    def device_str(self):
        return str(self.device) if self.device else "unknown"


bundle = ModelBundle()
