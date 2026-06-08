import re

_LINE_ITEM_FIELDS = {
    "menu.nm":            "name",
    "menu.num":           "item_num",
    "menu.cnt":           "quantity",
    "menu.price":         "price",
    "menu.unitprice":     "unit_price",
    "menu.discountprice": "discount",
    "menu.itemsubtotal":  "item_subtotal",
    "menu.sub.nm":        "sub_name",
    "menu.sub.cnt":       "sub_quantity",
    "menu.sub.price":     "sub_price",
    "menu.sub.unitprice": "sub_unit_price",
    "menu.etc":           None,
    "menu.vatyn":         None,
}

_SUMMARY_FIELDS = {
    "sub_total.subtotal_price":  "subtotal",
    "sub_total.discount_price":  "discount",
    "sub_total.service_price":   "service_charge",
    "sub_total.othersvc_price":  "other_charges",
    "sub_total.tax_price":       "tax",
    "sub_total.etc":             None,
    "total.total_price":         "total",
    "total.cashprice":           "cash_paid",
    "total.changeprice":         "change",
    "total.creditcardprice":     "credit_card",
    "total.emoneyprice":         "e_money",
    "total.menuqty_cnt":         "item_quantity_count",
    "total.menutype_cnt":        "item_type_count",
    "total.total_etc":           None,
    "void_menu.nm":              None,
    "void_menu.price":           None,
}

_NUMERIC = {
    "price", "unit_price", "discount", "item_subtotal",
    "quantity", "sub_quantity",
    "sub_price", "sub_unit_price", "subtotal", "service_charge",
    "other_charges", "tax", "total", "cash_paid", "change",
    "credit_card", "e_money", "item_quantity_count", "item_type_count",
}

_NOISE_RE = re.compile(r'^[\s*\-_|]+$')


def _is_noise(s: str) -> bool:
    return not s.strip() or bool(_NOISE_RE.match(s))


def _parse_number(raw: str):
    s = re.sub(r'[^\d.,]', '', raw.strip())
    if not s:
        return None
    if re.fullmatch(r'\d{1,3}(\.\d{3})+', s):          # 23.000 / 1.234.567  (dot-thousands)
        return float(s.replace('.', ''))
    if re.fullmatch(r'\d{1,3}(,\d{3})+', s):           # 33,000 / 1,234,567  (comma-thousands)
        return float(s.replace(',', ''))
    if re.fullmatch(r'\d{1,3}(\.\d{3})+,\d+', s):      # 1.234,50
        return float(s.replace('.', '').replace(',', '.'))
    if re.fullmatch(r'\d{1,3}(,\d{3})+\.\d+', s):      # 1,234.50
        return float(s.replace(',', ''))
    if re.fullmatch(r'\d+,\d+', s):                     # 23,50  (comma-decimal, last resort)
        return float(s.replace(',', '.'))
    try:
        return float(s)
    except ValueError:
        return None


def _extract_number(text: str):
    """Extract the largest number from a mixed text+number string like 'Total 23.000'.
    Handles OCR-fragmented thousands in all common forms:
      '33 000'  '33 ,000'  '33, 000'  '3 300'  '36 ,300'  → collapsed before parsing."""
    t = text
    # Collapse "digit [space] separator [space] 3-digits" → "digit3-digits"
    # Covers: "33 ,000", "33, 000", "33 . 000", "3 ,300"
    t = re.sub(r'(\d{1,3})\s*[,\.]\s*(\d{3})(?!\d)', lambda m: m.group(1) + m.group(2), t)
    # Collapse plain space-separated thousands: "33 000", "50 000"
    t = re.sub(r'\b(\d{1,3})\s+(\d{3})\b', lambda m: m.group(1) + m.group(2), t)
    # Second pass for multi-group numbers: "1 234 567" → "1234567"
    t = re.sub(r'\b(\d+)\s+(\d{3})\b', lambda m: m.group(1) + m.group(2), t)
    candidates = re.findall(r'\d[\d.,]*', t)
    values = [v for v in (_parse_number(c) for c in candidates) if v is not None]
    return max(values, default=None)


def _coerce(clean_key: str, raw_value: str):
    if clean_key in _NUMERIC:
        return _extract_number(raw_value)
    return raw_value.strip() or None


def normalize(raw: dict) -> dict:
    receipt = {}

    clean_items = []
    for row in raw.get("line_items", []):
        item = {}
        for cord_label, raw_value in row.items():
            clean_key = _LINE_ITEM_FIELDS.get(cord_label)
            if clean_key is None or _is_noise(raw_value):
                continue
            value = _coerce(clean_key, raw_value)
            if value is not None:
                item[clean_key] = value
        if "name" in item or "price" in item:
            clean_items.append(item)
    receipt["line_items"] = clean_items

    for cord_label, raw_value in raw.get("summary", {}).items():
        clean_key = _SUMMARY_FIELDS.get(cord_label)
        if clean_key is None or _is_noise(raw_value):
            continue
        value = _coerce(clean_key, raw_value)
        if value is not None and clean_key not in receipt:
            receipt[clean_key] = value

    return receipt
