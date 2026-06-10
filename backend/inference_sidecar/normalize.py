"""Convert Donut's raw CORD-schema output into the camelCase / float receipt
shape that the Express API and frontend expect.

Donut emits the raw CORD-v2 schema directly (string values, locale-dependent
thousand separators). This module:
  1. Maps CORD keys to clean keys (e.g. ``menu.unitprice`` → ``unit_price``).
  2. Parses every numeric field into a float, handling Indonesian-style dots,
     comma thousand separators, and decimal points robustly.
  3. Drops empty fields entirely (the frontend treats missing == "not on receipt").

The output matches the ``Receipt`` TypeScript type in
``frontend/src/lib/api.ts`` exactly.
"""

from __future__ import annotations

import re
from typing import Any

# --------------------------------------------------------------------------- #
# CORD key → clean key mappings.
# --------------------------------------------------------------------------- #

_LINE_ITEM_KEYS: dict[str, str] = {
    "nm":            "name",
    "sub_nm":        "sub_name",
    "num":           "item_num",
    "cnt":           "quantity",
    "unitprice":     "unit_price",
    "price":         "price",
    "discountprice": "discount",
}

_SUB_TOTAL_KEYS: dict[str, str] = {
    "subtotal_price": "subtotal",
    "discount_price": "discount",
    "service_price":  "service_charge",
    "tax_price":      "tax",
}

_TOTAL_KEYS: dict[str, str] = {
    "total_price":      "total",
    "cashprice":        "cash_paid",
    "changeprice":      "change",
    "creditcardprice":  "credit_card",
    "emoneyprice":      "e_money",
}

# Per-line-item keys whose values should be parsed as floats.
_NUMERIC_LINE_ITEM_KEYS: set[str] = {"quantity", "unit_price", "price", "discount"}


# --------------------------------------------------------------------------- #
# Number parsing.
# --------------------------------------------------------------------------- #

_NUM_KEEP = re.compile(r"[^\d.,\-]")


def _parse_number(value: Any) -> float | None:
    """Coerce a CORD-style number string to ``float``. Returns ``None`` for
    unparseable input.

    Handles every separator convention observed in CORD:
        "23,000"          → 23000.0   (US comma thousand)
        "23.000"          → 23000.0   (Indonesian dot thousand)
        "1,250,000"       → 1250000.0
        "1.250.000"       → 1250000.0
        "1,250.50"        → 1250.5    (US — dot decimal, comma thousand)
        "1.250,50"        → 1250.5    (EU — comma decimal, dot thousand)
        "23.5"            → 23.5      (single dot, non-3-digit tail = decimal)
        "Rp 23.000"       → 23000.0   (strips currency)
        23000             → 23000.0   (already numeric)
    """
    if value is None:
        return None
    if isinstance(value, bool):
        # bool is a subclass of int in Python — guard so True/False aren't
        # silently converted to 1.0 / 0.0.
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    s = _NUM_KEEP.sub("", value).strip()
    if not s or s in {".", ",", "-"}:
        return None

    has_dot = "." in s
    has_comma = "," in s

    if has_dot and has_comma:
        # Both present — the rightmost separator is the decimal point.
        if s.rfind(".") > s.rfind(","):
            # "1,250.50" → dot is decimal, comma is thousand.
            s = s.replace(",", "")
        else:
            # "1.250,50" → comma is decimal, dot is thousand.
            s = s.replace(".", "").replace(",", ".")
    elif has_comma:
        s = _resolve_single_separator(s, ",")
    elif has_dot:
        s = _resolve_single_separator(s, ".")

    try:
        return float(s)
    except ValueError:
        return None


def _resolve_single_separator(s: str, sep: str) -> str:
    """Decide whether a lone ``,`` (or ``.``) is a thousand or decimal separator.

    Heuristic: a trailing 3-digit group preceded by a non-empty prefix is a
    thousand separator (``23,000`` / ``1,250,000``). Anything else is a decimal
    point (``23.5`` / ``0,99``).
    """
    parts = s.split(sep)
    if len(parts) >= 2 and parts[0] and all(len(p) == 3 for p in parts[1:]):
        return s.replace(sep, "")
    if sep == ",":
        return s.replace(",", ".")
    return s  # lone dot = decimal already


# --------------------------------------------------------------------------- #
# Per-block helpers.
# --------------------------------------------------------------------------- #

def _normalise_line_item(raw: Any) -> dict[str, Any] | None:
    """Map one ``menu`` entry from CORD keys to clean keys + parse numbers.

    Returns ``None`` if no usable fields survive parsing — the caller drops
    empty entries entirely so the frontend never sees skeletons.
    """
    if not isinstance(raw, dict):
        return None

    out: dict[str, Any] = {}
    for src_key, src_val in raw.items():
        dst_key = _LINE_ITEM_KEYS.get(src_key)
        if dst_key is None or src_val is None or src_val == "":
            continue
        if dst_key in _NUMERIC_LINE_ITEM_KEYS:
            num = _parse_number(src_val)
            if num is not None:
                out[dst_key] = num
        else:
            text = str(src_val).strip()
            if text:
                out[dst_key] = text
    return out or None


def _normalise_summary(raw: Any, mapping: dict[str, str]) -> dict[str, float]:
    """Map a sub_total / total block: CORD keys → clean keys → floats."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for src_key, src_val in raw.items():
        dst_key = mapping.get(src_key)
        if dst_key is None or src_val is None or src_val == "":
            continue
        num = _parse_number(src_val)
        if num is not None:
            out[dst_key] = num
    return out


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #

def normalize(cord: Any) -> dict[str, Any]:
    """Top-level converter — call this with whatever ``predict()`` returned.

    Always returns a ``dict``. Empty ``{}`` is a valid output (nothing parsed).
    """
    if not isinstance(cord, dict):
        return {}

    # Line items: ``menu`` may be a list, a single dict (when one item),
    # missing, or null. Donut sometimes emits one of each.
    menu = cord.get("menu")
    items: list[dict[str, Any]] = []
    if isinstance(menu, list):
        for raw in menu:
            cleaned = _normalise_line_item(raw)
            if cleaned:
                items.append(cleaned)
    elif isinstance(menu, dict):
        cleaned = _normalise_line_item(menu)
        if cleaned:
            items.append(cleaned)

    receipt: dict[str, Any] = {}
    if items:
        receipt["line_items"] = items

    # Sub-total and total blocks are flat dicts of currency fields.
    # Total wins on the (unused) key collision because that's the canonical
    # receipt-level value.
    for k, v in _normalise_summary(cord.get("sub_total"), _SUB_TOTAL_KEYS).items():
        receipt[k] = v
    for k, v in _normalise_summary(cord.get("total"), _TOTAL_KEYS).items():
        receipt[k] = v

    return receipt
