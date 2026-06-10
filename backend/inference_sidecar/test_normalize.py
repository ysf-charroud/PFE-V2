"""Tests for normalize.py — run with: python test_normalize.py

Covers the tricky number-format cases (Indonesian dots, comma thousands,
mixed punctuation) plus the menu single-dict-vs-list quirk and empty-field
dropping. No external dependencies — pure Python so the sidecar venv is
not required.
"""

import sys

from normalize import _parse_number, normalize


def expect_equal(label, got, want):
    if got == want:
        print(f"  ok    {label}: {got!r}")
        return True
    print(f"  FAIL  {label}: got {got!r}, want {want!r}")
    return False


def test_parse_number():
    print("test _parse_number")
    cases = [
        ("23,000",        23000.0),
        ("23.000",        23000.0),
        ("1,250,000",     1250000.0),
        ("1.250.000",     1250000.0),
        ("1,250.50",      1250.50),
        ("1.250,50",      1250.50),
        ("23.5",          23.5),
        ("0,99",          0.99),
        ("23000",         23000.0),
        ("Rp 23.000",     23000.0),
        ("$ 1,250.99",    1250.99),
        ("  9,000  ",     9000.0),
        (23000,           23000.0),
        (23000.5,         23000.5),
        ("",              None),
        ("   ",           None),
        (".",             None),
        (",",             None),
        ("abc",           None),
        (None,            None),
        (True,            None),    # bool guard
        (False,           None),
    ]
    fails = 0
    for raw, want in cases:
        got = _parse_number(raw)
        if not expect_equal(f"_parse_number({raw!r})", got, want):
            fails += 1
    return fails


def test_normalize_full_receipt():
    print("\ntest normalize() — full CORD parse")
    cord = {
        "menu": [
            {
                "nm": "Caffe Latte",
                "num": "1",
                "cnt": "2",
                "unitprice": "4,500",
                "price": "9,000",
                "discountprice": "500",
            },
            {
                "nm": "Espresso",
                "cnt": "1",
                "price": "3.500",  # Indonesian thousand-dot
            },
        ],
        "sub_total": {
            "subtotal_price": "12.500",
            "discount_price": "500",
            "service_price": "900",
            "tax_price": "1,200",
        },
        "total": {
            "total_price":   "14,100",
            "cashprice":     "20,000",
            "changeprice":   "5,900",
        },
    }
    out = normalize(cord)

    fails = 0
    fails += 0 if expect_equal("line_items length", len(out.get("line_items", [])), 2) else 1
    fails += 0 if expect_equal("li[0].name",       out["line_items"][0]["name"],       "Caffe Latte") else 1
    fails += 0 if expect_equal("li[0].item_num",   out["line_items"][0]["item_num"],   "1") else 1
    fails += 0 if expect_equal("li[0].quantity",   out["line_items"][0]["quantity"],   2.0) else 1
    fails += 0 if expect_equal("li[0].unit_price", out["line_items"][0]["unit_price"], 4500.0) else 1
    fails += 0 if expect_equal("li[0].price",      out["line_items"][0]["price"],      9000.0) else 1
    fails += 0 if expect_equal("li[0].discount",   out["line_items"][0]["discount"],   500.0) else 1
    fails += 0 if expect_equal("li[1].price",      out["line_items"][1]["price"],      3500.0) else 1
    fails += 0 if expect_equal("subtotal",       out["subtotal"],       12500.0) else 1
    fails += 0 if expect_equal("discount",       out["discount"],       500.0)  else 1
    fails += 0 if expect_equal("service_charge", out["service_charge"], 900.0)  else 1
    fails += 0 if expect_equal("tax",            out["tax"],            1200.0) else 1
    fails += 0 if expect_equal("total",          out["total"],          14100.0) else 1
    fails += 0 if expect_equal("cash_paid",      out["cash_paid"],      20000.0) else 1
    fails += 0 if expect_equal("change",         out["change"],         5900.0)  else 1
    return fails


def test_menu_as_single_dict():
    print("\ntest normalize() — menu as single dict")
    cord = {"menu": {"nm": "Only Item", "price": "10,000"}}
    out = normalize(cord)
    fails = 0
    fails += 0 if expect_equal("items length", len(out.get("line_items", [])), 1) else 1
    fails += 0 if expect_equal("li[0].name",   out["line_items"][0]["name"],   "Only Item") else 1
    fails += 0 if expect_equal("li[0].price",  out["line_items"][0]["price"],  10000.0)    else 1
    return fails


def test_empty_and_missing_fields_dropped():
    print("\ntest normalize() — empty/missing fields are dropped")
    cord = {
        "menu": [
            {"nm": "Item", "price": "5,000", "discountprice": ""},   # empty discount dropped
            {"nm": "", "price": ""},                                 # entirely empty → dropped
            "garbage",                                               # non-dict → dropped
        ],
        "sub_total": {"subtotal_price": "5,000", "tax_price": None}, # None → dropped
        "total": {"total_price": "5,000"},
    }
    out = normalize(cord)
    fails = 0
    fails += 0 if expect_equal("items length",                   len(out["line_items"]), 1) else 1
    fails += 0 if expect_equal("'discount' not in li[0]",        "discount" not in out["line_items"][0], True) else 1
    fails += 0 if expect_equal("'tax' not in receipt (was None)", "tax" not in out, True) else 1
    fails += 0 if expect_equal("subtotal kept",                   out["subtotal"], 5000.0) else 1
    return fails


def test_malformed_input():
    print("\ntest normalize() — malformed top-level input")
    fails = 0
    fails += 0 if expect_equal("None input",   normalize(None),   {}) else 1
    fails += 0 if expect_equal("empty dict",   normalize({}),     {}) else 1
    fails += 0 if expect_equal("list input",   normalize([1, 2]), {}) else 1
    fails += 0 if expect_equal("string input", normalize("foo"),  {}) else 1
    return fails


def test_unknown_cord_keys_ignored():
    print("\ntest normalize() — unknown CORD keys are silently ignored")
    cord = {
        "menu":      [{"nm": "Item", "price": "1,000", "vatyn": "y", "etc": "junk"}],
        "sub_total": {"subtotal_price": "1,000", "othersvc_price": "50"},
        "total":     {"total_price": "1,000", "menutype_cnt": "1"},
    }
    out = normalize(cord)
    fails = 0
    # Only the mapped keys should appear
    fails += 0 if expect_equal("li[0] keys", set(out["line_items"][0].keys()), {"name", "price"}) else 1
    fails += 0 if expect_equal("receipt keys", set(out.keys()),                {"line_items", "subtotal", "total"}) else 1
    return fails


def test_payment_alternatives():
    print("\ntest normalize() — credit_card / e_money paths")
    cord = {
        "total": {"total_price": "100", "creditcardprice": "100", "emoneyprice": "0"},
    }
    out = normalize(cord)
    fails = 0
    fails += 0 if expect_equal("credit_card", out["credit_card"], 100.0) else 1
    fails += 0 if expect_equal("e_money",     out["e_money"],     0.0)   else 1
    return fails


def main():
    fails = 0
    fails += test_parse_number()
    fails += test_normalize_full_receipt()
    fails += test_menu_as_single_dict()
    fails += test_empty_and_missing_fields_dropped()
    fails += test_malformed_input()
    fails += test_unknown_cord_keys_ignored()
    fails += test_payment_alternatives()
    print()
    if fails:
        print(f"FAILED: {fails} assertion(s)")
        sys.exit(1)
    print("OK — all assertions passed")


if __name__ == "__main__":
    main()
