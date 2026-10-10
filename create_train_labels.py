#!/usr/bin/env python3
"""
Tạo train_labels.csv cho notebook EfficientNet-B1 + FLAIR.

Yêu cầu:
- Có một file CSV nguồn chứa nhãn MGMT chính thức.
- File nguồn phải có cột ID bệnh nhân và cột nhãn MGMT.
- Script KHÔNG tự sinh/đoán nhãn MGMT.

Ví dụ chạy:
    python create_train_labels.py \
        --input rsna_train_labels.csv \
        --output train_labels.csv

Nếu tên cột nguồn khác:
    python create_train_labels.py \
        --input source.csv \
        --id-col BraTS21ID \
        --label-col MGMT_value \
        --output train_labels.csv

Nếu muốn loại 3 ID thường được loại trong pipeline:
    python create_train_labels.py \
        --input source.csv \
        --output train_labels_582.csv \
        --exclude-bad-ids
"""

import argparse
from pathlib import Path
import pandas as pd


DEFAULT_BAD_IDS = {"00109", "00123", "00709"}

ID_CANDIDATES = [
    "BraTS21ID",
    "BraTS2021ID",
    "patient_id",
    "PatientID",
    "BraTS_ID",
    "id",
]

LABEL_CANDIDATES = [
    "MGMT_value",
    "MGMT",
    "mgmt",
    "label",
    "target",
]


def normalize_brats_id(value) -> str:
    """
    Chuẩn hóa ID về dạng 5 chữ số, ví dụ:
      0 -> 00000
      '109' -> 00109
      'BraTS2021_00109' -> 00109
    """
    s = str(value).strip()

    # Nếu pandas đọc số nguyên thành '109.0'
    if s.endswith(".0"):
        s = s[:-2]

    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        raise ValueError(f"Không thể chuẩn hóa patient ID: {value!r}")

    # Giữ 5 chữ số cuối cùng để xử lý dạng BraTS2021_00109
    return digits[-5:].zfill(5)


def detect_column(columns, candidates, kind):
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(
        f"Không tìm thấy cột {kind}. "
        f"Các cột hiện có: {list(columns)}. "
        f"Hãy chỉ định thủ công bằng tham số dòng lệnh."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Tạo và kiểm tra train_labels.csv cho BraTS/MGMT."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Đường dẫn tới CSV nguồn chứa nhãn MGMT chính thức.",
    )
    parser.add_argument(
        "--output",
        default="train_labels.csv",
        help="Tên file CSV đầu ra. Mặc định: train_labels.csv",
    )
    parser.add_argument(
        "--id-col",
        default=None,
        help="Tên cột ID trong CSV nguồn. Nếu bỏ trống, script tự nhận diện.",
    )
    parser.add_argument(
        "--label-col",
        default=None,
        help="Tên cột nhãn MGMT trong CSV nguồn. Nếu bỏ trống, script tự nhận diện.",
    )
    parser.add_argument(
        "--exclude-bad-ids",
        action="store_true",
        help="Loại các ID 00109, 00123, 00709.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file nguồn: {input_path}")

    df = pd.read_csv(input_path)

    id_col = args.id_col or detect_column(
        df.columns, ID_CANDIDATES, "patient ID"
    )
    label_col = args.label_col or detect_column(
        df.columns, LABEL_CANDIDATES, "MGMT"
    )

    print(f"[INFO] ID column   : {id_col}")
    print(f"[INFO] Label column: {label_col}")
    print(f"[INFO] Số dòng nguồn: {len(df)}")

    # Chỉ lấy 2 cột cần thiết
    out = df[[id_col, label_col]].copy()
    out.columns = ["BraTS21ID", "MGMT_value"]

    # Kiểm tra giá trị thiếu
    if out["BraTS21ID"].isna().any():
        bad = out[out["BraTS21ID"].isna()]
        raise ValueError(
            f"Phát hiện {len(bad)} dòng thiếu BraTS21ID."
        )

    if out["MGMT_value"].isna().any():
        bad = out[out["MGMT_value"].isna()]
        raise ValueError(
            f"Phát hiện {len(bad)} dòng thiếu MGMT_value."
        )

    # Chuẩn hóa ID
    out["BraTS21ID"] = out["BraTS21ID"].map(normalize_brats_id)

    # Chuyển nhãn sang số nguyên nếu có thể
    try:
        out["MGMT_value"] = pd.to_numeric(
            out["MGMT_value"], errors="raise"
        ).astype(int)
    except Exception as exc:
        raise ValueError(
            "MGMT_value phải là nhãn nhị phân 0/1."
        ) from exc

    # Kiểm tra MGMT_value ∈ {0,1}
    invalid = out[~out["MGMT_value"].isin([0, 1])]
    if not invalid.empty:
        raise ValueError(
            "Phát hiện MGMT_value không thuộc {0,1}:\n"
            + invalid.to_string(index=False)
        )

    # Kiểm tra ID trùng
    duplicated = out[out["BraTS21ID"].duplicated(keep=False)]
    if not duplicated.empty:
        raise ValueError(
            "Phát hiện BraTS21ID bị trùng:\n"
            + duplicated.sort_values("BraTS21ID").to_string(index=False)
        )

    # Loại các ID không dùng nếu được yêu cầu
    if args.exclude_bad_ids:
        before = len(out)
        out = out[~out["BraTS21ID"].isin(DEFAULT_BAD_IDS)].copy()
        removed = before - len(out)
        print(
            f"[INFO] Đã loại {removed} patient thuộc "
            f"{sorted(DEFAULT_BAD_IDS)}"
        )

    # Sắp xếp ID để dễ kiểm tra
    out = out.sort_values("BraTS21ID").reset_index(drop=True)

    # Ghi CSV, ép ID thành chuỗi 5 chữ số
    out.to_csv(output_path, index=False)

    # Thống kê
    counts = out["MGMT_value"].value_counts().sort_index()

    print("\n========== KIỂM TRA ==========")
    print(f"Số patient       : {len(out)}")
    print(f"Số ID duy nhất   : {out['BraTS21ID'].nunique()}")
    print("Các MGMT_value    :", sorted(out["MGMT_value"].unique().tolist()))
    print("\nPhân bố nhãn:")
    for label in [0, 1]:
        n = int(counts.get(label, 0))
        pct = 100.0 * n / len(out) if len(out) else 0.0
        print(f"  MGMT={label}: {n} ({pct:.2f}%)")

    assert out["BraTS21ID"].nunique() == len(out)
    assert out["MGMT_value"].isin([0, 1]).all()

    print(f"\n[OK] Đã tạo: {output_path.resolve()}")


if __name__ == "__main__":
    main()
