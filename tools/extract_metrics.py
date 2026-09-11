#!/usr/bin/env python3
"""Extract sklearn-style metrics from executed Jupyter notebooks.

Usage: python tools/extract_metrics.py output/kernel3
"""
import json
import os
import re
import sys
from pathlib import Path
import csv


METRIC_KEYS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "roc_auc",
    "pr_auc",
]


def parse_metrics_from_text(text):
    """Return metrics, confusion matrix, supports, and CPU total time if found."""
    metrics = {}
    cm = None
    support_0 = None
    support_1 = None
    cpu_total_time = None

    # Extract the total from IPython timing output, preserving its unit.
    m = re.search(
        r"\btotal\s*:\s*([0-9]*\.?[0-9]+)\s*(min|ms|us|s)\b",
        text,
        re.IGNORECASE,
    )
    if m:
        cpu_total_time = f"{m.group(1)} {m.group(2)}"

    # Extract key: value lines like "accuracy: 0.9383"
    for key in METRIC_KEYS:
        m = re.search(rf"^{key}\s*:\s*([0-9]*\.?[0-9]+)", text, re.MULTILINE | re.IGNORECASE)
        if m:
            metrics[key] = float(m.group(1))

    # Confusion matrix: look for a bracketed 2x2 block
    m = re.search(r"Confusion matrix:\s*\[?\s*\[?\s*([0-9]+)\s+([0-9]+)\s*\]?\s*\[?\s*([0-9]+)\s+([0-9]+)", text, re.IGNORECASE)
    if m:
        cm = f"[[{m.group(1)} {m.group(2)}]\n [{m.group(3)} {m.group(4)}]]"

    # Classification report: extract support numbers for class 0 and 1
    # Look for lines that start with whitespace+0 or 1 then multiple spaces then support
    rep_lines = re.findall(r"^\s*(0|1)\s+.*?\s+(\d+)\s*$", text, re.MULTILINE)
    if rep_lines:
        for cls, supp in rep_lines:
            if cls == "0":
                support_0 = int(supp)
            elif cls == "1":
                support_1 = int(supp)

    # Some reports include 'support' in a different aligned format; fallback search
    if support_0 is None or support_1 is None:
        m0 = re.search(r"^\s*0\s+.*?\s+(\d+)\s*$", text, re.MULTILINE)
        m1 = re.search(r"^\s*1\s+.*?\s+(\d+)\s*$", text, re.MULTILINE)
        if m0:
            support_0 = int(m0.group(1))
        if m1:
            support_1 = int(m1.group(1))

    return metrics, cm, support_0, support_1, cpu_total_time


def extract_from_notebook(nb_path: Path):
    try:
        raw = nb_path.read_text(encoding="utf8")
        nb = json.loads(raw)
    except Exception as e:
        print(f"Failed to read {nb_path}: {e}")
        return None

    all_texts = []
    cells = nb.get("cells", [])
    for cell in cells:
        outputs = cell.get("outputs", [])
        for out in outputs:
            # outputs may have 'text' or 'data' with 'text/plain'
            text = ""
            if "text" in out:
                if isinstance(out["text"], list):
                    text = "".join(out["text"]) 
                else:
                    text = out["text"]
            elif out.get("output_type") == "execute_result":
                data = out.get("data", {})
                text = data.get("text/plain", "")
                if isinstance(text, list):
                    text = "".join(text)

            if not text:
                continue
            all_texts.append(text)

    cpu_total_time = None
    for text in all_texts:
        _, _, _, _, parsed_cpu_total_time = parse_metrics_from_text(text)
        if parsed_cpu_total_time is not None:
            cpu_total_time = parsed_cpu_total_time

    # Search cells from last to first for the latest printed metrics block.
    for text in reversed(all_texts):
        if "Sklearn binary metrics" in text or re.search(r"^accuracy:\s*[0-9]", text, re.IGNORECASE | re.MULTILINE):
            metrics, cm, s0, s1, _ = parse_metrics_from_text(text)
            return {
                "metrics": metrics,
                "confusion_matrix": cm,
                "support_0": s0,
                "support_1": s1,
                "cpu_total_time": cpu_total_time,
            }

    return None


def parse_filename(nb_path: Path):
    # kernel from parent folder
    kernel = os.path.basename(os.path.dirname(str(nb_path)))
    name = nb_path.stem
    parts = name.split('.')
    model = parts[0] if parts else name
    clazz = None
    # try to find a known radiomics class segment
    for p in parts:
        if re.match(r"^(firstorder|glcm|gldm|glrlm|glszm|ngtdm)$", p, re.IGNORECASE):
            clazz = p
            break
    # fallback: use second or third segment heuristics
    if not clazz and len(parts) >= 3:
        clazz = parts[2]
    if not clazz and len(parts) >= 2:
        clazz = parts[1]

    return clazz, kernel, model


def process_folder(folder: Path):
    nbs = list(folder.rglob("*.ipynb"))
    if not nbs:
        print(f"No notebooks found in {folder}")
        return

    total = 0
    rows_written = 0

    # aggregate rows per directory so a single metrics.csv contains one row per notebook
    rows_by_dir = {}
    header = [
        "class",
        "kernel",
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
        "confusion_matrix",
        "support_0",
        "support_1",
        "cpu_total_time",
    ]

    for nb in nbs:
        total += 1
        result = extract_from_notebook(nb)
        if result is None:
            print(f"No metrics found in {nb}")
            continue

        clazz, kernel, model = parse_filename(nb)
        row = {
            "class": clazz or "",
            "kernel": kernel or "",
            "model": model or "",
            "accuracy": result["metrics"].get("accuracy") if result["metrics"] else None,
            "precision": result["metrics"].get("precision") if result["metrics"] else None,
            "recall": result["metrics"].get("recall") if result["metrics"] else None,
            "f1": result["metrics"].get("f1") if result["metrics"] else None,
            "mcc": result["metrics"].get("mcc") if result["metrics"] else None,
            "roc_auc": result["metrics"].get("roc_auc") if result["metrics"] else None,
            "pr_auc": result["metrics"].get("pr_auc") if result["metrics"] else None,
            "confusion_matrix": result.get("confusion_matrix"),
            "support_0": result.get("support_0"),
            "support_1": result.get("support_1"),
            "cpu_total_time": result.get("cpu_total_time"),
        }

        d = nb.parent
        rows_by_dir.setdefault(d, []).append((nb.name, row))
        rows_written += 1

    # write one metrics.csv per directory with all rows
    for d, items in rows_by_dir.items():
        out_csv = Path(d) / "metrics.csv"
        with open(out_csv, "w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=["notebook"] + header)
            writer.writeheader()
            for nb_name, row in items:
                out_row = {"notebook": nb_name}
                out_row.update(row)
                writer.writerow(out_row)
        print(f"Wrote {out_csv} with {len(items)} rows")

    print(f"Processed {total} notebooks, wrote {rows_written} metrics.csv files.")


def main():
    if len(sys.argv) < 2:
        print("Usage: extract_metrics.py <folder>")
        sys.exit(2)
    folder = Path(sys.argv[1])
    if not folder.exists():
        print(f"Folder not found: {folder}")
        sys.exit(1)
    process_folder(folder)


if __name__ == "__main__":
    main()
