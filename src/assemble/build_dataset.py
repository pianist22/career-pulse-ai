from pathlib import Path
import pandas as pd
from collections import Counter
from src.config import load_config
from src.utils.io_helpers import load_table

def show_stats(df: pd.DataFrame, title: str):
    print(f"\n== {title} ==")
    print("Rows:", len(df))
    labels = df["label"].tolist()
    print("Unique labels:", len(set(labels)))
    print("Top-10 labels:", Counter(labels).most_common(10))
    print("Sample text:", (df["text"].iloc or "")[:400], "...")

def main():
    cfg = load_config()
    base = Path(cfg["data"]["processed_dir"])
    for split in ["train", "val", "test"]:
        p = base / f"classification_{split}.{cfg['data']['export_format']}"
        df = load_table(p.as_posix())
        df = df.rename(columns={"text_trunc": "text"}) if "text_trunc" in df.columns else df
        show_stats(df, split)

if __name__ == "__main__":
    main()
