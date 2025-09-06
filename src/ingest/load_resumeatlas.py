from datasets import load_dataset
import pandas as pd
from pathlib import Path
from src.config import load_config
from src.utils.io_helpers import save_table

def load_resumeatlas(split_map=None) -> pd.DataFrame:
    ds = load_dataset("ahmedheakl/resume-atlas")
    # Expecting 'Text' and 'Category' fields based on dataset card
    frames = []
    for name, subset in ds.items():
        df = subset.to_pandas()
        df = df.rename(columns={"Text": "text", "Category": "label"})
        df["source_split"] = name
        frames.append(df[["text", "label", "source_split"]])
    return pd.concat(frames, ignore_index=True)

def main():
    cfg = load_config()
    out_dir = Path(cfg["data"]["processed_dir"])
    out_path = out_dir / f"resumeatlas_raw.{cfg['data']['export_format']}"
    df = load_resumeatlas()
    save_table(df, out_path.as_posix())
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
