from pathlib import Path
import pandas as pd

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def save_table(df: pd.DataFrame, out_path: str) -> None:
    p = Path(out_path)
    ensure_dir(p.parent.as_posix())
    if p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
    else:
        df.to_parquet(p, index=False)

def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)
