from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import load_config
from src.utils.io_helpers import load_table, save_table
from src.utils.text_clean import normalize_text
from src.utils.pii import redact_pii
from src.ner.entity_extraction import NERProcessor

def clean_text(text: str, cfg: dict) -> str:
    t = text or ""
    t = normalize_text(
        t,
        lower_case=cfg["preprocessing"]["lower_case"],
        remove_urls=cfg["preprocessing"]["remove_urls"],
        remove_handles=cfg["preprocessing"]["remove_handles"],
        remove_specials=cfg["preprocessing"]["remove_specials"],
        expand_contr=cfg["preprocessing"]["expand_contractions"],
    )
    if cfg["preprocessing"]["redact_pii"]:
        t = redact_pii(t)
    return t

def truncate_words(text: str, n_words: int) -> str:
    if n_words <= 0:
        return text
    parts = text.split()
    return " ".join(parts[:n_words])

def extract_entities_from_text(text: str, cfg: dict) -> dict:
    """
    Extract entities from text using NER if enabled
    
    Args:
        text: Text to extract entities from
        cfg: Configuration dictionary
        
    Returns:
        Dictionary containing extracted entities or empty dict if NER disabled
    """
    if not cfg.get("ner", {}).get("enabled", False):
        return {}
    
    try:
        model_name = cfg.get("ner", {}).get("model_name", "en_core_web_sm")
        ner_processor = NERProcessor(model_name=model_name)
        entities = ner_processor.extract_entities(text)
        return ner_processor.entities_to_features(entities)
    except Exception as e:
        print(f"Warning: NER processing failed: {e}")
        return {}

def main():
    cfg = load_config()
    processed_dir = Path(cfg["data"]["processed_dir"])
    interim_dir = Path(cfg["data"]["interim_dir"])

    # Load public ResumeAtlas if configured
    frames = []
    resumeatlas_raw = processed_dir / f"resumeatlas_raw.{cfg['data']['export_format']}"
    if resumeatlas_raw.exists():
        df_ra = load_table(resumeatlas_raw.as_posix())
        df_ra["source"] = "resumeatlas"
        frames.append(df_ra[["text", "label", "source"]])

    # Load local ingested files (unlabeled) and assign a placeholder label or keep label=None
    local_raw = Path(cfg["data"]["interim_dir"]) / f"local_raw.{cfg['data']['export_format']}"
    if local_raw.exists():
        df_local = load_table(local_raw.as_posix())
        df_local["label"] = None
        df_local["source"] = "local"
        frames.append(df_local[["text", "label", "source"]])

    assert frames, "No input tables found. Run ingestion first."
    df_all = pd.concat(frames, ignore_index=True)
    df_all["text"] = df_all["text"].fillna("")

    # Clean
    df_all["text_clean"] = df_all["text"].apply(lambda x: clean_text(x, cfg))

    # Extract entities using NER (if enabled)
    if cfg.get("ner", {}).get("enabled", False) and cfg.get("ner", {}).get("save_entity_features", False):
        print("Extracting entities using NER...")
        entity_features = df_all["text_clean"].apply(lambda x: extract_entities_from_text(x, cfg))
        
        # Convert entity features to DataFrame columns
        entity_df = pd.DataFrame(entity_features.tolist())
        if not entity_df.empty:
            # Add entity columns to main dataframe
            for col in entity_df.columns:
                df_all[f"entity_{col}"] = entity_df[col]
            print(f"Added {len(entity_df.columns)} entity feature columns")

    # Truncate for classification efficiency
    n_words = int(cfg["classification"]["truncate_words"])
    df_all["text_trunc"] = df_all["text_clean"].apply(lambda x: truncate_words(x, n_words))

    # Keep classification-ready subset (labeled)
    df_cls = df_all[df_all["label"].notna()].copy()

    # Split
    train_size = cfg["splits"]["train"]
    val_size = cfg["splits"]["val"]
    test_size = cfg["splits"]["test"]
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    df_train, df_tmp = train_test_split(df_cls, test_size=(val_size + test_size), random_state=cfg["project"]["seed"], stratify=df_cls["label"])
    rel_test = test_size / (val_size + test_size)
    df_val, df_test = train_test_split(df_tmp, test_size=rel_test, random_state=cfg["project"]["seed"], stratify=df_tmp["label"])

    # Save
    for name, frame in [("train", df_train), ("val", df_val), ("test", df_test)]:
        out = processed_dir / f"classification_{name}.{cfg['data']['export_format']}"
        frame[["text_trunc", "label"]].rename(columns={"text_trunc": "text"}).reset_index(drop=True)
        save_table(frame[["text_trunc", "label"]].rename(columns={"text_trunc": "text"}), out.as_posix())
        print(f"Saved: {out}")

if __name__ == "__main__":
    main()
