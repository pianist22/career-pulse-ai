from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import load_config
from src.utils.io_helpers import load_table, save_table
from src.utils.text_clean import normalize_text
from src.utils.pii import redact_pii
from src.preprocess.ner_extraction import extract_entities_from_resume
from src.utils.ner_helpers import create_entity_features_dataframe, NERAnalyzer

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
    Extract entities from resume text using NER.
    
    Args:
        text: Resume text to process
        cfg: Configuration dictionary
        
    Returns:
        Dictionary containing extracted entities
    """
    try:
        # Extract entities using the NER system
        entities = extract_entities_from_resume(text)
        return entities
    except Exception as e:
        print(f"Error extracting entities: {e}")
        # Return empty entities structure on error
        return {
            'personal_info': {},
            'skills': [],
            'experience': [],
            'education': [],
            'contact_info': {},
            'companies': [],
            'locations': [],
            'dates': [],
            'urls': [],
            'certifications': [],
            'projects': [],
            'languages': [],
            'metadata': {'text_length': len(text), 'word_count': len(text.split()), 'sentence_count': 0, 'entity_count': 0}
        }

def create_ner_features(entities: dict) -> dict:
    """
    Create feature vector from extracted entities.
    
    Args:
        entities: Extracted entities dictionary
        
    Returns:
        Dictionary of NER-based features
    """
    features = {}
    
    # Basic entity counts
    features['total_skills'] = len(entities.get('skills', []))
    features['total_experience'] = len(entities.get('experience', []))
    features['total_education'] = len(entities.get('education', []))
    features['total_companies'] = len(entities.get('companies', []))
    features['total_locations'] = len(entities.get('locations', []))
    features['total_projects'] = len(entities.get('projects', []))
    features['total_certifications'] = len(entities.get('certifications', []))
    
    # Contact completeness
    contact_info = entities.get('contact_info', {})
    features['has_email'] = 1 if contact_info.get('email') else 0
    features['has_phone'] = 1 if contact_info.get('phone') else 0
    features['has_linkedin'] = 1 if contact_info.get('linkedin') else 0
    features['has_github'] = 1 if contact_info.get('github') else 0
    
    # Skill categories
    skills = entities.get('skills', [])
    skill_categories = {}
    for skill in skills:
        category = skill.get('category', 'unknown')
        skill_categories[category] = skill_categories.get(category, 0) + 1
    
    for category, count in skill_categories.items():
        features[f'skills_{category}'] = count
    
    # High confidence skills
    high_conf_skills = [s for s in skills if s.get('confidence', 0) > 0.7]
    features['high_confidence_skills'] = len(high_conf_skills)
    
    # Text quality metrics
    metadata = entities.get('metadata', {})
    features['text_length'] = metadata.get('text_length', 0)
    features['word_count'] = metadata.get('word_count', 0)
    features['entity_count'] = metadata.get('entity_count', 0)
    
    # Calculate densities
    word_count = features['word_count']
    if word_count > 0:
        features['entity_density'] = features['entity_count'] / word_count
        features['skill_density'] = features['total_skills'] / word_count
    else:
        features['entity_density'] = 0
        features['skill_density'] = 0
    
    return features

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

    # NER Processing - Extract entities from clean text
    print("Extracting entities using NER...")
    df_all["entities"] = df_all["text_clean"].apply(lambda x: extract_entities_from_text(x, cfg))
    
    # Create NER-based features
    print("Creating NER features...")
    df_all["ner_features"] = df_all["entities"].apply(create_ner_features)
    
    # Expand NER features into separate columns
    ner_features_df = pd.json_normalize(df_all["ner_features"])
    ner_features_df.index = df_all.index
    
    # Combine original data with NER features
    df_all = pd.concat([df_all, ner_features_df], axis=1)
    
    # Save entities and NER features for analysis
    entities_output = processed_dir / "extracted_entities.json"
    ner_features_output = processed_dir / "ner_features.parquet"
    
    # Save entities (sample for analysis)
    sample_entities = df_all["entities"].iloc[:10].tolist()
    import json
    with open(entities_output, 'w', encoding='utf-8') as f:
        json.dump(sample_entities, f, indent=2, ensure_ascii=False, default=str)
    
    # Save NER features
    ner_features_df.to_parquet(ner_features_output)
    
    print(f"Saved entities sample: {entities_output}")
    print(f"Saved NER features: {ner_features_output}")

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

    # Save classification datasets with NER features
    for name, frame in [("train", df_train), ("val", df_val), ("test", df_test)]:
        out = processed_dir / f"classification_{name}.{cfg['data']['export_format']}"
        
        # Include NER features in the classification dataset
        classification_cols = ["text_trunc", "label"] + [col for col in frame.columns if col.startswith(('total_', 'has_', 'skills_', 'entity_', 'skill_'))]
        available_cols = [col for col in classification_cols if col in frame.columns]
        
        if available_cols:
            save_table(frame[available_cols].rename(columns={"text_trunc": "text"}), out.as_posix())
        else:
            # Fallback to original format
            save_table(frame[["text_trunc", "label"]].rename(columns={"text_trunc": "text"}), out.as_posix())
        
        print(f"Saved: {out}")

if __name__ == "__main__":
    main()
