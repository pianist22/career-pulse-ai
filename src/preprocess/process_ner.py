"""
Dedicated NER Processing Script

This script processes resumes through the NER pipeline and creates
enhanced datasets with extracted entities and features for improved classification.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm

from src.config import load_config
from src.utils.io_helpers import load_table, save_table
from src.preprocess.ner_extraction import extract_entities_from_resume, ResumeNER
from src.utils.ner_helpers import NERAnalyzer, create_entity_features_dataframe, save_entities_to_json


def process_resumes_with_ner(input_file: str, output_dir: str, config: dict = None) -> Dict[str, Any]:
    """
    Process resumes through the NER pipeline.
    
    Args:
        input_file: Path to input resume data
        output_dir: Directory to save outputs
        config: Configuration dictionary
        
    Returns:
        Dictionary with processing results
    """
    if config is None:
        config = load_config()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {input_file}...")
    df = load_table(input_file)
    
    if df.empty:
        print("No data found in input file.")
        return {}
    
    # Initialize NER system
    print("Initializing NER system...")
    ner = ResumeNER()
    
    # Process each resume
    print(f"Processing {len(df)} resumes...")
    entities_list = []
    ner_features_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing resumes"):
        text = row.get('text', '')
        if not text or not text.strip():
            # Handle empty text
            entities = ner._empty_entities()
        else:
            try:
                entities = ner.extract_entities(text)
            except Exception as e:
                print(f"Error processing resume {idx}: {e}")
                entities = ner._empty_entities()
        
        entities_list.append(entities)
        
        # Create features from entities
        features = create_ner_features_from_entities(entities)
        ner_features_list.append(features)
    
    # Create results DataFrame
    results_df = df.copy()
    results_df['entities'] = entities_list
    results_df['ner_features'] = ner_features_list
    
    # Expand NER features into separate columns
    ner_features_df = pd.json_normalize(ner_features_list)
    ner_features_df.index = results_df.index
    
    # Combine with original data
    enhanced_df = pd.concat([results_df, ner_features_df], axis=1)
    
    # Save results
    enhanced_output = output_path / "enhanced_with_ner.parquet"
    enhanced_df.to_parquet(enhanced_output)
    
    # Save entities separately for analysis
    entities_output = output_path / "extracted_entities.json"
    save_entities_to_json(entities_list, entities_output.as_posix())
    
    # Save NER features
    ner_features_output = output_path / "ner_features.parquet"
    ner_features_df.to_parquet(ner_features_output)
    
    # Analyze entities
    print("Analyzing extracted entities...")
    analyzer = NERAnalyzer()
    analysis = analyzer.analyze_entities_batch(entities_list)
    
    # Save analysis
    analysis_output = output_path / "ner_analysis.json"
    with open(analysis_output, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    
    # Create summary report
    summary = create_ner_summary(analysis, len(df))
    summary_output = output_path / "ner_summary.txt"
    with open(summary_output, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Processing complete!")
    print(f"Enhanced dataset: {enhanced_output}")
    print(f"Entities: {entities_output}")
    print(f"NER features: {ner_features_output}")
    print(f"Analysis: {analysis_output}")
    print(f"Summary: {summary_output}")
    
    return {
        'enhanced_dataset': enhanced_output,
        'entities': entities_output,
        'ner_features': ner_features_output,
        'analysis': analysis_output,
        'summary': summary_output,
        'total_processed': len(df),
        'analysis_results': analysis
    }


def create_ner_features_from_entities(entities: Dict[str, Any]) -> Dict[str, Any]:
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
    features['total_languages'] = len(entities.get('languages', []))
    
    # Contact completeness
    contact_info = entities.get('contact_info', {})
    features['has_email'] = 1 if contact_info.get('email') else 0
    features['has_phone'] = 1 if contact_info.get('phone') else 0
    features['has_linkedin'] = 1 if contact_info.get('linkedin') else 0
    features['has_github'] = 1 if contact_info.get('github') else 0
    features['contact_completeness'] = sum([
        features['has_email'],
        features['has_phone'],
        features['has_linkedin'],
        features['has_github']
    ]) / 4.0
    
    # Skill categories
    skills = entities.get('skills', [])
    skill_categories = {}
    for skill in skills:
        category = skill.get('category', 'unknown')
        skill_categories[category] = skill_categories.get(category, 0) + 1
    
    # Add skill category features
    for category, count in skill_categories.items():
        features[f'skills_{category}'] = count
    
    # High confidence skills
    high_conf_skills = [s for s in skills if s.get('confidence', 0) > 0.7]
    features['high_confidence_skills'] = len(high_conf_skills)
    features['avg_skill_confidence'] = np.mean([s.get('confidence', 0) for s in skills]) if skills else 0
    
    # Text quality metrics
    metadata = entities.get('metadata', {})
    features['text_length'] = metadata.get('text_length', 0)
    features['word_count'] = metadata.get('word_count', 0)
    features['sentence_count'] = metadata.get('sentence_count', 0)
    features['entity_count'] = metadata.get('entity_count', 0)
    
    # Calculate densities
    word_count = features['word_count']
    if word_count > 0:
        features['entity_density'] = features['entity_count'] / word_count
        features['skill_density'] = features['total_skills'] / word_count
        features['experience_density'] = features['total_experience'] / word_count
    else:
        features['entity_density'] = 0
        features['skill_density'] = 0
        features['experience_density'] = 0
    
    # Experience quality metrics
    experience = entities.get('experience', [])
    if experience:
        features['avg_experience_length'] = np.mean([len(exp.get('context', '')) for exp in experience])
        features['has_job_titles'] = sum(1 for exp in experience if exp.get('job_title'))
        features['has_companies'] = sum(1 for exp in experience if exp.get('company'))
    else:
        features['avg_experience_length'] = 0
        features['has_job_titles'] = 0
        features['has_companies'] = 0
    
    # Education quality metrics
    education = entities.get('education', [])
    if education:
        features['has_degrees'] = sum(1 for edu in education if edu.get('degree'))
        features['has_universities'] = sum(1 for edu in education if edu.get('university'))
        features['has_fields'] = sum(1 for edu in education if edu.get('field'))
    else:
        features['has_degrees'] = 0
        features['has_universities'] = 0
        features['has_fields'] = 0
    
    return features


def create_ner_summary(analysis: Dict[str, Any], total_resumes: int) -> str:
    """
    Create a human-readable summary of NER analysis.
    
    Args:
        analysis: NER analysis results
        total_resumes: Total number of resumes processed
        
    Returns:
        Formatted summary string
    """
    summary_parts = []
    
    summary_parts.append("=" * 60)
    summary_parts.append("NER PROCESSING SUMMARY")
    summary_parts.append("=" * 60)
    summary_parts.append(f"Total Resumes Processed: {total_resumes}")
    summary_parts.append("")
    
    # Skill analysis
    skill_analysis = analysis.get('skill_analysis', {})
    summary_parts.append("SKILL ANALYSIS")
    summary_parts.append("-" * 20)
    summary_parts.append(f"Total Unique Skills: {skill_analysis.get('total_unique_skills', 0)}")
    summary_parts.append(f"Average Skills per Resume: {skill_analysis.get('avg_skills_per_resume', 0):.2f}")
    summary_parts.append(f"High Confidence Skills: {len(skill_analysis.get('high_confidence_skills', []))}")
    
    if skill_analysis.get('most_common_skills'):
        summary_parts.append("\nTop 10 Most Common Skills:")
        for skill, count in skill_analysis['most_common_skills'][:10]:
            summary_parts.append(f"  - {skill}: {count}")
    
    # Company analysis
    company_analysis = analysis.get('company_analysis', {})
    summary_parts.append(f"\nCOMPANY ANALYSIS")
    summary_parts.append("-" * 20)
    summary_parts.append(f"Total Unique Companies: {company_analysis.get('total_unique_companies', 0)}")
    summary_parts.append(f"Average Companies per Resume: {company_analysis.get('avg_companies_per_resume', 0):.2f}")
    
    if company_analysis.get('most_common_companies'):
        summary_parts.append("\nTop 10 Most Common Companies:")
        for company, count in company_analysis['most_common_companies'][:10]:
            summary_parts.append(f"  - {company}: {count}")
    
    # Education analysis
    education_analysis = analysis.get('education_analysis', {})
    summary_parts.append(f"\nEDUCATION ANALYSIS")
    summary_parts.append("-" * 20)
    summary_parts.append(f"Total Unique Degrees: {education_analysis.get('total_unique_degrees', 0)}")
    summary_parts.append(f"Average Education per Resume: {education_analysis.get('avg_education_per_resume', 0):.2f}")
    
    if education_analysis.get('most_common_degrees'):
        summary_parts.append("\nTop 5 Most Common Degrees:")
        for degree, count in education_analysis['most_common_degrees'][:5]:
            summary_parts.append(f"  - {degree}: {count}")
    
    # Contact analysis
    contact_analysis = analysis.get('contact_analysis', {})
    summary_parts.append(f"\nCONTACT ANALYSIS")
    summary_parts.append("-" * 20)
    summary_parts.append(f"Email Percentage: {contact_analysis.get('email_percentage', 0):.1f}%")
    summary_parts.append(f"Phone Percentage: {contact_analysis.get('phone_percentage', 0):.1f}%")
    summary_parts.append(f"LinkedIn Percentage: {contact_analysis.get('linkedin_percentage', 0):.1f}%")
    summary_parts.append(f"GitHub Percentage: {contact_analysis.get('github_percentage', 0):.1f}%")
    summary_parts.append(f"Average Contact Completeness: {contact_analysis.get('avg_contact_completeness', 0):.2f}")
    
    # Quality metrics
    quality_metrics = analysis.get('quality_metrics', {})
    summary_parts.append(f"\nQUALITY METRICS")
    summary_parts.append("-" * 20)
    summary_parts.append(f"Average Text Length: {quality_metrics.get('avg_text_length', 0):.0f} characters")
    summary_parts.append(f"Average Word Count: {quality_metrics.get('avg_word_count', 0):.0f} words")
    summary_parts.append(f"Average Entity Count: {quality_metrics.get('avg_entity_count', 0):.0f} entities")
    summary_parts.append(f"Average Entity Density: {quality_metrics.get('avg_entity_density', 0):.3f}")
    
    summary_parts.append("\n" + "=" * 60)
    
    return "\n".join(summary_parts)


def main():
    """Main function to process resumes with NER."""
    config = load_config()
    
    # Define input and output paths
    input_file = "data/processed/resumeatlas_raw.parquet"
    output_dir = "data/processed/ner_enhanced"
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Input file {input_file} not found. Please run data ingestion first.")
        return
    
    # Process resumes with NER
    results = process_resumes_with_ner(input_file, output_dir, config)
    
    if results:
        print("\nNER processing completed successfully!")
        print(f"Results saved to: {results['enhanced_dataset']}")
    else:
        print("NER processing failed.")


if __name__ == "__main__":
    main()
