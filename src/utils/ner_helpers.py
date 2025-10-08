"""
NER Helper Utilities

This module provides utility functions for processing and analyzing
extracted entities from resumes, including feature engineering,
entity analysis, and data visualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import json
from pathlib import Path


class NERAnalyzer:
    """
    Analyzer for Named Entity Recognition results.
    
    Provides methods for analyzing extracted entities across
    multiple resumes, generating insights, and creating features.
    """
    
    def __init__(self):
        """Initialize the NER analyzer."""
        self.entity_stats = defaultdict(list)
        self.skill_frequency = Counter()
        self.company_frequency = Counter()
        self.location_frequency = Counter()
    
    def analyze_entities_batch(self, entities_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a batch of extracted entities.
        
        Args:
            entities_list: List of entity dictionaries from multiple resumes
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            'total_resumes': len(entities_list),
            'skill_analysis': self._analyze_skills(entities_list),
            'company_analysis': self._analyze_companies(entities_list),
            'location_analysis': self._analyze_locations(entities_list),
            'education_analysis': self._analyze_education(entities_list),
            'experience_analysis': self._analyze_experience(entities_list),
            'contact_analysis': self._analyze_contact_info(entities_list),
            'quality_metrics': self._analyze_quality_metrics(entities_list)
        }
        
        return analysis
    
    def _analyze_skills(self, entities_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze skill patterns across resumes."""
        all_skills = []
        skill_categories = defaultdict(int)
        high_confidence_skills = []
        
        for entities in entities_list:
            if entities.get('skills'):
                for skill in entities['skills']:
                    all_skills.append(skill['skill'])
                    skill_categories[skill['category']] += 1
                    if skill['confidence'] > 0.7:
                        high_confidence_skills.append(skill['skill'])
        
        skill_counter = Counter(all_skills)
        high_conf_counter = Counter(high_confidence_skills)
        
        return {
            'total_unique_skills': len(skill_counter),
            'most_common_skills': skill_counter.most_common(20),
            'high_confidence_skills': high_conf_counter.most_common(20),
            'skill_categories': dict(skill_categories),
            'avg_skills_per_resume': len(all_skills) / len(entities_list) if entities_list else 0
        }
    
    def _analyze_companies(self, entities_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze company patterns across resumes."""
        all_companies = []
        
        for entities in entities_list:
            if entities.get('companies'):
                all_companies.extend(entities['companies'])
        
        company_counter = Counter(all_companies)
        
        return {
            'total_unique_companies': len(company_counter),
            'most_common_companies': company_counter.most_common(20),
            'avg_companies_per_resume': len(all_companies) / len(entities_list) if entities_list else 0
        }
    
    def _analyze_locations(self, entities_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze location patterns across resumes."""
        all_locations = []
        
        for entities in entities_list:
            if entities.get('locations'):
                all_locations.extend(entities['locations'])
        
        location_counter = Counter(all_locations)
        
        return {
            'total_unique_locations': len(location_counter),
            'most_common_locations': location_counter.most_common(20),
            'avg_locations_per_resume': len(all_locations) / len(entities_list) if entities_list else 0
        }
    
    def _analyze_education(self, entities_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze education patterns across resumes."""
        all_degrees = []
        all_universities = []
        all_fields = []
        
        for entities in entities_list:
            if entities.get('education'):
                for edu in entities['education']:
                    if edu.get('degree'):
                        all_degrees.append(edu['degree'])
                    if edu.get('university'):
                        all_universities.append(edu['university'])
                    if edu.get('field'):
                        all_fields.append(edu['field'])
        
        degree_counter = Counter(all_degrees)
        university_counter = Counter(all_universities)
        field_counter = Counter(all_fields)
        
        return {
            'total_unique_degrees': len(degree_counter),
            'most_common_degrees': degree_counter.most_common(10),
            'most_common_universities': university_counter.most_common(10),
            'most_common_fields': field_counter.most_common(10),
            'avg_education_per_resume': len(all_degrees) / len(entities_list) if entities_list else 0
        }
    
    def _analyze_experience(self, entities_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze experience patterns across resumes."""
        all_job_titles = []
        all_companies = []
        experience_years = []
        
        for entities in entities_list:
            if entities.get('experience'):
                for exp in entities['experience']:
                    if exp.get('job_title'):
                        all_job_titles.append(exp['job_title'])
                    if exp.get('company'):
                        all_companies.append(exp['company'])
        
        job_title_counter = Counter(all_job_titles)
        company_counter = Counter(all_companies)
        
        return {
            'total_unique_job_titles': len(job_title_counter),
            'most_common_job_titles': job_title_counter.most_common(20),
            'most_common_companies': company_counter.most_common(20),
            'avg_experience_per_resume': len(all_job_titles) / len(entities_list) if entities_list else 0
        }
    
    def _analyze_contact_info(self, entities_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze contact information completeness."""
        contact_completeness = []
        email_count = 0
        phone_count = 0
        linkedin_count = 0
        github_count = 0
        
        for entities in entities_list:
            contact_info = entities.get('contact_info', {})
            
            # Count contact methods
            if contact_info.get('email'):
                email_count += 1
            if contact_info.get('phone'):
                phone_count += 1
            if contact_info.get('linkedin'):
                linkedin_count += 1
            if contact_info.get('github'):
                github_count += 1
            
            # Calculate completeness score
            completeness = sum([
                1 if contact_info.get('email') else 0,
                1 if contact_info.get('phone') else 0,
                1 if contact_info.get('linkedin') else 0,
                1 if contact_info.get('github') else 0
            ]) / 4.0
            
            contact_completeness.append(completeness)
        
        return {
            'email_percentage': (email_count / len(entities_list)) * 100 if entities_list else 0,
            'phone_percentage': (phone_count / len(entities_list)) * 100 if entities_list else 0,
            'linkedin_percentage': (linkedin_count / len(entities_list)) * 100 if entities_list else 0,
            'github_percentage': (github_count / len(entities_list)) * 100 if entities_list else 0,
            'avg_contact_completeness': np.mean(contact_completeness) if contact_completeness else 0
        }
    
    def _analyze_quality_metrics(self, entities_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quality metrics across resumes."""
        text_lengths = []
        word_counts = []
        entity_counts = []
        entity_densities = []
        
        for entities in entities_list:
            metadata = entities.get('metadata', {})
            text_lengths.append(metadata.get('text_length', 0))
            word_counts.append(metadata.get('word_count', 0))
            entity_counts.append(metadata.get('entity_count', 0))
            
            # Calculate entity density
            word_count = metadata.get('word_count', 1)
            entity_count = metadata.get('entity_count', 0)
            entity_densities.append(entity_count / word_count)
        
        return {
            'avg_text_length': np.mean(text_lengths) if text_lengths else 0,
            'avg_word_count': np.mean(word_counts) if word_counts else 0,
            'avg_entity_count': np.mean(entity_counts) if entity_counts else 0,
            'avg_entity_density': np.mean(entity_densities) if entity_densities else 0,
            'text_length_std': np.std(text_lengths) if text_lengths else 0,
            'word_count_std': np.std(word_counts) if word_counts else 0
        }


def create_entity_features_dataframe(entities_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a DataFrame with entity-based features for machine learning.
    
    Args:
        entities_list: List of entity dictionaries
        
    Returns:
        DataFrame with entity features
    """
    features_list = []
    
    for i, entities in enumerate(entities_list):
        features = {
            'resume_id': i,
            'text_length': entities.get('metadata', {}).get('text_length', 0),
            'word_count': entities.get('metadata', {}).get('word_count', 0),
            'sentence_count': entities.get('metadata', {}).get('sentence_count', 0),
            'entity_count': entities.get('metadata', {}).get('entity_count', 0),
        }
        
        # Skill features
        skills = entities.get('skills', [])
        features['total_skills'] = len(skills)
        features['high_confidence_skills'] = len([s for s in skills if s.get('confidence', 0) > 0.7])
        
        # Skill category counts
        skill_categories = defaultdict(int)
        for skill in skills:
            skill_categories[skill.get('category', 'unknown')] += 1
        
        for category, count in skill_categories.items():
            features[f'skills_{category}'] = count
        
        # Experience features
        experience = entities.get('experience', [])
        features['experience_count'] = len(experience)
        
        # Education features
        education = entities.get('education', [])
        features['education_count'] = len(education)
        
        # Contact features
        contact_info = entities.get('contact_info', {})
        features['has_email'] = 1 if contact_info.get('email') else 0
        features['has_phone'] = 1 if contact_info.get('phone') else 0
        features['has_linkedin'] = 1 if contact_info.get('linkedin') else 0
        features['has_github'] = 1 if contact_info.get('github') else 0
        
        # Project features
        projects = entities.get('projects', [])
        features['project_count'] = len(projects)
        
        # Certification features
        certifications = entities.get('certifications', [])
        features['certification_count'] = len(certifications)
        
        # Language features
        languages = entities.get('languages', [])
        features['language_count'] = len(languages)
        
        # Company features
        companies = entities.get('companies', [])
        features['company_count'] = len(companies)
        
        # Location features
        locations = entities.get('locations', [])
        features['location_count'] = len(locations)
        
        # Calculate derived features
        word_count = features['word_count']
        if word_count > 0:
            features['entity_density'] = features['entity_count'] / word_count
            features['skill_density'] = features['total_skills'] / word_count
        else:
            features['entity_density'] = 0
            features['skill_density'] = 0
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)


def save_entities_to_json(entities: Dict[str, Any], filepath: str):
    """
    Save extracted entities to JSON file.
    
    Args:
        entities: Entity dictionary
        filepath: Path to save JSON file
    """
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Recursively convert numpy types
    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [clean_dict(item) for item in d]
        else:
            return convert_numpy(d)
    
    cleaned_entities = clean_dict(entities)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(cleaned_entities, f, indent=2, ensure_ascii=False)


def load_entities_from_json(filepath: str) -> Dict[str, Any]:
    """
    Load extracted entities from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Entity dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_entity_summary(entities: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of extracted entities.
    
    Args:
        entities: Entity dictionary
        
    Returns:
        Formatted summary string
    """
    summary_parts = []
    
    # Personal info
    personal_info = entities.get('personal_info', {})
    if personal_info.get('name'):
        summary_parts.append(f"Name: {personal_info['name']}")
    if personal_info.get('job_titles'):
        summary_parts.append(f"Job Titles: {', '.join(personal_info['job_titles'])}")
    
    # Contact info
    contact_info = entities.get('contact_info', {})
    if contact_info.get('email'):
        summary_parts.append(f"Email: {contact_info['email']}")
    if contact_info.get('phone'):
        summary_parts.append(f"Phone: {contact_info['phone']}")
    
    # Skills
    skills = entities.get('skills', [])
    if skills:
        top_skills = [skill['skill'] for skill in skills[:5]]
        summary_parts.append(f"Top Skills: {', '.join(top_skills)}")
    
    # Experience
    experience = entities.get('experience', [])
    if experience:
        summary_parts.append(f"Experience: {len(experience)} positions")
    
    # Education
    education = entities.get('education', [])
    if education:
        summary_parts.append(f"Education: {len(education)} degrees")
    
    # Companies
    companies = entities.get('companies', [])
    if companies:
        summary_parts.append(f"Companies: {', '.join(companies[:3])}")
    
    return "\n".join(summary_parts)


def main():
    """Example usage of NER helper utilities."""
    # Sample entities for testing
    sample_entities = {
        'personal_info': {'name': 'John Smith', 'job_titles': ['Software Engineer']},
        'skills': [
            {'skill': 'Python', 'category': 'programming', 'confidence': 0.9},
            {'skill': 'Django', 'category': 'framework', 'confidence': 0.8},
            {'skill': 'AWS', 'category': 'cloud', 'confidence': 0.7}
        ],
        'experience': [
            {'job_title': 'Senior Software Engineer', 'company': 'Google', 'duration': '2020-2023'},
            {'job_title': 'Software Developer', 'company': 'Microsoft', 'duration': '2018-2020'}
        ],
        'education': [
            {'degree': 'Bachelor of Science', 'university': 'Stanford University', 'field': 'Computer Science'}
        ],
        'contact_info': {'email': 'john@example.com', 'phone': '(555) 123-4567'},
        'companies': ['Google', 'Microsoft'],
        'locations': ['San Francisco', 'California'],
        'metadata': {'text_length': 1500, 'word_count': 250, 'sentence_count': 15, 'entity_count': 25}
    }
    
    # Create summary
    summary = create_entity_summary(sample_entities)
    print("=== ENTITY SUMMARY ===")
    print(summary)
    
    # Create features DataFrame
    features_df = create_entity_features_dataframe([sample_entities])
    print("\n=== ENTITY FEATURES ===")
    print(features_df.head())


if __name__ == "__main__":
    main()
