"""
Named Entity Recognition (NER) Module for Resume Processing

This module provides comprehensive NER capabilities for extracting structured information
from resume text, including skills, companies, locations, education, and experience details.
Uses multiple approaches for high accuracy: spaCy, regex patterns, and custom rules.
"""

import re
import spacy
import nltk
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
import pandas as pd
from pathlib import Path

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ResumeNER:
    """
    High-accuracy Named Entity Recognition for resume processing.
    
    Combines multiple NER approaches:
    1. spaCy's pre-trained models for general entities
    2. Custom regex patterns for resume-specific entities
    3. Domain-specific knowledge bases for skills and technologies
    4. Context-aware entity extraction
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the NER system with spaCy model.
        
        Args:
            model_name: spaCy model to use for NER
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Please install with: python -m spacy download {model_name}")
            raise
        
        # Initialize skill and technology databases
        self._initialize_skill_databases()
        
        # Initialize regex patterns for resume-specific entities
        self._initialize_regex_patterns()
        
        # Initialize context patterns
        self._initialize_context_patterns()
    
    def _initialize_skill_databases(self):
        """Initialize comprehensive skill and technology databases."""
        
        # Programming Languages
        self.programming_languages = {
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c', 'php', 'ruby', 'go',
            'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css', 'scss',
            'sass', 'less', 'xml', 'json', 'yaml', 'bash', 'powershell', 'perl', 'lua',
            'dart', 'clojure', 'haskell', 'erlang', 'elixir', 'f#', 'ocaml', 'prolog'
        }
        
        # Frameworks and Libraries
        self.frameworks = {
            'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring',
            'laravel', 'rails', 'asp.net', 'jquery', 'bootstrap', 'tailwind', 'sass',
            'webpack', 'babel', 'gulp', 'grunt', 'npm', 'yarn', 'pip', 'maven', 'gradle',
            'docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab', 'bitbucket'
        }
        
        # Databases and Technologies
        self.databases = {
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
            'dynamodb', 'oracle', 'sqlite', 'mariadb', 'neo4j', 'couchdb', 'firebase'
        }
        
        # Cloud Platforms
        self.cloud_platforms = {
            'aws', 'azure', 'gcp', 'google cloud', 'amazon web services', 'microsoft azure',
            'heroku', 'digitalocean', 'linode', 'vultr', 'cloudflare'
        }
        
        # Data Science and ML
        self.data_science_tools = {
            'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
            'matplotlib', 'seaborn', 'plotly', 'jupyter', 'anaconda', 'conda',
            'scipy', 'statsmodels', 'xgboost', 'lightgbm', 'catboost'
        }
        
        # Business and Soft Skills
        self.business_skills = {
            'project management', 'agile', 'scrum', 'kanban', 'leadership', 'communication',
            'teamwork', 'problem solving', 'analytical thinking', 'time management',
            'customer service', 'sales', 'marketing', 'business development'
        }
        
        # Combine all skills
        self.all_skills = (
            self.programming_languages | self.frameworks | self.databases | 
            self.cloud_platforms | self.data_science_tools | self.business_skills
        )
    
    def _initialize_regex_patterns(self):
        """Initialize regex patterns for resume-specific entity extraction."""
        
        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Phone number patterns (various formats)
        self.phone_patterns = [
            re.compile(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}'),
            re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
            re.compile(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}')
        ]
        
        # URL patterns
        self.url_pattern = re.compile(
            r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
        )
        
        # LinkedIn profile pattern
        self.linkedin_pattern = re.compile(
            r'(?:linkedin\.com/in/|linkedin\.com/pub/)[\w\-]+/?'
        )
        
        # GitHub profile pattern
        self.github_pattern = re.compile(
            r'(?:github\.com/)[\w\-]+/?'
        )
        
        # Date patterns (various formats)
        self.date_patterns = [
            re.compile(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b', re.IGNORECASE),
            re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
            re.compile(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'),
            re.compile(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b', re.IGNORECASE)
        ]
        
        # Experience duration patterns
        self.duration_patterns = [
            re.compile(r'\b\d+\+?\s*(?:years?|yrs?)\b', re.IGNORECASE),
            re.compile(r'\b\d+\+?\s*(?:months?|mos?)\b', re.IGNORECASE),
            re.compile(r'\b\d+\+?\s*(?:weeks?|wks?)\b', re.IGNORECASE)
        ]
        
        # Education degree patterns
        self.degree_patterns = [
            re.compile(r'\b(?:bachelor|master|phd|doctorate|associate|diploma|certificate)\b', re.IGNORECASE),
            re.compile(r'\b(?:b\.?s\.?|b\.?a\.?|m\.?s\.?|m\.?a\.?|m\.?b\.?a\.?|ph\.?d\.?|d\.?ph\.?)\b', re.IGNORECASE)
        ]
    
    def _initialize_context_patterns(self):
        """Initialize context-aware patterns for better entity extraction."""
        
        # Job title context patterns
        self.job_title_contexts = [
            r'(?:worked as|position|role|title|job|employed as)\s*:?\s*([^,\n]+)',
            r'(?:senior|junior|lead|principal|staff|associate|director|manager|engineer|developer|analyst|consultant|specialist|coordinator|administrator|architect|scientist|researcher|designer|coordinator)\s+([^,\n]+)',
        ]
        
        # Company context patterns
        self.company_contexts = [
            r'(?:at|@|company|organization|employer|firm|corporation)\s*:?\s*([^,\n]+)',
            r'(?:worked at|employed at|interned at|consulted for)\s+([^,\n]+)',
        ]
        
        # Skills context patterns
        self.skills_contexts = [
            r'(?:skills?|technologies?|tools?|languages?|frameworks?|expertise|proficient in|experienced with|knowledge of)\s*:?\s*([^.\n]+)',
            r'(?:programming|technical|software|development|data|analytics|cloud|database|web|mobile|frontend|backend|full.?stack)\s+(?:skills?|technologies?|tools?)\s*:?\s*([^.\n]+)',
        ]
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract all entities from resume text with high accuracy.
        
        Args:
            text: Resume text to process
            
        Returns:
            Dictionary containing all extracted entities
        """
        if not text or not text.strip():
            return self._empty_entities()
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract different types of entities
        entities = {
            'personal_info': self._extract_personal_info(text, doc),
            'skills': self._extract_skills(text, doc),
            'experience': self._extract_experience(text, doc),
            'education': self._extract_education(text, doc),
            'contact_info': self._extract_contact_info(text),
            'companies': self._extract_companies(text, doc),
            'locations': self._extract_locations(text, doc),
            'dates': self._extract_dates(text),
            'urls': self._extract_urls(text),
            'certifications': self._extract_certifications(text, doc),
            'projects': self._extract_projects(text, doc),
            'languages': self._extract_languages(text, doc)
        }
        
        # Add metadata
        entities['metadata'] = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(list(doc.sents)),
            'entity_count': sum(len(v) if isinstance(v, list) else 1 for v in entities.values() if v)
        }
        
        return entities
    
    def _empty_entities(self) -> Dict[str, Any]:
        """Return empty entity structure."""
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
            'metadata': {'text_length': 0, 'word_count': 0, 'sentence_count': 0, 'entity_count': 0}
        }
    
    def _extract_personal_info(self, text: str, doc) -> Dict[str, str]:
        """Extract personal information like name, title."""
        personal_info = {}
        
        # Extract names using spaCy NER
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.split()) >= 2:
                personal_info['name'] = ent.text
                break
        
        # Extract job titles from context
        job_titles = []
        for pattern in self.job_title_contexts:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                title = match.group(1).strip()
                if len(title) > 3 and len(title) < 50:
                    job_titles.append(title)
        
        if job_titles:
            personal_info['job_titles'] = list(set(job_titles))
        
        return personal_info
    
    def _extract_skills(self, text: str, doc) -> List[Dict[str, Any]]:
        """Extract technical and soft skills with confidence scores."""
        skills = []
        text_lower = text.lower()
        
        # Extract skills from predefined databases
        for skill in self.all_skills:
            if skill.lower() in text_lower:
                # Calculate confidence based on context
                confidence = self._calculate_skill_confidence(skill, text)
                if confidence > 0.3:  # Threshold for inclusion
                    skills.append({
                        'skill': skill,
                        'category': self._categorize_skill(skill),
                        'confidence': confidence,
                        'context': self._get_skill_context(skill, text)
                    })
        
        # Extract skills from spaCy entities
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT"] and len(ent.text) > 2:
                if any(keyword in ent.text.lower() for keyword in ['software', 'tool', 'technology', 'framework']):
                    skills.append({
                        'skill': ent.text,
                        'category': 'technology',
                        'confidence': 0.7,
                        'context': 'spacy_ner'
                    })
        
        # Extract skills from context patterns
        for pattern in self.skills_contexts:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                skills_text = match.group(1)
                # Split and clean skills
                skill_list = [s.strip() for s in re.split(r'[,;|&]', skills_text)]
                for skill in skill_list:
                    if len(skill) > 2 and len(skill) < 30:
                        skills.append({
                            'skill': skill,
                            'category': 'general',
                            'confidence': 0.6,
                            'context': 'pattern_match'
                        })
        
        # Remove duplicates and sort by confidence
        unique_skills = {}
        for skill in skills:
            key = skill['skill'].lower()
            if key not in unique_skills or skill['confidence'] > unique_skills[key]['confidence']:
                unique_skills[key] = skill
        
        return sorted(unique_skills.values(), key=lambda x: x['confidence'], reverse=True)
    
    def _calculate_skill_confidence(self, skill: str, text: str) -> float:
        """Calculate confidence score for a skill based on context."""
        confidence = 0.5  # Base confidence
        
        # Check for skill-related keywords nearby
        skill_pos = text.lower().find(skill.lower())
        if skill_pos != -1:
            # Look for context keywords
            context_start = max(0, skill_pos - 50)
            context_end = min(len(text), skill_pos + len(skill) + 50)
            context = text[context_start:context_end].lower()
            
            skill_keywords = ['experience', 'proficient', 'skilled', 'expert', 'knowledge', 'familiar', 'used', 'worked with']
            if any(keyword in context for keyword in skill_keywords):
                confidence += 0.3
            
            # Check for years of experience
            if re.search(r'\d+\+?\s*(?:years?|yrs?)', context):
                confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _categorize_skill(self, skill: str) -> str:
        """Categorize a skill into predefined categories."""
        skill_lower = skill.lower()
        
        if skill_lower in self.programming_languages:
            return 'programming'
        elif skill_lower in self.frameworks:
            return 'framework'
        elif skill_lower in self.databases:
            return 'database'
        elif skill_lower in self.cloud_platforms:
            return 'cloud'
        elif skill_lower in self.data_science_tools:
            return 'data_science'
        elif skill_lower in self.business_skills:
            return 'business'
        else:
            return 'general'
    
    def _get_skill_context(self, skill: str, text: str) -> str:
        """Get context around a skill mention."""
        skill_pos = text.lower().find(skill.lower())
        if skill_pos != -1:
            context_start = max(0, skill_pos - 30)
            context_end = min(len(text), skill_pos + len(skill) + 30)
            return text[context_start:context_end].strip()
        return ""
    
    def _extract_experience(self, text: str, doc) -> List[Dict[str, Any]]:
        """Extract work experience information."""
        experience = []
        
        # Look for experience patterns
        exp_patterns = [
            r'(\d{4})\s*[-–]\s*(\d{4}|\bpresent\b|\bcurrent\b)',
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\s*[-–]\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}',
        ]
        
        for pattern in exp_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract surrounding context for job details
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]
                
                # Try to extract job title and company from context
                job_title = self._extract_job_title_from_context(context)
                company = self._extract_company_from_context(context)
                
                experience.append({
                    'duration': match.group(),
                    'job_title': job_title,
                    'company': company,
                    'context': context.strip()
                })
        
        return experience
    
    def _extract_job_title_from_context(self, context: str) -> str:
        """Extract job title from experience context."""
        # Look for common job title patterns
        title_patterns = [
            r'(?:senior|junior|lead|principal|staff|associate|director|manager|engineer|developer|analyst|consultant|specialist|coordinator|administrator|architect|scientist|researcher|designer)\s+([^,\n]+)',
            r'([^,\n]+)\s+(?:engineer|developer|analyst|consultant|specialist|manager|director|architect)',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_company_from_context(self, context: str) -> str:
        """Extract company name from experience context."""
        # Look for company indicators
        company_patterns = [
            r'(?:at|@)\s+([A-Z][^,\n]+)',
            r'(?:company|corporation|inc|llc|ltd|group|systems|solutions|technologies|consulting)\s+([^,\n]+)',
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_education(self, text: str, doc) -> List[Dict[str, Any]]:
        """Extract education information."""
        education = []
        
        # Look for degree patterns
        for pattern in self.degree_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                # Try to extract university and field of study
                university = self._extract_university_from_context(context)
                field = self._extract_field_from_context(context)
                
                education.append({
                    'degree': match.group(),
                    'university': university,
                    'field': field,
                    'context': context.strip()
                })
        
        return education
    
    def _extract_university_from_context(self, context: str) -> str:
        """Extract university name from education context."""
        # Look for university indicators
        uni_patterns = [
            r'(?:university|college|institute|school)\s+of\s+([^,\n]+)',
            r'([A-Z][^,\n]*(?:university|college|institute|school))',
        ]
        
        for pattern in uni_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_field_from_context(self, context: str) -> str:
        """Extract field of study from education context."""
        # Look for field indicators
        field_patterns = [
            r'(?:in|of|majoring in|studied)\s+([^,\n]+)',
            r'(?:computer science|engineering|business|mathematics|physics|chemistry|biology|economics|psychology|sociology|political science|history|literature|art|music)',
        ]
        
        for pattern in field_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information."""
        contact_info = {}
        
        # Extract emails
        emails = self.email_pattern.findall(text)
        if emails:
            contact_info['email'] = emails[0]
        
        # Extract phone numbers
        phones = []
        for pattern in self.phone_patterns:
            phones.extend(pattern.findall(text))
        if phones:
            contact_info['phone'] = phones[0]
        
        # Extract LinkedIn
        linkedin = self.linkedin_pattern.findall(text)
        if linkedin:
            contact_info['linkedin'] = linkedin[0]
        
        # Extract GitHub
        github = self.github_pattern.findall(text)
        if github:
            contact_info['github'] = github[0]
        
        return contact_info
    
    def _extract_companies(self, text: str, doc) -> List[str]:
        """Extract company names."""
        companies = []
        
        # Extract from spaCy NER
        for ent in doc.ents:
            if ent.label_ == "ORG":
                companies.append(ent.text)
        
        # Extract from context patterns
        for pattern in self.company_contexts:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                company = match.group(1).strip()
                if len(company) > 2 and len(company) < 50:
                    companies.append(company)
        
        return list(set(companies))
    
    def _extract_locations(self, text: str, doc) -> List[str]:
        """Extract location information."""
        locations = []
        
        # Extract from spaCy NER
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                locations.append(ent.text)
        
        return list(set(locations))
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text."""
        dates = []
        
        for pattern in self.date_patterns:
            dates.extend(pattern.findall(text))
        
        return list(set(dates))
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        urls = []
        urls.extend(self.url_pattern.findall(text))
        urls.extend(self.linkedin_pattern.findall(text))
        urls.extend(self.github_pattern.findall(text))
        
        return list(set(urls))
    
    def _extract_certifications(self, text: str, doc) -> List[Dict[str, str]]:
        """Extract certifications and licenses."""
        certifications = []
        
        # Look for certification patterns
        cert_patterns = [
            r'(?:certified|certification|license|licensed)\s+in\s+([^,\n]+)',
            r'([A-Z]{2,}\s+[A-Z]{2,})',  # Common cert patterns like "AWS Certified"
            r'(?:pmp|cissp|aws|azure|gcp|scrum|agile|itil|prince2)',
        ]
        
        for pattern in cert_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                certifications.append({
                    'certification': match.group(),
                    'context': self._get_context_around_match(match, text)
                })
        
        return certifications
    
    def _extract_projects(self, text: str, doc) -> List[Dict[str, str]]:
        """Extract project information."""
        projects = []
        
        # Look for project patterns
        project_patterns = [
            r'(?:project|developed|built|created|designed)\s*:?\s*([^.\n]+)',
            r'(?:github|repository|repo)\s*:?\s*([^,\n]+)',
        ]
        
        for pattern in project_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                projects.append({
                    'project': match.group(1).strip(),
                    'context': self._get_context_around_match(match, text)
                })
        
        return projects
    
    def _extract_languages(self, text: str, doc) -> List[str]:
        """Extract spoken languages."""
        languages = []
        
        # Look for language patterns
        lang_patterns = [
            r'(?:languages?|speaks?|fluent in|proficient in)\s*:?\s*([^.\n]+)',
            r'(?:english|spanish|french|german|italian|portuguese|chinese|japanese|korean|arabic|hindi|russian)',
        ]
        
        for pattern in lang_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                lang_text = match.group(1) if match.groups() else match.group()
                # Split by common separators
                lang_list = re.split(r'[,;|&]', lang_text)
                for lang in lang_list:
                    lang = lang.strip()
                    if len(lang) > 1 and len(lang) < 20:
                        languages.append(lang)
        
        return list(set(languages))
    
    def _get_context_around_match(self, match, text: str, context_size: int = 50) -> str:
        """Get context around a regex match."""
        start = max(0, match.start() - context_size)
        end = min(len(text), match.end() + context_size)
        return text[start:end].strip()
    
    def create_entity_features(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create feature vectors from extracted entities for ML models.
        
        Args:
            entities: Extracted entities dictionary
            
        Returns:
            Dictionary of features for machine learning
        """
        features = {}
        
        # Skill-based features
        if entities['skills']:
            skill_categories = defaultdict(int)
            for skill in entities['skills']:
                skill_categories[skill['category']] += 1
            
            features['skill_counts'] = dict(skill_categories)
            features['total_skills'] = len(entities['skills'])
            features['high_confidence_skills'] = len([s for s in entities['skills'] if s['confidence'] > 0.7])
        
        # Experience features
        features['experience_count'] = len(entities['experience'])
        features['education_count'] = len(entities['education'])
        features['certification_count'] = len(entities['certifications'])
        features['project_count'] = len(entities['projects'])
        
        # Contact completeness
        contact_score = 0
        if entities['contact_info'].get('email'):
            contact_score += 1
        if entities['contact_info'].get('phone'):
            contact_score += 1
        if entities['contact_info'].get('linkedin'):
            contact_score += 1
        if entities['contact_info'].get('github'):
            contact_score += 1
        
        features['contact_completeness'] = contact_score / 4.0
        
        # Text quality features
        features['text_length'] = entities['metadata']['text_length']
        features['word_count'] = entities['metadata']['word_count']
        features['sentence_count'] = entities['metadata']['sentence_count']
        features['entity_density'] = entities['metadata']['entity_count'] / max(entities['metadata']['word_count'], 1)
        
        return features


def extract_entities_from_resume(text: str, model_name: str = "en_core_web_sm") -> Dict[str, Any]:
    """
    Convenience function to extract entities from resume text.
    
    Args:
        text: Resume text to process
        model_name: spaCy model to use
        
    Returns:
        Dictionary containing all extracted entities
    """
    ner = ResumeNER(model_name)
    return ner.extract_entities(text)


def batch_extract_entities(texts: List[str], model_name: str = "en_core_web_sm") -> List[Dict[str, Any]]:
    """
    Extract entities from multiple resume texts in batch.
    
    Args:
        texts: List of resume texts
        model_name: spaCy model to use
        
    Returns:
        List of entity dictionaries
    """
    ner = ResumeNER(model_name)
    return [ner.extract_entities(text) for text in texts]


def main():
    """Example usage of the NER system."""
    # Sample resume text
    sample_text = """
    John Smith
    Software Engineer
    john.smith@email.com
    (555) 123-4567
    linkedin.com/in/johnsmith
    github.com/johnsmith
    
    EXPERIENCE
    Senior Software Engineer at Google (2020-2023)
    - Developed web applications using Python, Django, and React
    - Worked with AWS cloud services and Docker
    - Led a team of 5 developers
    
    Software Developer at Microsoft (2018-2020)
    - Built scalable applications using C# and .NET
    - Experience with Azure cloud platform
    - Collaborated with cross-functional teams
    
    EDUCATION
    Bachelor of Science in Computer Science
    Stanford University (2014-2018)
    
    SKILLS
    Programming: Python, Java, JavaScript, C#, SQL
    Frameworks: Django, React, .NET, Spring Boot
    Cloud: AWS, Azure, Google Cloud
    Tools: Git, Docker, Kubernetes, Jenkins
    
    CERTIFICATIONS
    AWS Certified Solutions Architect
    Microsoft Azure Fundamentals
    """
    
    # Extract entities
    entities = extract_entities_from_resume(sample_text)
    
    # Print results
    print("=== EXTRACTED ENTITIES ===")
    for category, data in entities.items():
        if data and category != 'metadata':
            print(f"\n{category.upper()}:")
            if isinstance(data, list):
                for item in data:
                    print(f"  - {item}")
            elif isinstance(data, dict):
                for key, value in data.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {data}")


if __name__ == "__main__":
    main()
