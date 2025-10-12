"""
Named Entity Recognition for Resume Processing
Extracts entities like skills, education, experience, etc. from resume text
"""

import spacy
import pandas as pd
import re
from typing import Dict, List, Any, Optional
from pathlib import Path


class NERProcessor:
    """
    Main NER processor that coordinates different entity extraction modules
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize NER processor with spaCy model
        
        Args:
            model_name: spaCy model to use (default: en_core_web_sm)
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Warning: {model_name} not found. Using blank English model.")
            self.nlp = spacy.blank("en")
        
        self.skill_matcher = SkillMatcher()
        self.education_parser = EducationParser()
        
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract all entities from resume text
        
        Args:
            text: Resume text to process
            
        Returns:
            Dictionary containing all extracted entities
        """
        if not text or not text.strip():
            return self._get_empty_entities()
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract different types of entities
        entities = {
            "skills": self._extract_skills(text, doc),
            "education": self._extract_education(text, doc),
            "experience": self._extract_experience(text, doc),
            "certifications": self._extract_certifications(text, doc),
            "technologies": self._extract_technologies(text, doc),
            "companies": self._extract_companies(text, doc),
            "degrees": self._extract_degrees(text, doc),
            "locations": self._extract_locations(doc),
            "dates": self._extract_dates(doc),
            "emails": self._extract_emails(text),
            "phones": self._extract_phones(text),
            "urls": self._extract_urls(text)
        }
        
        return entities
    
    def _extract_skills(self, text: str, doc) -> List[str]:
        """Extract technical and soft skills"""
        skills = []
        
        # Use skill matcher for technical skills
        technical_skills = self.skill_matcher.extract_skills(text)
        skills.extend(technical_skills)
        
        # Extract soft skills using patterns
        soft_skills_patterns = [
            r'\b(?:leadership|communication|teamwork|problem solving|analytical|creative|innovative|adaptable|detail oriented|time management|project management|customer service|negotiation|presentation|collaboration|mentoring|training|supervision|strategic planning|decision making|critical thinking|interpersonal|multitasking|organizational|self motivated|proactive|results oriented)\b',
        ]
        
        for pattern in soft_skills_patterns:
            matches = re.findall(pattern, text.lower())
            skills.extend(matches)
        
        return list(set(skill.strip() for skill in skills if skill.strip()))
    
    def _extract_education(self, text: str, doc) -> List[Dict[str, str]]:
        """Extract education information"""
        return self.education_parser.extract_education(text, doc)
    
    def _extract_experience(self, text: str, doc) -> List[Dict[str, str]]:
        """Extract work experience information"""
        experiences = []
        
        # Look for job titles and companies
        experience_patterns = [
            r'(\w+(?:\s+\w+)*\s+(?:engineer|developer|analyst|manager|director|specialist|consultant|coordinator|administrator|supervisor|lead|senior|junior|assistant|executive|officer|representative|technician|architect|designer|programmer|scientist|researcher))\s+at\s+([A-Z][a-zA-Z\s&.,]+)',
            r'([A-Z][a-zA-Z\s&.,]+)\s*[-–]\s*(\w+(?:\s+\w+)*\s+(?:engineer|developer|analyst|manager|director|specialist|consultant|coordinator|administrator|supervisor|lead|senior|junior|assistant|executive|officer|representative|technician|architect|designer|programmer|scientist|researcher))',
        ]
        
        for pattern in experience_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                experiences.append({
                    "title": match.group(1).strip(),
                    "company": match.group(2).strip(),
                    "context": match.group(0).strip()
                })
        
        return experiences
    
    def _extract_certifications(self, text: str, doc) -> List[str]:
        """Extract professional certifications"""
        certifications = []
        
        cert_patterns = [
            r'\b(?:AWS|Azure|GCP|Google Cloud|Amazon Web Services|Microsoft Azure|Certified|Certificate|Certification)\s+([A-Z][a-zA-Z\s]+)',
            r'\b([A-Z]{2,}(?:\s+[A-Z]{2,})*)\s+(?:Certified|Certificate|Certification)',
            r'\b(?:PMP|CISSP|CISA|CISM|ITIL|Agile|Scrum|Six Sigma|Lean|TOGAF|Prince2)\b',
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            certifications.extend(matches)
        
        return list(set(cert.strip() for cert in certifications if cert.strip()))
    
    def _extract_technologies(self, text: str, doc) -> List[str]:
        """Extract technology/framework mentions"""
        technologies = []
        
        tech_patterns = [
            r'\b(?:Python|Java|JavaScript|TypeScript|C\+\+|C#|PHP|Ruby|Go|Rust|Swift|Kotlin|Scala|R|MATLAB|SQL|NoSQL|MongoDB|PostgreSQL|MySQL|Redis|Docker|Kubernetes|Jenkins|Git|AWS|Azure|GCP|React|Angular|Vue|Node\.js|Django|Flask|Spring|Laravel|Rails|TensorFlow|PyTorch|Pandas|NumPy|Scikit-learn|Hadoop|Spark|Elasticsearch|Apache Kafka|RabbitMQ|GraphQL|REST|API|Microservices|DevOps|CI/CD|Agile|Scrum|Kanban)\b',
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            technologies.extend(matches)
        
        return list(set(tech.strip() for tech in technologies if tech.strip()))
    
    def _extract_companies(self, text: str, doc) -> List[str]:
        """Extract company names"""
        companies = []
        
        # Use spaCy NER for organizations
        for ent in doc.ents:
            if ent.label_ == "ORG":
                companies.append(ent.text.strip())
        
        # Additional patterns for company names
        company_patterns = [
            r'\b(?:at|@|in|with)\s+([A-Z][a-zA-Z\s&.,]+(?:Inc|Corp|LLC|Ltd|Limited|Company|Co|Technologies|Tech|Systems|Solutions|Services|Consulting|Group|Partners|Associates))',
            r'\b([A-Z][a-zA-Z\s&.,]{2,})\s+(?:Inc|Corp|LLC|Ltd|Limited|Company|Co|Technologies|Tech|Systems|Solutions|Services|Consulting|Group|Partners|Associates)\b',
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            companies.extend(matches)
        
        return list(set(comp.strip() for comp in companies if comp.strip()))
    
    def _extract_degrees(self, text: str, doc) -> List[str]:
        """Extract academic degrees"""
        degrees = []
        
        degree_patterns = [
            r'\b(?:Bachelor|Master|PhD|Doctorate|Associate|Diploma|Certificate)\s+(?:of|in)\s+([A-Za-z\s]+)',
            r'\b(?:B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?A\.?|M\.?B\.?A\.?|Ph\.?D\.?|D\.?Phil\.?)\b',
            r'\b(?:Computer Science|Information Technology|Engineering|Business Administration|Data Science|Artificial Intelligence|Machine Learning|Software Engineering|Electrical Engineering|Mechanical Engineering|Civil Engineering|Chemical Engineering|Biomedical Engineering|Aerospace Engineering|Industrial Engineering|Materials Science|Mathematics|Statistics|Physics|Chemistry|Biology|Economics|Finance|Marketing|Management|Accounting|Psychology|Sociology|Political Science|History|English|Literature|Philosophy|Art|Design|Architecture|Medicine|Law|Education)\b',
        ]
        
        for pattern in degree_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            degrees.extend(matches)
        
        return list(set(degree.strip() for degree in degrees if degree.strip()))
    
    def _extract_locations(self, doc) -> List[str]:
        """Extract locations using spaCy NER"""
        locations = []
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:  # Geopolitical entity or Location
                locations.append(ent.text.strip())
        return list(set(locations))
    
    def _extract_dates(self, doc) -> List[str]:
        """Extract dates using spaCy NER"""
        dates = []
        for ent in doc.ents:
            if ent.label_ == "DATE":
                dates.append(ent.text.strip())
        return dates
    
    def _extract_emails(self, text: str) -> List[str]:
        """Extract email addresses"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(email_pattern, text)
    
    def _extract_phones(self, text: str) -> List[str]:
        """Extract phone numbers"""
        phone_pattern = r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        matches = re.findall(phone_pattern, text)
        return [f"({match[0]}) {match[1]}-{match[2]}" for match in matches]
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs"""
        url_pattern = r'https?://\S+|www\.\S+'
        return re.findall(url_pattern, text)
    
    def _get_empty_entities(self) -> Dict[str, Any]:
        """Return empty entities structure"""
        return {
            "skills": [],
            "education": [],
            "experience": [],
            "certifications": [],
            "technologies": [],
            "companies": [],
            "degrees": [],
            "locations": [],
            "dates": [],
            "emails": [],
            "phones": [],
            "urls": []
        }
    
    def entities_to_features(self, entities: Dict[str, Any]) -> Dict[str, str]:
        """
        Convert extracted entities to feature strings for ML pipeline
        
        Args:
            entities: Dictionary of extracted entities
            
        Returns:
            Dictionary with entity features as strings
        """
        return {
            "skills_str": " | ".join(entities.get("skills", [])),
            "technologies_str": " | ".join(entities.get("technologies", [])),
            "certifications_str": " | ".join(entities.get("certifications", [])),
            "companies_str": " | ".join(entities.get("companies", [])),
            "degrees_str": " | ".join(entities.get("degrees", [])),
            "locations_str": " | ".join(entities.get("locations", [])),
            "education_str": " | ".join([str(edu) for edu in entities.get("education", [])]),
            "experience_str": " | ".join([str(exp) for exp in entities.get("experience", [])]),
        }
