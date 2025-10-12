"""
Education Information Parser
Extracts educational background from resume text
"""

import re
from typing import List, Dict, Optional


class EducationParser:
    """
    Parses education information from resume text
    """
    
    def __init__(self):
        """Initialize education parser with degree and institution patterns"""
        self.degree_patterns = self._get_degree_patterns()
        self.institution_patterns = self._get_institution_patterns()
        
    def _get_degree_patterns(self) -> List[Dict[str, str]]:
        """Get regex patterns for different degree types"""
        return [
            {
                "pattern": r'\b(?:Bachelor|B\.?S\.?|B\.?A\.?|B\.?E\.?|B\.?Tech|B\.?Eng)\s+(?:of\s+)?(?:Science|Arts|Engineering|Technology|Computer Science|Information Technology|Business Administration|Management|Finance|Marketing|Accounting|Economics|Mathematics|Physics|Chemistry|Biology|Psychology|Sociology|Political Science|History|English|Literature|Philosophy|Art|Design|Architecture|Medicine|Law|Education)\b',
                "degree_type": "Bachelor",
                "field_keywords": ["Science", "Arts", "Engineering", "Technology", "Computer Science", "Information Technology", "Business Administration", "Management", "Finance", "Marketing", "Accounting", "Economics", "Mathematics", "Physics", "Chemistry", "Biology", "Psychology", "Sociology", "Political Science", "History", "English", "Literature", "Philosophy", "Art", "Design", "Architecture", "Medicine", "Law", "Education"]
            },
            {
                "pattern": r'\b(?:Master|M\.?S\.?|M\.?A\.?|M\.?B\.?A\.?|M\.?E\.?|M\.?Tech|M\.?Eng|M\.?Sc)\s+(?:of\s+)?(?:Science|Arts|Engineering|Technology|Computer Science|Information Technology|Business Administration|Management|Finance|Marketing|Accounting|Economics|Mathematics|Physics|Chemistry|Biology|Psychology|Sociology|Political Science|History|English|Literature|Philosophy|Art|Design|Architecture|Medicine|Law|Education)\b',
                "degree_type": "Master",
                "field_keywords": ["Science", "Arts", "Engineering", "Technology", "Computer Science", "Information Technology", "Business Administration", "Management", "Finance", "Marketing", "Accounting", "Economics", "Mathematics", "Physics", "Chemistry", "Biology", "Psychology", "Sociology", "Political Science", "History", "English", "Literature", "Philosophy", "Art", "Design", "Architecture", "Medicine", "Law", "Education"]
            },
            {
                "pattern": r'\b(?:PhD|Ph\.?D\.?|Doctorate|D\.?Phil\.?|Doctor)\s+(?:of\s+)?(?:Philosophy|Science|Arts|Engineering|Technology|Computer Science|Information Technology|Business Administration|Management|Finance|Marketing|Accounting|Economics|Mathematics|Physics|Chemistry|Biology|Psychology|Sociology|Political Science|History|English|Literature|Philosophy|Art|Design|Architecture|Medicine|Law|Education)\b',
                "degree_type": "PhD",
                "field_keywords": ["Philosophy", "Science", "Arts", "Engineering", "Technology", "Computer Science", "Information Technology", "Business Administration", "Management", "Finance", "Marketing", "Accounting", "Economics", "Mathematics", "Physics", "Chemistry", "Biology", "Psychology", "Sociology", "Political Science", "History", "English", "Literature", "Philosophy", "Art", "Design", "Architecture", "Medicine", "Law", "Education"]
            },
            {
                "pattern": r'\b(?:Associate|A\.?S\.?|A\.?A\.?|A\.?A\.?S\.?)\s+(?:of\s+)?(?:Science|Arts|Applied Science|Engineering|Technology|Computer Science|Information Technology|Business Administration|Management|Finance|Marketing|Accounting|Economics|Mathematics|Physics|Chemistry|Biology|Psychology|Sociology|Political Science|History|English|Literature|Philosophy|Art|Design|Architecture|Medicine|Law|Education)\b',
                "degree_type": "Associate",
                "field_keywords": ["Science", "Arts", "Applied Science", "Engineering", "Technology", "Computer Science", "Information Technology", "Business Administration", "Management", "Finance", "Marketing", "Accounting", "Economics", "Mathematics", "Physics", "Chemistry", "Biology", "Psychology", "Sociology", "Political Science", "History", "English", "Literature", "Philosophy", "Art", "Design", "Architecture", "Medicine", "Law", "Education"]
            },
            {
                "pattern": r'\b(?:Diploma|Certificate|Certification)\s+(?:in\s+)?(?:Computer Science|Information Technology|Engineering|Technology|Business Administration|Management|Finance|Marketing|Accounting|Economics|Mathematics|Physics|Chemistry|Biology|Psychology|Sociology|Political Science|History|English|Literature|Philosophy|Art|Design|Architecture|Medicine|Law|Education|Programming|Web Development|Data Science|Cybersecurity|Digital Marketing|Project Management|Human Resources|Operations|Supply Chain|Logistics)\b',
                "degree_type": "Diploma/Certificate",
                "field_keywords": ["Computer Science", "Information Technology", "Engineering", "Technology", "Business Administration", "Management", "Finance", "Marketing", "Accounting", "Economics", "Mathematics", "Physics", "Chemistry", "Biology", "Psychology", "Sociology", "Political Science", "History", "English", "Literature", "Philosophy", "Art", "Design", "Architecture", "Medicine", "Law", "Education", "Programming", "Web Development", "Data Science", "Cybersecurity", "Digital Marketing", "Project Management", "Human Resources", "Operations", "Supply Chain", "Logistics"]
            }
        ]
    
    def _get_institution_patterns(self) -> List[str]:
        """Get patterns for educational institutions"""
        return [
            r'\b(?:University|College|Institute|School|Academy|Technical|Polytechnic)\s+of\s+([A-Za-z\s&.,-]+)',
            r'\b([A-Za-z\s&.,-]+)\s+(?:University|College|Institute|School|Academy|Technical|Polytechnic)\b',
            r'\b(?:IIT|MIT|Stanford|Harvard|Yale|Princeton|Columbia|Cornell|Dartmouth|Brown|Pennsylvania|Chicago|Northwestern|Duke|Vanderbilt|Rice|Emory|Georgetown|Carnegie Mellon|CMU|Berkeley|UCLA|UCSD|UCSB|UCI|UCD|UCR|UCM|UCSC|UCSF|Caltech|Georgia Tech|Virginia Tech|Texas A&M|Purdue|Illinois|Michigan|Wisconsin|Minnesota|Ohio State|Penn State|Rutgers|Maryland|Florida|North Carolina|Virginia|Georgia|Alabama|Tennessee|Kentucky|Louisiana|Mississippi|Arkansas|Oklahoma|Texas|New Mexico|Arizona|Colorado|Utah|Nevada|California|Oregon|Washington|Alaska|Hawaii)\b',
            r'\b[A-Z][a-z]+\s+(?:University|College|Institute|School|Academy|Technical|Polytechnic)\b'
        ]
    
    def extract_education(self, text: str, doc=None) -> List[Dict[str, str]]:
        """
        Extract education information from resume text
        
        Args:
            text: Resume text to analyze
            doc: Optional spaCy document (not used in current implementation)
            
        Returns:
            List of education dictionaries
        """
        if not text:
            return []
        
        education_info = []
        
        # Extract degrees
        degrees = self._extract_degrees(text)
        
        # Extract institutions
        institutions = self._extract_institutions(text)
        
        # Extract years
        years = self._extract_years(text)
        
        # Extract GPAs
        gpas = self._extract_gpas(text)
        
        # Combine information
        education_info.extend(degrees)
        
        # Try to match degrees with institutions and years
        for degree in degrees:
            matched_info = self._match_education_context(text, degree)
            if matched_info:
                education_info.append(matched_info)
        
        # Add standalone institutions
        for institution in institutions:
            education_info.append({
                "type": "Institution",
                "name": institution,
                "degree": "",
                "field": "",
                "year": "",
                "gpa": "",
                "context": institution
            })
        
        return education_info
    
    def _extract_degrees(self, text: str) -> List[Dict[str, str]]:
        """Extract degree information"""
        degrees = []
        
        for pattern_info in self.degree_patterns:
            pattern = pattern_info["pattern"]
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                full_match = match.group(0)
                
                # Determine field of study
                field = self._extract_field_of_study(full_match, pattern_info["field_keywords"])
                
                degrees.append({
                    "type": "Degree",
                    "name": full_match,
                    "degree": pattern_info["degree_type"],
                    "field": field,
                    "year": "",
                    "gpa": "",
                    "context": full_match
                })
        
        return degrees
    
    def _extract_institutions(self, text: str) -> List[str]:
        """Extract educational institutions"""
        institutions = []
        
        for pattern in self.institution_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                institution = match.group(1) if match.groups() else match.group(0)
                institutions.append(institution.strip())
        
        return list(set(institutions))
    
    def _extract_years(self, text: str) -> List[str]:
        """Extract graduation years"""
        year_patterns = [
            r'\b(?:graduated|completed|finished)\s+(?:in\s+)?(\d{4})\b',
            r'\b(\d{4})\s*(?:graduation|graduated|completed|finished)\b',
            r'\b(\d{4})\s*[-–]\s*(\d{4})\b',  # Year range
            r'\b(?:from|to)\s+(\d{4})\b'
        ]
        
        years = []
        for pattern in year_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    years.extend(match)
                else:
                    years.append(match)
        
        return years
    
    def _extract_gpas(self, text: str) -> List[str]:
        """Extract GPA information"""
        gpa_patterns = [
            r'\bGPA\s*:?\s*(\d+\.?\d*)\s*(?:out\s+of\s+\d+\.?\d*)?\b',
            r'\b(\d+\.?\d*)\s*(?:GPA|gpa)\b',
            r'\bGrade\s+Point\s+Average\s*:?\s*(\d+\.?\d*)\b'
        ]
        
        gpas = []
        for pattern in gpa_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            gpas.extend(matches)
        
        return gpas
    
    def _extract_field_of_study(self, degree_text: str, field_keywords: List[str]) -> str:
        """Extract field of study from degree text"""
        for field in field_keywords:
            if field.lower() in degree_text.lower():
                return field
        return ""
    
    def _match_education_context(self, text: str, degree_info: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Match degree with nearby institution and year information"""
        degree_text = degree_info["context"]
        degree_pos = text.find(degree_text)
        
        if degree_pos == -1:
            return None
        
        # Look for nearby context (within 200 characters)
        start = max(0, degree_pos - 100)
        end = min(len(text), degree_pos + len(degree_text) + 100)
        context = text[start:end]
        
        # Extract institution from context
        institutions = self._extract_institutions(context)
        institution = institutions[0] if institutions else ""
        
        # Extract year from context
        years = self._extract_years(context)
        year = years[0] if years else ""
        
        # Extract GPA from context
        gpas = self._extract_gpas(context)
        gpa = gpas[0] if gpas else ""
        
        return {
            "type": "Education",
            "name": f"{degree_info['degree']} - {institution}".strip(" -"),
            "degree": degree_info["degree"],
            "field": degree_info["field"],
            "institution": institution,
            "year": year,
            "gpa": gpa,
            "context": context.strip()
        }
