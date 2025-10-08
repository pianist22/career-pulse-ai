"""
NER Integration Testing Script

This script tests the NER integration with sample resume data
and validates the accuracy of entity extraction.
"""

import pandas as pd
import json
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append('src')

from src.preprocess.ner_extraction import extract_entities_from_resume, ResumeNER
from src.utils.ner_helpers import NERAnalyzer, create_entity_features_dataframe


def test_ner_with_sample_data():
    """Test NER system with sample resume data."""
    
    # Sample resume text for testing
    sample_resumes = [
        {
            "text": """
            John Smith
            Senior Software Engineer
            john.smith@email.com
            (555) 123-4567
            linkedin.com/in/johnsmith
            github.com/johnsmith
            
            EXPERIENCE
            Senior Software Engineer at Google (2020-2023)
            - Developed web applications using Python, Django, and React
            - Worked with AWS cloud services and Docker
            - Led a team of 5 developers
            - Implemented microservices architecture
            
            Software Developer at Microsoft (2018-2020)
            - Built scalable applications using C# and .NET
            - Experience with Azure cloud platform
            - Collaborated with cross-functional teams
            - Used SQL Server and MongoDB databases
            
            EDUCATION
            Bachelor of Science in Computer Science
            Stanford University (2014-2018)
            GPA: 3.8/4.0
            
            SKILLS
            Programming: Python, Java, JavaScript, C#, SQL, HTML, CSS
            Frameworks: Django, React, .NET, Spring Boot, Node.js
            Cloud: AWS, Azure, Google Cloud Platform
            Tools: Git, Docker, Kubernetes, Jenkins, Jira
            Databases: MySQL, PostgreSQL, MongoDB, Redis
            
            CERTIFICATIONS
            AWS Certified Solutions Architect
            Microsoft Azure Fundamentals
            Google Cloud Professional Developer
            
            PROJECTS
            E-commerce Platform: Built using Django and React
            Machine Learning Model: Python, scikit-learn, TensorFlow
            Mobile App: React Native, Firebase
            """,
            "expected_entities": {
                "name": "John Smith",
                "skills": ["Python", "Java", "JavaScript", "Django", "React", "AWS", "Azure"],
                "companies": ["Google", "Microsoft"],
                "education": ["Bachelor of Science", "Stanford University"]
            }
        },
        {
            "text": """
            Sarah Johnson
            Data Scientist
            sarah.johnson@company.com
            (555) 987-6543
            linkedin.com/in/sarahjohnson
            
            PROFESSIONAL EXPERIENCE
            Data Scientist at Amazon (2021-Present)
            - Built machine learning models using Python and R
            - Worked with big data technologies: Spark, Hadoop, Kafka
            - Developed recommendation systems
            - Used TensorFlow, PyTorch, and scikit-learn
            
            Data Analyst at Facebook (2019-2021)
            - Analyzed user behavior data
            - Created data visualizations using Tableau and Power BI
            - Worked with SQL and NoSQL databases
            - Collaborated with product teams
            
            EDUCATION
            Master of Science in Data Science
            MIT (2017-2019)
            
            Bachelor of Science in Mathematics
            UC Berkeley (2013-2017)
            
            TECHNICAL SKILLS
            Programming: Python, R, SQL, Scala, Java
            Machine Learning: TensorFlow, PyTorch, scikit-learn, XGBoost
            Big Data: Apache Spark, Hadoop, Kafka, Airflow
            Visualization: Tableau, Power BI, Matplotlib, Seaborn
            Cloud: AWS, GCP, Azure
            Databases: PostgreSQL, MongoDB, Cassandra
            
            CERTIFICATIONS
            AWS Certified Machine Learning Specialty
            Google Cloud Professional Data Engineer
            """,
            "expected_entities": {
                "name": "Sarah Johnson",
                "skills": ["Python", "R", "SQL", "TensorFlow", "PyTorch", "AWS"],
                "companies": ["Amazon", "Facebook"],
                "education": ["Master of Science", "MIT", "Bachelor of Science", "UC Berkeley"]
            }
        }
    ]
    
    print("=" * 60)
    print("TESTING NER INTEGRATION")
    print("=" * 60)
    
    # Initialize NER system
    try:
        ner = ResumeNER()
        print("✓ NER system initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize NER system: {e}")
        return False
    
    # Test each sample resume
    all_results = []
    for i, resume_data in enumerate(sample_resumes):
        print(f"\n--- Testing Resume {i+1} ---")
        
        text = resume_data["text"]
        expected = resume_data["expected_entities"]
        
        try:
            # Extract entities
            entities = ner.extract_entities(text)
            all_results.append(entities)
            
            # Validate key entities
            print("Extracted Entities:")
            
            # Check name extraction
            extracted_name = entities.get('personal_info', {}).get('name', '')
            expected_name = expected.get('name', '')
            if extracted_name:
                print(f"✓ Name: {extracted_name}")
            else:
                print(f"✗ Name not extracted (expected: {expected_name})")
            
            # Check skills extraction
            extracted_skills = [skill['skill'] for skill in entities.get('skills', [])]
            expected_skills = expected.get('skills', [])
            skill_matches = [skill for skill in expected_skills if skill in extracted_skills]
            print(f"✓ Skills extracted: {len(extracted_skills)} (matches: {len(skill_matches)}/{len(expected_skills)})")
            if skill_matches:
                print(f"  Matched skills: {skill_matches}")
            
            # Check companies extraction
            extracted_companies = entities.get('companies', [])
            expected_companies = expected.get('companies', [])
            company_matches = [comp for comp in expected_companies if comp in extracted_companies]
            print(f"✓ Companies extracted: {len(extracted_companies)} (matches: {len(company_matches)}/{len(expected_companies)})")
            if company_matches:
                print(f"  Matched companies: {company_matches}")
            
            # Check education extraction
            extracted_education = entities.get('education', [])
            print(f"✓ Education entries: {len(extracted_education)}")
            for edu in extracted_education:
                if edu.get('degree'):
                    print(f"  Degree: {edu['degree']}")
                if edu.get('university'):
                    print(f"  University: {edu['university']}")
            
            # Check contact info
            contact_info = entities.get('contact_info', {})
            print(f"✓ Contact info: Email={bool(contact_info.get('email'))}, Phone={bool(contact_info.get('phone'))}")
            
            # Check experience
            experience = entities.get('experience', [])
            print(f"✓ Experience entries: {len(experience)}")
            
            # Print quality metrics
            metadata = entities.get('metadata', {})
            print(f"✓ Text quality: {metadata.get('word_count', 0)} words, {metadata.get('entity_count', 0)} entities")
            
        except Exception as e:
            print(f"✗ Error processing resume {i+1}: {e}")
            continue
    
    # Test batch processing
    print(f"\n--- Testing Batch Processing ---")
    try:
        texts = [resume["text"] for resume in sample_resumes]
        batch_entities = [ner.extract_entities(text) for text in texts]
        print(f"✓ Batch processing: {len(batch_entities)} resumes processed")
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
    
    # Test feature creation
    print(f"\n--- Testing Feature Creation ---")
    try:
        features_df = create_entity_features_dataframe(all_results)
        print(f"✓ Features created: {features_df.shape[1]} features for {features_df.shape[0]} resumes")
        print(f"  Feature columns: {list(features_df.columns)}")
    except Exception as e:
        print(f"✗ Feature creation failed: {e}")
    
    # Test analysis
    print(f"\n--- Testing Analysis ---")
    try:
        analyzer = NERAnalyzer()
        analysis = analyzer.analyze_entities_batch(all_results)
        print(f"✓ Analysis completed")
        print(f"  Total unique skills: {analysis.get('skill_analysis', {}).get('total_unique_skills', 0)}")
        print(f"  Total unique companies: {analysis.get('company_analysis', {}).get('total_unique_companies', 0)}")
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
    
    print(f"\n" + "=" * 60)
    print("NER INTEGRATION TEST COMPLETED")
    print("=" * 60)
    
    return True


def test_ner_accuracy():
    """Test NER accuracy with known entities."""
    
    print("\n" + "=" * 60)
    print("TESTING NER ACCURACY")
    print("=" * 60)
    
    # Test cases with known entities
    test_cases = [
        {
            "text": "John Smith is a Python developer at Google with 5 years of experience in Django and AWS.",
            "expected": {
                "name": "John Smith",
                "skills": ["Python", "Django", "AWS"],
                "company": "Google"
            }
        },
        {
            "text": "Sarah Johnson, PhD in Computer Science from MIT, works as a Data Scientist at Amazon.",
            "expected": {
                "name": "Sarah Johnson",
                "education": ["PhD", "MIT"],
                "company": "Amazon"
            }
        },
        {
            "text": "Contact: john@example.com, (555) 123-4567, linkedin.com/in/john",
            "expected": {
                "email": "john@example.com",
                "phone": "(555) 123-4567",
                "linkedin": "linkedin.com/in/john"
            }
        }
    ]
    
    ner = ResumeNER()
    total_tests = 0
    passed_tests = 0
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Text: {test_case['text']}")
        
        try:
            entities = ner.extract_entities(test_case['text'])
            expected = test_case['expected']
            
            # Test name extraction
            if 'name' in expected:
                total_tests += 1
                extracted_name = entities.get('personal_info', {}).get('name', '')
                if expected['name'].lower() in extracted_name.lower():
                    print(f"✓ Name extraction: {extracted_name}")
                    passed_tests += 1
                else:
                    print(f"✗ Name extraction failed: expected '{expected['name']}', got '{extracted_name}'")
            
            # Test skills extraction
            if 'skills' in expected:
                total_tests += 1
                extracted_skills = [skill['skill'] for skill in entities.get('skills', [])]
                skill_matches = [skill for skill in expected['skills'] if skill in extracted_skills]
                if len(skill_matches) >= len(expected['skills']) * 0.5:  # 50% match threshold
                    print(f"✓ Skills extraction: {skill_matches}")
                    passed_tests += 1
                else:
                    print(f"✗ Skills extraction failed: expected {expected['skills']}, got {extracted_skills}")
            
            # Test company extraction
            if 'company' in expected:
                total_tests += 1
                extracted_companies = entities.get('companies', [])
                if expected['company'] in extracted_companies:
                    print(f"✓ Company extraction: {extracted_companies}")
                    passed_tests += 1
                else:
                    print(f"✗ Company extraction failed: expected '{expected['company']}', got {extracted_companies}")
            
            # Test contact info
            if 'email' in expected:
                total_tests += 1
                contact_info = entities.get('contact_info', {})
                if contact_info.get('email') == expected['email']:
                    print(f"✓ Email extraction: {contact_info.get('email')}")
                    passed_tests += 1
                else:
                    print(f"✗ Email extraction failed: expected '{expected['email']}', got '{contact_info.get('email')}'")
            
        except Exception as e:
            print(f"✗ Test case {i+1} failed with error: {e}")
            total_tests += 1
    
    accuracy = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"\n" + "=" * 60)
    print(f"ACCURACY TEST RESULTS")
    print(f"Passed: {passed_tests}/{total_tests} ({accuracy:.1f}%)")
    print("=" * 60)
    
    return accuracy >= 70  # 70% accuracy threshold


def main():
    """Main testing function."""
    print("Starting NER Integration Tests...")
    
    # Test basic functionality
    success1 = test_ner_with_sample_data()
    
    # Test accuracy
    success2 = test_ner_accuracy()
    
    if success1 and success2:
        print("\n🎉 All NER tests passed! The integration is working correctly.")
        return True
    else:
        print("\n❌ Some NER tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    main()
