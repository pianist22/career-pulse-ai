"""
Quick Start Script for AI Resume Shortener with NER

This script provides a quick way to test the NER integration
without running the full pipeline.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

def quick_ner_test():
    """Quick test of NER functionality."""
    print("🚀 QUICK START - NER INTEGRATION TEST")
    print("=" * 50)
    
    try:
        # Test NER extraction
        from src.preprocess.ner_extraction import extract_entities_from_resume
        
        sample_text = """
        John Smith
        Senior Software Engineer
        john.smith@email.com
        (555) 123-4567
        
        EXPERIENCE
        Senior Software Engineer at Google (2020-2023)
        - Developed web applications using Python, Django, and React
        - Worked with AWS cloud services and Docker
        
        EDUCATION
        Bachelor of Science in Computer Science
        Stanford University (2014-2018)
        
        SKILLS
        Python, Java, JavaScript, Django, React, AWS, Docker
        """
        
        print("📝 Testing NER extraction...")
        entities = extract_entities_from_resume(sample_text)
        
        print("✅ NER extraction successful!")
        print(f"📊 Extracted {len(entities.get('skills', []))} skills")
        print(f"🏢 Found {len(entities.get('companies', []))} companies")
        print(f"🎓 Found {len(entities.get('education', []))} education entries")
        
        # Show sample results
        if entities.get('skills'):
            print(f"\n🔧 Sample Skills: {[skill['skill'] for skill in entities['skills'][:5]]}")
        
        if entities.get('companies'):
            print(f"🏢 Companies: {entities['companies']}")
        
        if entities.get('contact_info', {}).get('email'):
            print(f"📧 Email: {entities['contact_info']['email']}")
        
        return True
        
    except Exception as e:
        print(f"❌ NER test failed: {e}")
        return False

def main():
    """Main function for quick start."""
    print("🎯 AI RESUME SHORTENER - QUICK START")
    print("=" * 50)
    
    # Test NER functionality
    if quick_ner_test():
        print("\n✅ NER INTEGRATION IS WORKING!")
        print("\n🚀 To run the complete pipeline, use:")
        print("   python run_complete_pipeline.py")
        print("\n📚 For detailed NER testing, use:")
        print("   python test_ner_integration.py")
        print("\n🎮 For NER demonstration, use:")
        print("   python run_ner_pipeline.py")
    else:
        print("\n❌ NER INTEGRATION HAS ISSUES")
        print("Please check the error messages above and fix dependencies.")

if __name__ == "__main__":
    main()
