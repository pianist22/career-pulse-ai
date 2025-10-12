# NER Integration Guide - Career Pulse AI

## 🎯 Complete Step-by-Step Integration Plan

This guide provides a comprehensive plan for integrating Named Entity Recognition (NER) into the Career Pulse AI resume classification system **without disrupting the existing workflow**.

## 📋 Prerequisites

### 1. Install Dependencies
```bash
# Install Python requirements
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm

# Or use the setup script
python setup_ner.py
```

### 2. Verify Installation
```python
import spacy
nlp = spacy.load("en_core_web_sm")
print("✅ NER setup complete!")
```

## 🚀 Execution Steps

### Step 1: Setup NER Environment
```bash
# Run the NER setup script
python setup_ner.py
```

### Step 2: Configure NER Settings
The `config.yaml` file has been updated with NER configuration:
```yaml
ner:
  enabled: true
  model_name: "en_core_web_sm"
  extract_skills: true
  extract_education: true
  extract_experience: true
  extract_certifications: true
  extract_technologies: true
  extract_companies: true
  extract_locations: true
  extract_dates: true
  extract_contact_info: true
  save_entity_features: true
```

### Step 3: Run Existing Workflow with NER
```bash
# 1. Create test data (optional)
python create_test_files.py

# 2. Load ResumeAtlas dataset
python -m src.ingest.load_resumeatlas

# 3. Process local files (if you have them)
python -m src.ingest.load_local_files

# 4. Run preprocessing WITH NER (this is where NER gets integrated)
python -m src.preprocess.normalize
```

### Step 4: Verify NER Integration
Check the generated parquet files for entity columns:
```python
import pandas as pd

# Load processed data
train = pd.read_parquet("data/processed/classification_train.parquet")
print("Columns:", train.columns.tolist())

# Check for NER features
entity_columns = [col for col in train.columns if col.startswith('entity_')]
print(f"NER features found: {len(entity_columns)}")
print("Entity columns:", entity_columns)
```

### Step 5: Run Enhanced Training
```bash
# Run the enhanced training notebook
jupyter notebook notebooks/02_ner_enhanced_training.ipynb
```

## 🔧 NER Features Extracted

### 1. **Skills Extraction**
- Programming languages (Python, Java, JavaScript, etc.)
- Frameworks (React, Django, Spring, etc.)
- Tools (Docker, Kubernetes, Git, etc.)
- Soft skills (leadership, communication, etc.)

### 2. **Education Information**
- Degrees (Bachelor's, Master's, PhD, etc.)
- Fields of study (Computer Science, Engineering, etc.)
- Institutions (Universities, Colleges, etc.)
- Graduation years and GPAs

### 3. **Work Experience**
- Job titles and positions
- Company names
- Experience duration

### 4. **Certifications**
- Professional certifications (AWS, Azure, PMP, etc.)
- Technical certifications
- Industry-specific certifications

### 5. **Technical Stack**
- Databases (MySQL, MongoDB, etc.)
- Cloud platforms (AWS, Azure, GCP, etc.)
- Development tools and IDEs

### 6. **Contact Information**
- Email addresses
- Phone numbers
- URLs and LinkedIn profiles

## 📊 Expected Improvements

### Before NER Integration:
- **Accuracy**: ~88.5%
- **Features**: Text-only classification
- **Limitations**: Limited understanding of resume structure

### After NER Integration:
- **Expected Accuracy**: 90-95%+ (estimated)
- **Features**: Text + Structured entities
- **Benefits**: 
  - Better job category classification
  - Enhanced feature engineering
  - Improved understanding of resume content

## 🔄 Workflow Integration Points

### 1. **Non-Breaking Integration**
- NER runs **alongside** existing preprocessing
- Original workflow remains unchanged
- NER features are **additional** columns
- Can be disabled via config

### 2. **Processing Flow**
```
Original Text → Text Cleaning → NER Processing → Entity Features → Combined Features → Classification
```

### 3. **Data Flow**
```
Input: Raw Resume Text
↓
Text Preprocessing (existing)
↓
NER Processing (new)
↓
Entity Feature Extraction (new)
↓
Combined Feature Engineering (new)
↓
Enhanced Classification (improved)
```

## 🛠️ Troubleshooting

### Common Issues:

#### 1. spaCy Model Not Found
```bash
# Install the model
python -m spacy download en_core_web_sm

# Verify installation
python -c "import spacy; spacy.load('en_core_web_sm')"
```

#### 2. NER Processing Fails
- Check if `ner.enabled: true` in config.yaml
- Verify spaCy installation
- Check for memory issues with large datasets

#### 3. No Entity Columns Found
- Ensure `save_entity_features: true` in config.yaml
- Re-run preprocessing: `python -m src.preprocess.normalize`
- Check console output for NER processing messages

#### 4. Performance Issues
- Use smaller spaCy model: `en_core_web_sm`
- Reduce batch size in processing
- Consider processing in chunks for large datasets

## 📈 Performance Optimization

### 1. **Model Selection**
```yaml
# For speed (recommended for development)
ner:
  model_name: "en_core_web_sm"

# For accuracy (production)
ner:
  model_name: "en_core_web_lg"
```

### 2. **Selective Feature Extraction**
```yaml
ner:
  extract_skills: true
  extract_education: true
  extract_experience: false  # Disable if not needed
  extract_certifications: true
  # ... other features
```

### 3. **Batch Processing**
For large datasets, consider processing in batches:
```python
# Process in chunks to avoid memory issues
chunk_size = 1000
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    # Process chunk with NER
```

## 🎯 Next Steps

### 1. **Immediate Actions**
- [ ] Run `python setup_ner.py`
- [ ] Update config.yaml with NER settings
- [ ] Run preprocessing with NER enabled
- [ ] Verify entity columns are created

### 2. **Training & Evaluation**
- [ ] Run enhanced training notebook
- [ ] Compare accuracy with/without NER
- [ ] Analyze entity feature importance
- [ ] Fine-tune NER parameters

### 3. **Advanced Features**
- [ ] Custom entity types for specific domains
- [ ] Entity-based feature selection
- [ ] Multi-model ensemble with NER features
- [ ] Real-time NER processing for new resumes

## 📞 Support

If you encounter issues:

1. **Check logs**: Look for NER processing messages in console output
2. **Verify config**: Ensure NER settings are correct in config.yaml
3. **Test components**: Run individual NER modules separately
4. **Memory issues**: Reduce batch sizes or use smaller spaCy models

## 🎉 Success Metrics

You'll know NER integration is successful when:

✅ Entity columns appear in processed data  
✅ Enhanced training shows improved accuracy  
✅ Entity features contribute to classification  
✅ No disruption to existing workflow  

---

**Happy NER Integration! 🚀**
