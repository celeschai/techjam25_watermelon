# Business Review Moderation Pipeline

A comprehensive machine learning pipeline for automatically detecting policy violations and evaluating review quality in business reviews using Google's Gemma-3-4B-IT multimodal model.

## Project Overview

This project implements an automated review moderation system that analyzes business reviews for:
- **Policy Violations**: Advertisement content, irrelevant content, and non-visitor rants
- **Content Quality**: Review helpfulness and sentiment-rating consistency
- **Multimodal Analysis**: Both text and image content analysis

The system processes Google Reviews data from Vermont, USA, and provides both a standalone pipeline and an interactive Streamlit dashboard for real-time analysis.

## Architecture

| Stage | Data Sources | Preprocessing | ML Pipeline | Dashboard |
|-------|--------------|---------------|-------------|-----------|
| **Input** | Google Reviews | Data Cleaning | Gemma Model | Streamlit UI |
| **Process** | Business Metadata | Feature Engineering | Policy Detection | Results Visualization |

## Key Features

- **Multimodal Analysis**: Processes both text reviews and uploaded images
- **Policy Enforcement**: Detects ads, irrelevant content, and non-visitor rants
- **Quality Assessment**: Evaluates review helpfulness and sentiment consistency
- **Interactive Dashboard**: Real-time CSV upload and processing via Streamlit
- **Robust Processing**: Handles missing data, encoding issues, and network failures gracefully

## Data Collection and Preprocessing

### 1. Data Sources
- **Reviews Dataset**: 324,725 Google Reviews from Vermont businesses
  - Source: `https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/review-Vermont_10.json.gz`
  - Columns: user_id, time, rating, text, pics, resp, gmap_id
  
- **Business Metadata**: 11,291 business locations
  - Source: `https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/meta-Vermont.json.gz`
  - Columns: name, address, category, description, coordinates, etc.

### 2. Data Cleaning Pipeline
```python
# Reviews Dataset
- Remove duplicate reviews (user_id, text, gmap_id, time)
- Convert text to lowercase
- Clean whitespace in text fields
- Collapse nested picture URLs into flat lists
- Extract business response text from nested structures

# Business Metadata
- Remove irrelevant columns (address, coordinates, pricing, etc.)
- Clean category and description text
- Remove duplicates based on name and gmap_id
```

### 3. Data Enrichment
- **Web Scraping**: Used Google Maps Review Scraper to collect:
  - Business categories
  - Business images for visual analysis
- **Data Merging**: Combined reviews and metadata using gmap_id
- **Manual Labeling**: Added boolean columns for policy violations and quality metrics

### 4. Final Dataset Structure
- **Input Columns**: review_id, user_id, time, rating, text, pics_collapsed, name, category
- **Output Columns**: All input + policy violation flags + quality metrics
- **Filtering**: Removed reviews with only ratings (no text/images) - reduces dataset by ~45%

## Modeling and Policy Enforcement

### 1. Model Architecture
- **Base Model**: Google Gemma-3-4B-IT (Instruction Tuned)
- **Task**: Multimodal image-text-to-text generation
- **Device Support**: MPS (Apple Silicon), CUDA, CPU fallback
- **Precision**: bfloat16 for memory efficiency

### 2. Policy Detection Categories

#### Text-Based Violations
- **Advertisement Detection** (`is_text_ad`)
  - Marketing language, external business promotion
  - Contact information, promotional codes
  - Examples: "Visit our website", "Follow me on Instagram for promo codes"

- **Irrelevant Content** (`is_text_irrelevant`)
  - Unrelated topics, personal issues
  - Political commentary, unrelated events
  - Examples: "The food was good, but I hate politics"

- **Non-Visitor Rants** (`is_text_rant`)
  - Reviews from people who never visited
  - Second-hand information, hearsay
  - Examples: "Never been here, but heard it's terrible"

#### Image-Based Violations
- **Advertisement Images** (`is_image_ad`)
  - Promotional content, marketing materials
  - External business logos, promotional text

- **Irrelevant Images** (`is_image_irrelevant`)
  - Unrelated to business/location
  - Personal photos, unrelated content

### 3. Quality Metrics

#### Helpfulness Assessment
- **not_helpful**: No useful information beyond basic facts
- **helpful**: Some helpful information for visit decisions
- **very_helpful**: Detailed, specific information that significantly impacts decisions

#### Sentiment Consistency
- **sensibility**: Alignment between text sentiment and numerical rating
- Helps identify potentially fake or manipulated reviews

### 4. Prompt Engineering
Each analysis task uses carefully crafted prompts with:
- Clear task definitions
- Example violations from training data
- Structured JSON output requirements
- Business context for informed decisions

## Setup Instructions

### 1. Environment Requirements
```bash
# Python 3.8+ required
python --version

# Install dependencies
pip install -r requirements.txt
```

### 2. Hugging Face Authentication
```bash
# Create .env file with your HF token
echo "HF_TOKEN=your_token_here" > .env

# Or set environment variable
export HF_TOKEN=your_token_here
```

### 3. Model Download
The pipeline automatically downloads the Gemma-3-4B-IT model on first run (~8GB).

### 4. Data Preparation
```bash
# Run the data processing notebook
jupyter notebook data_processing.ipynb

# Or use the test file preparation script
python test_file_prep.py vt_merged.csv 1000 500 --output-dir ./splits
```

## Usage

### 1. Standalone Pipeline
```bash
# Process validation dataset
python gemma_pipeline.py

# Output: pipeline_output.csv with all analysis results
```

### 2. Interactive Dashboard
```bash
# Launch Streamlit dashboard
streamlit run dashboard.py

# Upload CSV via web interface
# Real-time processing and visualization
```

### 3. Test Data Preparation
```bash
# Split dataset into test/validation sets
python test_file_prep.py input.csv 1000 500 --seed 42

# Options:
# --output-dir: Output directory (default: current)
# --seed: Random seed for reproducibility
```

## How to Reproduce Results

### 1. Data Pipeline Reproduction
```python
# 1. Download raw data
# 2. Run data_processing.ipynb cells sequentially
# 3. Verify output files: vt_merged.csv, vt_metadata.csv

# Expected outputs:
# - Reviews: ~178,000 rows (after filtering)
# - Columns: 20+ including policy flags
```

### 2. Model Training Reproduction
```bash
# 1. Set HF_TOKEN in .env
# 2. Run pipeline on validation set
python gemma_pipeline.py

# Expected outputs:
# - Processing time: ~X seconds per review
# - Policy violation counts
# - Quality metric distributions
```

### 3. Dashboard Reproduction
```bash
# 1. Launch dashboard
streamlit run dashboard.py

# 2. Upload vt_merged.csv
# 3. Wait for backend processing
# 4. View results in interactive interface
```

## Performance Characteristics

### Processing Speed
- **Text Analysis**: ~1-2 seconds per review
- **Image Analysis**: ~3-5 seconds per image (limited to 3 images per review)
- **Total Pipeline**: Varies by dataset size and hardware

### Resource Requirements
- **Memory**: 8GB+ RAM recommended
- **Storage**: 10GB+ for model and data
- **GPU**: MPS (Apple Silicon) or CUDA for acceleration

### Accuracy Metrics
- **Policy Detection**: Based on LLM classification with confidence scores
- **Fallback Handling**: Conservative approach (false negatives over false positives)
- **Error Recovery**: Graceful handling of parsing failures and network issues

## File Structure

```
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── data_processing.ipynb              # Data collection and preprocessing
├── gemma_pipeline.py                  # Main ML pipeline
├── gemma_pipeline_dashboard.py        # Dashboard-specific pipeline
├── examples.py                        # Training examples for prompts
├── test_file_prep.py                  # Dataset splitting utilities
├── dashboard.py                       # Streamlit dashboard interface
├── .env                               # Environment variables (create this)
└── data/                              # Data files (not in repo)
    ├── vt_merged.csv                  # Processed dataset
    ├── vt_metadata.csv                # Business metadata
    └── imagecategories.csv            # Scraped business data
```

## Input/Output Schemas

### Input CSV Requirements
- **Required Columns**: user_id, time, rating, text, pics_collapsed, name, category
- **Encoding**: UTF-8, UTF-8-sig, or Latin-1
- **Format**: Standard CSV with headers

### Output CSV Structure
- **Original Data**: All input columns preserved
- **Policy Flags**: Boolean columns for each violation type
- **Quality Metrics**: Helpfulness ratings and sentiment consistency
- **Analysis Metadata**: Processing timestamps and confidence scores

## Error Handling

### Robust Processing
- **Encoding Issues**: Multiple encoding fallbacks
- **Missing Data**: Graceful handling of NaN values
- **Network Failures**: Timeout protection for image downloads
- **Model Errors**: Fallback to safe default values

### User Feedback
- **Validation Errors**: Clear error messages for schema issues
- **Processing Status**: Real-time progress indicators
- **Debug Information**: Expandable logs and tracebacks

## Future Enhancements

### Planned Features
- **Batch Processing**: Support for larger datasets
- **Model Fine-tuning**: Custom training on review data
- **API Integration**: RESTful endpoints for external systems
- **Advanced Analytics**: Temporal trends and business insights

### Performance Optimizations
- **Parallel Processing**: Multi-threaded review analysis
- **Caching**: Model and result caching for repeated queries
- **Streaming**: Real-time processing of incoming reviews

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Run tests and linting
5. Submit a pull request

### Code Standards
- **Documentation**: Comprehensive docstrings and comments
- **Error Handling**: Graceful degradation and user feedback
- **Testing**: Unit tests for core functionality
- **Performance**: Monitoring and optimization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Data Source**: UCSD McAuley Lab for Google Local dataset
- **Model**: Google Gemma-3-4B-IT for multimodal analysis
- **Framework**: Hugging Face Transformers and Streamlit
- **Infrastructure**: Apple Silicon MPS acceleration support

## Support

For questions, issues, or contributions:
1. Check the documentation and examples
2. Review existing issues and discussions
3. Create a new issue with detailed information
4. Contact the development team

---

*This README provides a comprehensive overview of the Business Review Moderation Pipeline. For detailed implementation specifics, refer to the individual Python files and Jupyter notebooks.*
