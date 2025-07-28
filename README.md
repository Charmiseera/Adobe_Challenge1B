# Document Intelligence System - Usage Guide

## Prerequisites
- Docker installed and running
- PDF documents to analyze

## Setup

1. **Directory Structure:**
```
c:\Users\asus\OneDrive\Desktop\Adobeb\
├── 1b.py                    # Main processing script
├── requirements.txt         # Python dependencies  
├── Dockerfile              # Docker configuration
├── input_config.json       # Input configuration
├── pdf_folder/             # Place your PDF files here
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
└── 1b_output/              # Output will be generated here
    └── round1b_output.json
```

2. **Configure Input (input_config.json):**
```json
{
  "persona": {
    "role": "Your Role (e.g., PhD Researcher in Computational Biology)"
  },
  "job_to_be_done": {
    "task": "Your specific task (e.g., Prepare a comprehensive literature review...)"
  },
  "documents": [
    {"filename": "document1.pdf"},
    {"filename": "document2.pdf"},
    {"filename": "document3.pdf"}
  ]
}
```

3. **Sample Test Cases:**

**Academic Research:**
```json
{
  "persona": {"role": "PhD Researcher in Computational Biology"},
  "job_to_be_done": {"task": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks for Graph Neural Networks in Drug Discovery (2022-2024)"},
  "documents": [
    {"filename": "gnn_drug_discovery_paper1.pdf"},
    {"filename": "molecular_property_prediction.pdf"},
    {"filename": "benchmark_datasets.pdf"}
  ]
}
```

**Financial Analysis:**
```json
{
  "persona": {"role": "Investment Analyst"},
  "job_to_be_done": {"task": "Analyze revenue trends, R&D investments, and market positioning strategies"},
  "documents": [
    {"filename": "annual_report_2023.pdf"},
    {"filename": "quarterly_earnings.pdf"}
  ]
}
```

**Educational Content:**
```json
{
  "persona": {"role": "Undergraduate Chemistry Student"},
  "job_to_be_done": {"task": "Identify key concepts and mechanisms for exam preparation on reaction kinetics"},
  "documents": [
    {"filename": "organic_chemistry_ch1.pdf"},
    {"filename": "reaction_mechanisms.pdf"}
  ]
}
```

## Running the System

### Method 1: Using Docker (Recommended)
```powershell
# Build the Docker image (done once)
docker build -t document-analyzer .

# Run the analysis
docker run -v ${PWD}/pdf_folder:/app/pdf_folder -v ${PWD}/input_config.json:/app/input_config.json -v ${PWD}/1b_output:/app/1b_output document-analyzer
```

### Method 2: Using Python directly
```powershell
# Install dependencies
pip install -r requirements.txt

# Run the analysis
python 1b.py --input_json ./input_config.json --pdf_folder ./pdf_folder --output_folder ./1b_output --top_k 5 --use_mmr
```

## Command Line Options

- `--input_json`: Path to input configuration JSON file
- `--pdf_folder`: Folder containing PDF documents
- `--output_folder`: Folder to write output JSON
- `--top_k`: Number of top sections to output (default: 5)
- `--use_mmr`: Use Maximum Marginal Relevance for diversity

## Output Format

The system generates `round1b_output.json` with:

```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "PhD Researcher in Computational Biology",
    "job": "Prepare a comprehensive literature review...",
    "processing_timestamp": "2025-07-28 10:30:45.123456"
  },
  "extracted_sections": [
    {
      "document": "doc1.pdf",
      "page_number": 1,
      "section_title": "Introduction to Graph Neural Networks",
      "importance_rank": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "doc1.pdf",
      "refined_text": "Key sentences extracted based on relevance...",
      "page_number": 1
    }
  ]
}
```

## System Constraints

- **CPU-only execution**: Optimized for CPU processing
- **Model size**: ≤ 1GB (using all-MiniLM-v2 model ~90MB)
- **Processing time**: ≤ 60 seconds for 3-5 documents
- **No internet access**: All models downloaded during build

## Troubleshooting

1. **PDF not found**: Ensure PDFs are in the `pdf_folder` directory
2. **Memory issues**: Reduce number of documents or use smaller PDFs
3. **Processing timeout**: Check document complexity and size
4. **Missing dependencies**: Rebuild Docker image or reinstall requirements

## Performance Tips

- Use PDFs with clear text (avoid scanned images)
- Limit to 3-10 documents per analysis
- Ensure PDF filenames match exactly in input_config.json
- Use descriptive persona and job descriptions for better results
