[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# AI-Powered Digital Disaster Database for Bangladesh

**Automated extraction of near real-time disaster information from Bangladeshi English newspapers using NLP, stored in MongoDB.**

---

## Project Overview

This project is my BSc final semester research work. It implements a fully automated daily pipeline that scrapes disaster-related news articles from major Bangladeshi English-language newspapers, extracts structured information using a custom-trained **XLM-RoBERTa** Named Entity Recognition (NER) model, and stores the processed records in **MongoDB Atlas**.

The system focuses primarily on flood-related events and aims to support disaster management, research, and early warning initiatives in Bangladesh.

## Features

- Daily automated scraping from 4 major news sources
- Intelligent filtering for disaster-related articles
- Custom NER model for extracting disaster-specific entities
- Gazetteer-based location enhancement (District, Upazila, Union)
- Duplicate detection and clean storage in MongoDB
- Modular and extensible pipeline

## Tech Stack

- **Language**: Python 3.10+
- **Scraping**: `newspaper3k`, BeautifulSoup4, requests
- **NLP**: spaCy, spacy-transformers, XLM-RoBERTa
- **Database**: MongoDB Atlas + pymongo
- **Annotation**: Label Studio
- **Training**: Google Colab (GPU)
- **Others**: python-dotenv, schedule, pandas

## Project Structure

```
disaster_db/
├── data/                  # Preprocessed articles and gazetteers
├── models/                # Trained spaCy models
├── outputs/               # Sample outputs and results
├── unused/                # Legacy scripts
├── scraper.py             # News scraping logic
├── preprocessor.py        # Text cleaning and preparation
├── entity_linker.py       # Main NLP extraction pipeline
├── db_writer.py           # MongoDB storage logic
├── pipeline.py            # End-to-end daily pipeline
├── evaluator.py           # Model evaluation
├── requirements.txt
├── .env.example
└── README.md
```

## Installation

1. Clone the repository:

```
git clone https://github.com/gmnayem631/disaster_db.git
cd disaster_db
```

2. Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Create a .env file and add your MongoDB Atlas connection string:

```
MONGO_URI=mongodb+srv://<username>:<password>@...
```

5. Download the spaCy model:

```
python -m spacy download en_core_web_sm
```

## Usage

- Run the full pipeline:

```
python pipeline.py
```

- Run individual components

```
python scraper.py
python entity_linker.py
```

> **Note:** The trained model and annotated dataset are not included in this repository due to size. You will need to train your own model using the provided scripts.

## Model Performance

### Entity-wise F1 Scores

| Entity Type       | F1 Score   |
| ----------------- | ---------- |
| BD_DISTRICT       | 94.23%     |
| DISASTER_TYPE     | 89.74%     |
| BD_UPAZILA        | 81.70%     |
| AGENCIES_INVOLVED | 76.60%     |
| DISPLACED         | 65.41%     |
| MISSING           | 64.00%     |
| FATALITIES        | 61.21%     |
| AFFECTED_PEOPLE   | 56.18%     |
| BD_UNION          | 51.72%     |
| RELIEF_INFO       | 47.48%     |
| **Overall F1**    | **79.21%** |

**Average processing time:** 2.549 seconds per article

## Model Comparison

| Model           | Dataset Size | Overall F1 |
| --------------- | ------------ | ---------- |
| CNN spaCy       | 72           | 60.47%     |
| DistilBERT      | 72           | 67.05%     |
| RoBERTa         | 220          | 78.16%     |
| DistilBERT      | 220          | 77.37%     |
| mBERT           | 220          | 77.52%     |
| **XLM-RoBERTa** | **220**      | **79.21%** |

## Limitations

- Currently supports English language only
- Prototype focuses mainly on flood disasters
- Publish date extraction fails for some sources
- Requires the machine to remain on for local scheduling
- Small training dataset limits performance on rare entities

## Future Work

- Support for Bangla language news
- Multi-disaster type coverage (cyclone, landslide, etc.)
- Cloud deployment (AWS/GCP) for 24/7 operation
- Larger annotated dataset
- Event deduplication and coreference resolution

**Built by** [Gulam Mustafa Nayem](https://github.com/gmnayem631/)  
BSc Final Semester Research Project  
[LinkedIn](https://linkedin.com/in/gulam-mustafa-nayem/)  
**Contributions and suggestions are welcome!**
