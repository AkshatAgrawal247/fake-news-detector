# Fake News Detection

A machine learning project that uses various classification algorithms to detect fake news articles based on their content. The project implements and compares multiple models including Logistic Regression, Random Forest, XGBoost, and Support Vector Machine (SVM).

## Overview

This project aims to tackle the growing problem of fake news by implementing an automated detection system using machine learning techniques. The system analyzes the content of news articles and classifies them as either genuine or fake.

## Dataset

The dataset used in this project is from Kaggle's "Fake News Detection Dataset":

- Source: [Fake News Detection Datasets](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
- Contains two files:
  - `True.csv`: Collection of genuine news articles
  - `Fake.csv`: Collection of fake news articles

## Requirements

### Python Version
- Python 3.8 or higher

### Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.2
xgboost>=1.4.2
nltk>=3.6.3
matplotlib>=3.4.3
jupyter>=1.0.0
ipykernel>=6.0.0
```

## Setup Instructions

1. **Create a Virtual Environment (Recommended)**
   ```bash
   # Using venv
   python -m venv venv

   # Activate the virtual environment
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK Data**
   ```python
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
   ```

4. **Download the Dataset**
   - Go to [Fake News Detection Datasets](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
   - Download `True.csv` and `Fake.csv`
   - Place both files in the project root directory

## Running the Project

1. **Run the Notebook in Google Colab**

   - Open fake_news_detector.ipynb in Colab

   - Execute cells step by step (Shift+Enter) or run all at once

2. **View Results**
   - The notebook will display:
     - Model training progress
     - Accuracy comparisons
     - Confusion matrices
     - Classification reports

## Project Structure

```
├── fake_news_detector.ipynb    # Main collab notebook with all the code
├── True.csv                   # Dataset with real news articles
├── Fake.csv                   # Dataset with fake news articles
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Implementation Steps

1. **Data Loading and Preprocessing**
   - Loading the true and fake news datasets
   - Combining title and text content
   - Text cleaning and normalization
   - Stop word removal and lemmatization

2. **Feature Engineering**
   - TF-IDF vectorization
   - Feature selection (top 10,000 features)

3. **Model Training and Evaluation**
   - Data splitting (70% training, 30% testing)
   - Training multiple classification models
   - Performance evaluation using various metrics
   - Visualization of results

## Model Performance

The project includes visualization of:
- Accuracy comparison between different models
- Confusion matrices for each model
- Detailed classification reports including precision, recall, and F1-score

## Troubleshooting

Common issues and solutions:
1. **NLTK Data Missing**
   - Error: `Resource punkt not found`
   - Solution: Run the NLTK download command mentioned in Setup Instructions

2. **Memory Issues**
   - Error: `MemoryError` during model training
   - Solution: Reduce `max_features` in TfidfVectorizer or use a smaller sample size


## Future Improvements

- Implementation of deep learning models
- Feature importance analysis
- Hyperparameter tuning
- Real-time news article classification
- Web interface for user interaction

## Author

Akshat Agrawal

## Acknowledgments

- Dataset provided by Kaggle https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets
- NLTK community for text processing tools
- Scikit-learn community for machine learning implementations

