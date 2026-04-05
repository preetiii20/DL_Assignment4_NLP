# NLP Preprocessing and Text Classification

## Overview

This project implements NLP preprocessing techniques and builds a multi-class text classification model using the HuffPost News Category Dataset. Three machine learning classifiers are trained and compared on their ability to predict the news category from article headlines and short descriptions.

---

## Dataset

- Source: HuffPost News Category Dataset (news.json)
- Format: JSON (line-delimited), converted to CSV
- Total records: 189,814
- Columns: link, headline, category, short_description, authors, date, text, clean_text, label
- Categories: 41 unique news categories (e.g. U.S. NEWS, COMEDY, PARENTING, POLITICS)

---

## Project Structure

```
DL_Practical4/
│
├── news.json                  Raw dataset (original JSON)
├── news_dataset.csv           Converted CSV from JSON
├── news_processed.csv         Final cleaned and preprocessed dataset
├── Notenook.ipynb             Main Jupyter notebook
└── README.md                  Project documentation
```

---

## Steps Performed

### Step 1 - Convert JSON to CSV
The raw JSON file is read using pandas and saved as a CSV file for easier handling.

### Step 2 - Verify CSV File
The CSV is reloaded and inspected using head(), shape, and columns to confirm successful conversion.

### Step 3 - Basic Cleaning
Rows with missing values in headline, short_description, and category columns are dropped. The index is reset after cleaning. Total rows after cleaning: 189,814.

### Step 4 - Create Main Text Column
A combined text column is created by concatenating the headline and short_description fields with a space separator.

### Step 5 - Save Final Processed Dataset
The cleaned dataframe is saved to news_processed.csv for use in subsequent steps.

### Step 6 - Text Preprocessing
A custom preprocessing function is applied to the text column. The following operations are performed in order:

- Lowercasing all text
- Removing numeric digits using regular expressions
- Removing punctuation using string.punctuation
- Tokenization using nltk word_tokenize
- Stopword removal using nltk English stopwords
- Lemmatization using nltk WordNetLemmatizer

The result is stored in a new column called clean_text.

### Step 7 - Encode Labels
The category column (string labels) is converted to integer labels using sklearn LabelEncoder. The encoded values are stored in a label column.

### Step 8 - Train Test Split
The dataset is split into training and test sets using an 80/20 ratio with random_state=42 for reproducibility.

- X: clean_text column
- y: label column

### Step 9 - TF-IDF Vectorization
Text is converted to numerical feature vectors using TfidfVectorizer with max_features=5000. The vectorizer is fit on the training set and used to transform both training and test sets.

### Step 10 - Train Models
Three classifiers are trained on the TF-IDF feature matrix:

- Naive Bayes: MultinomialNB (suitable for text data with TF-IDF counts)
- Logistic Regression: max_iter=1000 to ensure convergence across 41 classes
- Support Vector Machine: LinearSVC for efficient linear classification

### Step 11 - Evaluate Models
Accuracy scores on the test set:

| Model                | Accuracy  |
|---------------------|-----------|
| Naive Bayes          | 0.5320    |
| Logistic Regression  | 0.5934    |
| SVM (LinearSVC)      | 0.5888    |

### Step 12 - Detailed Report
Full classification reports (precision, recall, F1-score, support) are printed for each model using sklearn classification_report.

### Step 17 - Predict on Custom Text
A predict_text() function is defined that accepts a raw string, preprocesses it using the same pipeline, vectorizes it, and returns predictions from all three models.

Example predictions:

- "Government announces new policy" -> POLITICS (all three models)
- "Funny comedy movie released" -> ENTERTAINMENT (all three models)

---

## Libraries Used

| Library              | Purpose                              |
|---------------------|--------------------------------------|
| pandas               | Data loading, manipulation, CSV I/O  |
| nltk                 | Tokenization, stopwords, lemmatization |
| scikit-learn         | Vectorization, models, evaluation    |
| matplotlib           | Plotting (category distribution)     |
| re, string           | Text cleaning utilities              |

---

## How to Run

1. Install dependencies:
   ```
   pip install pandas scikit-learn nltk matplotlib
   ```

2. Download NLTK resources (run once inside Python):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

3. Place news.json in the project directory.

4. Open Notenook.ipynb in VS Code or Jupyter and run all cells in order from top to bottom.

---

## Notes

- Accuracy in the 53-59% range is expected for a 41-category classification problem with TF-IDF limited to 5000 features. The dataset is large and imbalanced across categories.
- The authors column has 32,955 missing values; it is not used as a feature.
- The predict_text function reuses the same preprocessing pipeline and vectorizer fitted during training, ensuring consistent results.

---

## Author

Student Name: [Your Name]
Roll Number: [Your Roll Number]
Subject: Deep Learning Practical
Submission Date: April 2026
