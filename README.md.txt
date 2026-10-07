# Fake News Prediction using Logistic Regression

## Project Overview

This project detects whether a news article is **Fake News** or **True News** using Natural Language Processing (NLP) and Machine Learning.

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert news text into numerical features and **Logistic Regression** for classification.

The dataset contains separate collections of fake and true news articles.

## Technologies Used

* Python
* NumPy
* Pandas
* Regular Expressions (re)
* NLTK
* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression
* Google Colab

## Dataset

The project uses the **Fake and Real News Dataset**, containing two files:

* `Fake.csv` — Fake news articles
* `True.csv` — True news articles

The datasets contain information such as:

* News title
* News text
* Subject
* Date

For this project, the **news text** is used as the main input feature.

## Data Preparation

The fake and true news datasets were combined into a single dataset.

Labels were assigned as follows:

* `0` → Fake News
* `1` → True News

The `subject` and `date` columns were removed because they were not required for the classification task.

The dataset was then shuffled before splitting it into training and testing sets.

## Machine Learning Pipeline

The project follows this workflow:

```text
News Dataset
     ↓
Data Cleaning
     ↓
Combine Fake + True News
     ↓
Assign Labels
     ↓
Shuffle Dataset
     ↓
Train/Test Split
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Prediction
     ↓
Evaluation
```

## Train/Test Split

The dataset was divided into:

* **80% Training Data**
* **20% Testing Data**

Stratified splitting was used to maintain the class distribution between the training and testing sets.

## TF-IDF Vectorization

TF-IDF was used to convert the news text into numerical feature vectors that can be processed by the machine learning model.

The training data was used to fit the TF-IDF vectorizer, while the same fitted vectorizer was used to transform the test data.

## Machine Learning Model

The classification model used in this project is:

**Logistic Regression**

Logistic Regression is a commonly used classification algorithm that predicts the probability of an input belonging to a particular class.

In this project:

```text
0 → Fake News
1 → True News
```

## Results

### Training Accuracy

**99.19%**

### Test Accuracy

**98.61%**

The model achieved high accuracy on both the training and testing datasets, indicating strong classification performance on this dataset.

## Classification Report

The model achieved approximately **99% precision, recall, and F1-score** for both classes.

```text
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      4696
           1       0.99      0.99      0.99      4284

    accuracy                           0.99      8980
   macro avg       0.99      0.99      0.99      8980
weighted avg       0.99      0.99      0.99      8980
```

## Confusion Matrix

```text
[[4634   62]
 [  50 4234]]
```

This means:

* **4,634** fake news articles were correctly classified as fake.
* **62** fake news articles were classified as true.
* **50** true news articles were classified as fake.
* **4,234** true news articles were correctly classified as true.

## Custom News Prediction

A custom prediction function was implemented to classify new news articles.

Example output:

```text
Actual Label: Fake News
Predicted Label: Fake News
```

The model can therefore be used to test new news text after training.

## Project Structure

```text
Fake News Prediction Logistic Regression/
├── README.md
└── fake_news_prediction_logistic_regression.ipynb
```

## How to Run

1. Clone or download this repository.
2. Open `fake_news_prediction_logistic_regression.ipynb`.
3. Open the notebook in **Google Colab** or **Jupyter Notebook**.
4. Run the cells sequentially.
5. The dataset will be downloaded during the notebook execution.
6. Train the model and test predictions.

## Author

**Arafat Islam**

BSc in Electronics and Telecommunication Engineering
Chittagong University of Engineering and Technology (CUET)
