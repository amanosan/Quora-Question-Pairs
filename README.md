# Quora Question Pair

## 1. Problem Statement

### 1.1 Description
Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

### 1.2 Problem Statement
* Identify which questions asked in Quora are duplicates of questions that have already been asked.
* This could be useful to instantly provide answers to the questions that have already been answered.
* ***The Task in hand is to predict whether a Pair of Questions are duplicate or not***

The source of the Problem: https://www.kaggle.com/c/quora-question-pairs

***

## 2. Machine Learning 

### 2.1 Data Overview
- The data given is in Train.csv file.
- Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate.
- ***Note: To get the final_features.csv and other csv files, run the EDA_FE.ipynb notebook.***

### 2.2 Type of Machine Learning Problem
This is a binary classification problem, for a given pair of questions we need to predict whether they are duplicate or not.

### 2.3 Performance Metrics
* Log-Loss
* Confusion Matrix
  
### 2.4 Modelling 
* Started with a Baseline Model which gave random probabilities
* Moved on to *Logistic Regression* with default Hyperparameters
* Tuned *Logistic Regression* with Hyperopt.
* Moved on to *SVM* with the default Hyperparameters.
* Moved on to *XGboost* and tuned with Hyperopt
* Moved on to *RandomForest* and tuned with Hyperopt.
  
**Each and every metric along with the model were logged using MLFlow.**

**Found XGBoost after Hyperparameter Tuning to be performing the best.**
***
