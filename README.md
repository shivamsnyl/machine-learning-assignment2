# Machine Learning Assignment 2

## (a)Problem Statement

The goal is to predict whether a person's annual income exceeds $50,000 using demographic and employment-related attributes.

Target Variable:
- <=50K
- >50K

This is a binary classification problem.

---

## (b)Dataset Description

Dataset Name:
Adult Income Dataset

Source:
UCI Machine Learning Repository

Dataset Characteristics:
- Instances: 48,842
- Features: 14
- Type: Classification
- Missing Values: Yes

Features:
- age
- workclass
- fnlwgt
- education
- education-num
- marital-status
- occupation
- relationship
- race
- sex
- capital-gain
- capital-loss
- hours-per-week
- native-country

Target:
- income

Classes:
- <=50K
- >50K

---

## (c)Github Repository Link

https://github.com/shivamsnyl/machine-learning-assignment2.git

---

## (d)Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

---

## Evaluation Metrics

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Logistic Regression | 0.8268079226163059|0.8547718784176969|0.7226720647773279|0.45535714285714285|0.5586854460093896|0.47666657616570984|

| Decision Tree |0.8085367726086289|0.738680782|0.6020343293070566|0.6039540816326531|0.6029926774912449|0.4768434143990752|

| KNN |0.8263473053892215|0.8471032763459276|0.6610169491525424|0.5720663265306123|0.6133333333333333|0.5042946508231155|

| Naive Bayes |0.7921080915092891|0.829811085|0.6515580736543909|0.29336734693877553|0.4045734388742304|0.3350356951322123|

| SVM |0.8472286196837095|0.8863396597263781|0.7519788918205804|0.545280612|0.6321626617375231|0.549909868|

---

## Observations

### Logistic Regression
Provides strong baseline performance and good generalization.

### Decision Tree
Easy to interpret but may overfit the training data.

### KNN
Produces competitive performance but prediction time is high.

### Naive Bayes
Very fast training and prediction but assumes feature independence.

### SVM
Provides excellent class separation and robust performance.

### Overall Winner
SVM
