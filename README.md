# CS 135 Project 01: Logistic Regression for Image Classification

## Overview
This project explores the application of logistic regression to classify images of handwritten digits (specifically the digits 8 and 9) and clothing items (trousers vs. dresses). The analysis includes preprocessing, model fitting, hyperparameter tuning, and evaluation using various metrics.

## Mathematical Models and Functions

### Logistic Regression
Logistic Regression is used as the primary classifier in this project. It employs a logistic function to model a binary dependent variable, predicting the probability that a given input image belongs to one of two classes.

#### Logistic Function
The logistic function, also known as the sigmoid function, is defined as:


σ(z) = 1 / (1 + e^(-z))

where z is the linear combination of features and weights in logistic regression.

### Model Fitting
We use the `LogisticRegression` class from `sklearn.linear_model`, applying the `liblinear` solver. The logistic regression model is fit to the training data using different maximum iteration parameters to explore the convergence behavior of the solver.

### Regularization
To prevent overfitting and enhance the model's generalization capabilities, regularization techniques are applied:
- **L1 Regularization (Lasso Regression):** Tends to shrink certain coefficients to zero, effectively performing feature selection.
- **L2 Regularization (Ridge Regression):** Shrinks the coefficients evenly but does not set them to zero.
- **Elastic-net Regularization:** A combination of L1 and L2 regularization, controlled by the mixing parameter `l1_ratio`.

### Hyperparameters
- **C (Inverse of regularization strength):** Smaller values specify stronger regularization.
- **max_iter:** Specifies the maximum number of iterations for the solvers to converge.
- **l1_ratio:** The Elastic-net mixing parameter (only applicable if penalty='elasticnet').

### Model Evaluation
Model performance is evaluated using accuracy, log loss, and a confusion matrix. Log loss provides a measure of accuracy where the prediction input is a probability value between 0 and 1. We use `sklearn.metrics.log_loss` for this purpose.

### Feature Transformations
Various feature transformations are explored to enhance model performance:
- **Preprocessing:** Conversion to lowercase, removal of punctuation and numbers, and exclusion of stopwords and alphanumeric words.
- **Dimensionality Reduction:** Using techniques like PCA (Principal Component Analysis) to reduce the number of features.
- **Noise Reduction:** Techniques like blurring are applied to the image data to reduce overfitting.

### Plots and Visualizations
Various plots are generated to visualize:
- The effects of different `max_iter` values on model accuracy and log loss.
- The impact of different regularization strengths and the evolution of feature weights.
- Misclassifications through sample images of false positives and negatives.
