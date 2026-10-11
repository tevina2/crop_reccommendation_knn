# Precision Crop Recommendation System (KNN Classifier)

An agricultural decision-support tool that predicts optimal crop cultivation based on soil nutrient composition and agro-climatic conditions.

## Overview
Selecting crops without aligning with specific soil nutrients and weather conditions leads to poor yields and resource waste. This project implements a **K-Nearest Neighbors (KNN)** multi-class classifier using Scikit-Learn to automate crop selection based on precise soil and atmospheric measurements.

## Dataset Features
- **N, P, K**: Nitrogen, Phosphorus, and Potassium ratios in soil
- **Temperature**: Ambient temperature (°C)
- **Humidity**: Relative atmospheric humidity (%)
- **pH**: Soil acidity/alkalinity level
- **Rainfall**: Seasonal precipitation (mm)
- **Target Label**: 22 unique agricultural crop varieties (e.g., rice, maize, coffee, chickpea)

## Implementation Steps
1. **Data Preprocessing & Scaling**: Standardized continuous features using `StandardScaler` to ensure balanced Euclidean distance calculations.
2. **Train/Test Splitting**: Stratified 80/20 train-test partition across all crop classes.
3. **Model Training**: Trained a baseline KNN classifier ($K=5$).
4. **Hyperparameter Tuning**: Iterated over values of $K \in [1, 20]$ to identify the optimal number of neighbors.
5. **Evaluation**: Evaluated performance using accuracy, classification report (precision, recall, F1-score), and confusion analysis.

## Key Results
- **Model Accuracy**: ~98% on test data
- **Hyperparameter Insights**: Peak accuracy observed around $K = 5$

## Project Structure
- `crop_knn.py`: Full end-to-end Python pipeline (preprocessing, model training, evaluation, and inference).
- `k_tuning_plot.png`: Visualization showing test accuracy across different values of $K$.
-
