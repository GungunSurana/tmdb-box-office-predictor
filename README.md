# TMDB Box Office Revenue Prediction 🎬
This project predicts movie box office revenue using metadata from The Movie Database (TMDB). 
The goal is to build and compare machine learning models that estimate revenue based on features 
such as budget, popularity, votes, genres, and release date.

## Problem Statement
Given movie attributes, predict the box office revenue for movies in the test dataset.
Model performance is evaluated using Root Mean Squared Error (RMSE).

## Dataset
The dataset is provided by Kaggle and includes:
- Budget
- Runtime
- Genres
- Popularity
- Vote average & vote count
- Release date

The target variable is **revenue**.

## Approach
1. Data preprocessing and feature engineering
2. Handling missing values using median imputation
3. Log transformation of revenue to handle skewness
4. Training and comparing multiple regression models
5. Selecting the best model based on RMSE
6. Generating predictions in Kaggle submission format

## Models Used
- Linear Regression (baseline)
- Random Forest Regressor
- XGBoost Regressor (best performance)

## Evaluation Metric
- Root Mean Squared Error (RMSE)

## Files
- `tmdb_box_office_prediction.ipynb` – Main notebook with full pipeline
- `requirements.txt` – Python dependencies

## Results
The final model predicts revenue for unseen movies and generates a submission file 
in the required Kaggle format.

## Contact
**Gungun**  GitHub:GungunSurana
