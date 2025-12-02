# Titanic Model - Lab Changes

## Overview
This project trains a Random Forest classifier to predict passenger survival on the Titanic dataset.

## Changes Made from Template

### 1. Dataset Change
- **Before:** Used the Iris dataset (`load_iris()` from sklearn)
- **After:** Used the Titanic dataset loaded from a public CSV URL

### 2. Feature Engineering
- Selected relevant features: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`
- Encoded categorical variable `Sex` using `LabelEncoder` (male/female → 0/1)
- Dropped rows with missing values using `dropna()`

### 3. Target Variable
- **Before:** Predicted iris flower species (3 classes)
- **After:** Predicted passenger survival (binary: 0 = died, 1 = survived)

### 4. Dependencies
Added `pandas` to `requirements.txt` for data loading and manipulation:

scikit-learn
joblib
pandas

### 5. Dockerfile
- Changed `COPY src/ .` to `COPY . .` to support flat directory structure
- Used `python:3.10-slim` base image for smaller container size
- Added `--no-cache-dir` flag to pip install

## How to Run
```bash
docker build -t titanic-model .
docker run titanic-model
```
