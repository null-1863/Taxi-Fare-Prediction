# Taxi Fare Prediction

This project is focused on predicting the fare amount for taxi rides based on factors such as pickup and dropoff locations, the number of passengers, and time of day. The model is trained using historical data and applies machine learning techniques for accurate predictions.

# Table of Contents
**1. Project Overview**

**2. Dataset**

**3. Features**

**4. Modeling Approach**

**5. Installation**

**6. Usage**

**7. Evaluation Metrics**

**8. Results**

**9. Contributing**

**10. License**

# Project Overview

The goal of this project is to predict taxi fares in New York City using machine learning algorithms. Given a dataset containing ride information like pickup and dropoff coordinates, passenger count, and timestamps, we build a regression model that forecasts the fare amount for a given ride.

# Key Objectives:
* Clean and preprocess the data.
* Extract meaningful features such as distance between locations and date-time-based variables.
* Train and evaluate machine learning models for fare prediction.
* Visualize the results and interpret model performance.

# Dataset

* The dataset contains the following columns:
* fare_amount: The fare paid for the ride.
* pickup_datetime: The date and time when the ride was initiated.
* pickup_longitude, pickup_latitude: The geographical coordinates of the pickup location.
* dropoff_longitude, dropoff_latitude: The geographical coordinates of the dropoff location.
* passenger_count: The number of passengers in the taxi ride.

The data includes multiple rows of historical taxi trips in New York City. You can download the dataset from public repositories 
 **https://www.kaggle.com/code/dster/nyc-taxi-fare-starter-kernel-simple-linear-model/input?select=train.csv**.

# Features
The key features used in the model include:

* Pickup and Dropoff Coordinates: Longitude and latitude used to calculate the distance between two points using the Haversine formula.
* Distance: The calculated distance between pickup and dropoff locations.
* Time of Day: Extracted hour from pickup_datetime to understand the effect of peak and off-peak times.
* Day of the Week: Extracted from pickup_datetime to analyze weekday vs weekend effects.
* Passenger Count: Number of passengers as a predictor of shared rides.

# Modeling Approach
We implemented the following steps:

1. Exploratory Data Analysis (EDA):

    * Visualized fare distribution, missing data, and outliers.
    * Analyzed feature correlations.

2. Data Preprocessing:

    * Cleaned missing values.
    *   Removed outliers (e.g., negative fares, extreme fare values).
    * Converted pickup_datetime to a proper datetime format and extracted relevant features.

3. Feature Engineering:

    * Created new features like distance, hour, and day_of_week to improve model accuracy.

4. Model Training:

    * Used a Random Forest Regressor to train the model on the cleaned dataset.
    * Splitted the data into training (80%) and test (20%) sets.
    * Fine-tuned hyperparameters using cross-validation.
      
5. Model Evaluation:

    * Predicted fare amounts on the test set and evaluated performance using RMSE and MAE.

# Installation

**Prerequisites**
  * Python 3.x
  * Jupyter Notebook (or Jupyter Lab)
  * Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

**Setup Instructions**

  * Clone the repository:
    
        git clone https://github.com/null-1863/TaxiFarePrediction.git
    
  * Navigate to the project directory:
    
        cd TaxiFarePrediction
    
  * Install the required dependencies:
    
        pip install -r requirements.txt

  * Run the Jupyter notebook:
    
        jupyter notebook TaxiFare_prediction.ipynb

# Usage

  * Load the dataset using:
    
        df = pd.read_csv('path_to_your_data.csv')

  * Preprocess and clean the data:
      * Remove outliers and handle missing values.
      * Feature engineering.

  * Train the model:
    
        model.fit(X_train, y_train)
    
  * Predict fare amounts for test data:
    
        y_pred = model.predict(X_test)

  * Evaluate the model's performance:

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Evaluation Metrics

We used the following metrics to evaluate model performance:

  * RMSE (Root Mean Squared Error): Measures the average magnitude of the error between the predicted and actual fare amounts. A lower RMSE indicates better model performance.

  * R² Score: tells you how well the independent variables explain the variability in the dependent variable:

      * High R² (close to 1): The model explains most of the variance in the target variable.

      * Low R² (close to 0): The model does not explain much of the variance.

# Results

After training the XGBoost Regression model and tuning the hyperparameters, we achieved the following results on the test set:

RMSE: 3.98

R2 Score: 0.82

A scatter plot of actual vs predicted fares shows a reasonably close fit between the two, with most of the points falling along the ideal diagonal line.

# Contributing

Contributions are welcome! If you would like to contribute to this project:

  * Fork the repository.
  * Create a feature branch (git checkout -b feature-branch).
  * Commit your changes (git commit -m 'Add a feature').
  * Push to the branch (git push origin feature-branch).
  * Open a pull request.

# License

This project is licensed under the MIT License - see the LICENSE file for details.
