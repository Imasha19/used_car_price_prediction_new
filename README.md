#🚗 Used Car Price Prediction

     A Machine Learning web application built with Python and Streamlit to predict the price of a used car based on key features like mileage, brand, year, fuel type, engine capacity, and more.

🔗 Live Demo

👉 Streamlit App:
https://imasha19-used-car-price-prediction-new-app-o3fn8o.streamlit.app/

👉 GitHub Repository:
https://github.com/Imasha19/used_car_price_prediction_new

📌 Project Overview

     This project aims to build an accurate ML model that predicts the selling price of a used car.
     It uses a supervised learning approach and analyzes various attributes such as:

Brand & Model

Year of Manufacture

Mileage Driven

Fuel Type

Transmission

Engine Capacity

Number of Owners

The application is deployed on Streamlit Cloud for easy access and user interaction.

🧠 Machine Learning Workflow
1. Data Preprocessing

Handling missing values

Removing duplicates

Encoding categorical variables

Feature scaling (if required)

2. Exploratory Data Analysis (EDA)

Distribution analysis

Correlation heatmaps

Outlier detection

3. Model Building

Algorithms used (depending on your implementation):

Linear Regression

Random Forest Regressor

Gradient Boosting

XGBoost

4. Model Evaluation

R² Score

MAE

RMSE

The best-performing model was selected and saved using pickle for deployment.

🎯 Features of the Web App

User-friendly Streamlit UI

Real-time price prediction

Input form for all vehicle details

Clean visualization 

Displays predicted market value instantly

| Component  | Technology                  |
| ---------- | --------------------------- |
| Frontend   | Streamlit                   |
| Backend    | Python                      |
| ML         | scikit-learn, pandas, numpy |
| Deployment | Streamlit Cloud             |


🚀 How to Run the Project Locally

# Clone the repository
git clone https://github.com/Imasha19/used_car_price_prediction_new

# Navigate to the folder
cd used_car_price_prediction_new

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py






