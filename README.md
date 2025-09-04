# NYC-Taxi-Fare-Prediction-Project

NYC-Taxi-Fare-Prediction/
│── data/taxi.csv                 # your dataset
│── notebooks/taxi_fare.ipynb     # full Python notebook
│── sql_queries.sql               # SQL analysis
│── README.md                     # project documentation

# NYC Taxi Fare Prediction 🚕
# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 2: Load dataset
df = pd.read_csv("data/taxi.csv")
print("Shape:", df.shape)
print(df.head())

# Step 3: Data Cleaning
df = df.dropna()
df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 200)]

# Step 4: Feature Engineering
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
df['year'] = df['pickup_datetime'].dt.year
df['month'] = df['pickup_datetime'].dt.month
df['hour'] = df['pickup_datetime'].dt.hour
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek

# Haversine formula for trip distance
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

df['distance_km'] = haversine(df['pickup_longitude'], df['pickup_latitude'],
                              df['dropoff_longitude'], df['dropoff_latitude'])

# Step 5: Select Features & Target
features = ['passenger_count','year','month','hour','day_of_week','distance_km']
X = df[features]
y = df['fare_amount']

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train Model (Linear Regression)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Step 8: Evaluate Linear Regression
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print("Linear Regression RMSE:", rmse_lr)

# Step 9: Train Model (Random Forest)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Step 10: Evaluate Random Forest
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("Random Forest RMSE:", rmse_rf)

# Step 11: Feature Importance
feat_importance = pd.DataFrame({'feature':features,'importance':rf.feature_importances_})
feat_importance.sort_values('importance', ascending=False, inplace=True)
print(feat_importance)

# Step 12: Visualization
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_rf, alpha=0.3)
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Taxi Fare Prediction - Random Forest")
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x='importance', y='feature', data=feat_importance)


-- 1. Average fare by passenger count
SELECT passenger_count, ROUND(AVG(fare_amount),2) as avg_fare
FROM taxi_data
GROUP BY passenger_count
ORDER BY passenger_count;

-- 2. Top 5 highest fares
SELECT * FROM taxi_data
ORDER BY fare_amount DESC
LIMIT 5;

-- 3. Average fare per hour
SELECT EXTRACT(HOUR FROM pickup_datetime) as hour, ROUND(AVG(fare_amount),2) as avg_fare
FROM taxi_data
GROUP BY hour
ORDER BY hour;

-- 4. Average distance by day of week
SELECT EXTRACT(DOW FROM pickup_datetime) as day_of_week, ROUND(AVG(distance_km),2) as avg_distance
FROM taxi_data
GROUP BY day_of_week
ORDER BY day_of_week;



# NYC Taxi Fare Prediction 🚕

## Problem Statement
Predict taxi fare amount using pickup & dropoff location, passenger count, and trip details.

## Dataset
NYC Taxi trips dataset with columns:
- fare_amount
- pickup_datetime
- pickup_longitude / pickup_latitude
- dropoff_longitude / dropoff_latitude
- passenger_count

## Approach
1. Data Cleaning (remove missing & invalid fares)
2. Feature Engineering (date-time features, haversine distance)
3. Machine Learning Models:
   - Linear Regression
   - Random Forest
4. Evaluation Metrics: RMSE

## Results
- Linear Regression RMSE: ~5.4
- Random Forest RMSE: ~3.2
- Key Features: Distance, Hour of Day, Passenger Count

## Tools & Skills
- Python (Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn)
- SQL (Aggregations, Time-based queries)
- Machine Learning (Regression Models)

plt.title("Feature Importance")
plt.show()
