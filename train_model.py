import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("data/jobs.csv")

# Function to convert salary text to numeric
def convert_salary(s):
    
    if pd.isna(s):
        return None
        
    s = s.replace("€","").replace(",","")
    
    if "-" in s:
        low, high = s.split("-")
        return (float(low) + float(high)) / 2
    else:
        return float(s)

# Create salary_numeric column
df["salary_numeric"] = df["salary"].apply(convert_salary)

# Remove missing values
df = df.dropna(subset=["job_title","salary_numeric"])

# Encode job title
X = pd.get_dummies(df["job_title"])

y = df["salary_numeric"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train model
model = RandomForestRegressor()

model.fit(X_train,y_train)

# Save model
joblib.dump(model,"salary_model.pkl")

print("Model trained and saved")