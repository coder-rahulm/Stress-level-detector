import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('Smartphone_Usage_Productivity_Dataset_50000.csv')

# Preprocessing
# Target variable: Work_Productivity_Score
# Features: Age, Gender, Occupation, Device_Type, Daily_Phone_Hours, Social_Media_Hours, Sleep_Hours, Stress_Level, App_Usage_Count, Caffeine_Intake_Cups, Weekend_Screen_Time_Hours

# Drop obviously non-predictive columns like User_ID
X = df.drop(columns=['User_ID', 'Work_Productivity_Score'])
y = df['Work_Productivity_Score']

# Identify categorical and numerical columns
categorical_cols = ['Gender', 'Occupation', 'Device_Type']
numerical_cols = ['Age', 'Daily_Phone_Hours', 'Social_Media_Hours', 'Sleep_Hours', 'Stress_Level', 'App_Usage_Count', 'Caffeine_Intake_Cups', 'Weekend_Screen_Time_Hours']

# Create preprocessing pipeline
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create the model pipeline
# Optimized for file size to stay under GitHub's 100MB limit
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42))])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training model...")
model.fit(X_train, y_train)

# Evaluate the model
print("Evaluating model...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Save the model
print("Saving model...")
joblib.dump(model, 'productivity_model.pkl')
print("Model saved as productivity_model.pkl")

# Save feature names for the app if needed (though pipeline handles it mostly)
# We might need them for Streamlit input generation
feature_names = {
    'numerical': numerical_cols,
    'categorical': categorical_cols,
    'categorical_options': {col: df[col].unique().tolist() for col in categorical_cols}
}
joblib.dump(feature_names, 'feature_metadata.pkl')
print("Feature metadata saved as feature_metadata.pkl")
