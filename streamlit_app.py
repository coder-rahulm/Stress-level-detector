import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Smartphone Usage & Productivity", layout="wide")

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv('Smartphone_Usage_Productivity_Dataset_50000.csv')
    return df

@st.cache_resource
def load_model():
    model = joblib.load('productivity_model.pkl')
    feature_metadata = joblib.load('feature_metadata.pkl')
    return model, feature_metadata

try:
    df = load_data()
    model, feature_metadata = load_model()
except FileNotFoundError:
    st.error("Model or dataset not found. Please run train_model.py first.")
    st.stop()

# Title and sidebar
st.title("📱 Smartphone Usage & Productivity Analysis")
st.markdown("Analyze how smartphone usage impacts work productivity and predict your own score.")

st.sidebar.header("User Input")

# Prediction Function
def user_input_features():
    inputs = {}
    
    # Numerical features
    for col in feature_metadata['numerical']:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        
        # Adjust step based on range
        step = 1.0 if col in ['Age', 'App_Usage_Count', 'Caffeine_Intake_Cups'] else 0.1
        
        inputs[col] = st.sidebar.slider(f"{col}", min_val, max_val, mean_val, step=step)
        
    # Categorical features
    for col in feature_metadata['categorical']:
        options = feature_metadata['categorical_options'][col]
        inputs[col] = st.sidebar.selectbox(f"{col}", options)
        
    return pd.DataFrame(inputs, index=[0])

# Main Layout
tab1, tab2 = st.tabs(["📊 Exploratory Data Analysis", "🤖 Prediction Model"])

with tab1:
    st.header("Dataset Overview")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Correlation Heatmap")
        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.subheader("Productivity Score Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['Work_Productivity_Score'], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    st.subheader("Usage vs Productivity")
    # Sample data for scatter plot to avoid performance issues
    sample_df = df.sample(min(5000, len(df)))
    fig, ax = plt.subplots()
    sns.scatterplot(data=sample_df, x='Daily_Phone_Hours', y='Work_Productivity_Score', hue='Gender', alpha=0.5, ax=ax)
    st.pyplot(fig)

with tab2:
    st.header("Predict Your Productivity Score")
    
    input_df = user_input_features()
    
    st.subheader("Your Input Parameters")
    st.write(input_df)
    
    if st.button("Predict Productivity Score"):
        prediction = model.predict(input_df)
        st.success(f"Predicted Work Productivity Score: {prediction[0]:.2f} / 10")
        
        # Interpretation
        if prediction[0] > 7:
            st.write("🎉 High productivity! Keep up the good work.")
        elif prediction[0] > 4:
            st.write("⚖️ Moderate productivity. Consider managing your screen time.")
        else:
            st.write("⚠️ Low productivity. You might want to reduce distractions.")
