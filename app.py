from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import io
import base64

app = Flask(__name__)

# Load data and model
try:
    df = pd.read_csv('Smartphone_Usage_Productivity_Dataset_50000.csv')
    model = joblib.load('productivity_model.pkl')
    feature_metadata = joblib.load('feature_metadata.pkl')
except FileNotFoundError:
    print("Error: Model or dataset not found. Please run train_model.py first.")
    df = None
    model = None
    feature_metadata = None

def get_img_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    if df is None:
        return "Error: Dataset not found."
    
    # Dataset Overview
    description = df.describe().to_html(classes='table table-striped', border=0)
    
    # 1. Correlation Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax1)
    heatmap_url = get_img_base64(fig1)
    plt.close(fig1)
    
    # 2. Productivity Score Distribution
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.histplot(df['Work_Productivity_Score'], bins=20, kde=True, ax=ax2)
    ax2.set_title("Productivity Score Distribution")
    dist_url = get_img_base64(fig2)
    plt.close(fig2)
    
    # 3. Usage vs Productivity (Sampled)
    sample_df = df.sample(min(1000, len(df)))
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=sample_df, x='Daily_Phone_Hours', y='Work_Productivity_Score', hue='Gender', alpha=0.6, ax=ax3)
    ax3.set_title("Daily Phone Hours vs Productivity (Sample)")
    scatter_url = get_img_base64(fig3)
    plt.close(fig3)

    return render_template('index.html', 
                           description=description,
                           heatmap_url=heatmap_url,
                           dist_url=dist_url,
                           scatter_url=scatter_url)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if model is None:
            return "Error: Model not loaded."
        
        # Extract features from form
        input_data = {}
        for col in feature_metadata['numerical']:
            input_data[col] = float(request.form.get(col))
        for col in feature_metadata['categorical']:
            input_data[col] = request.form.get(col)
            
        input_df = pd.DataFrame([input_data])
        
        # Predict
        prediction = model.predict(input_df)[0]
        
        # Interpretation
        if prediction > 7:
            msg = "High productivity! Keep up the good work."
            msg_class = "success"
        elif prediction > 4:
            msg = "Moderate productivity. Consider managing your screen time."
            msg_class = "warning"
        else:
            msg = "Low productivity. You might want to reduce distractions."
            msg_class = "danger"
            
        return render_template('result.html', prediction=round(prediction, 2), msg=msg, msg_class=msg_class)
    
    return render_template('predict.html', feature_metadata=feature_metadata)

if __name__ == '__main__':
    app.run(debug=True)
