# Smartphone Usage & Productivity Analysis 📱

This project analyzes the relationship between smartphone usage and work productivity using a dataset of 50,000 records. It includes a machine learning model to predict productivity scores and provides two web interface options: **Flask** and **Streamlit**.

## Features

-   **Exploratory Data Analysis (EDA)**: Visualize correlations, distributions, and relationships between phone usage and productivity.
-   **Productivity Prediction**: A Random Forest model that predicts `Work_Productivity_Score` based on user inputs.
-   **Web Dashboards**: Choose between a Flask-based or Streamlit-based web interface.

## Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/coder-rahulm/Stress-level-detector.git
    cd Stress-level-detector
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

### Option 1: Flask (Recommended)
Calculates and renders visualizations on the backend.
```bash
python3 app.py
```
Access at: `http://127.0.0.1:5000/`

### Option 2: Streamlit
Interactive dashboard with built-in widget support.
```bash
streamlit run streamlit_app.py
```
Access at: `http://localhost:8501/`

## Project Structure

-   `app.py`: Flask application backend.
-   `streamlit_app.py`: Streamlit application.
-   `templates/`: HTML templates for Flask.
-   `static/`: CSS for Flask.
-   `productivity_model.pkl`: Pre-trained model file.
-   `Smartphone_Usage_Productivity_Dataset_50000.csv`: Dataset.
