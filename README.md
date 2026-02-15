# Smartphone Usage & Productivity Analysis 📱

This project analyzes the relationship between smartphone usage and work productivity using a dataset of 50,000 records. It includes a machine learning model to predict productivity scores and a web dashboard built with **Flask**.

## Features

-   **Exploratory Data Analysis (EDA)**: Visualize correlations, distributions, and relationships between phone usage and productivity.
-   **Productivity Prediction**: A Random Forest model that predicts `Work_Productivity_Score` based on user inputs.
-   **Web Dashboard**: Flask-based web interface for easy interaction.

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

3.  **Run the Application**:
    ```bash
    python3 app.py
    ```
    The app will be available at `http://127.0.0.1:5000/`.

## Project Structure

-   `app.py`: The Flask application backend.
-   `templates/`: HTML templates (`index.html`, `predict.html`, `result.html`).
-   `static/`: CSS and other static assets.
-   `productivity_model.pkl`: Pre-trained model file.
-   `Smartphone_Usage_Productivity_Dataset_50000.csv`: The dataset used for training and analysis.
