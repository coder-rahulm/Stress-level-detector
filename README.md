# Smartphone Usage & Productivity Analysis 📱

This project analyzes the relationship between smartphone usage and work productivity using a dataset of 50,000 records. It includes a machine learning model to predict productivity scores and an interactive Streamlit dashboard for visualization and user predictions.

## Features

-   **Exploratory Data Analysis (EDA)**: Visualize correlations, distributions, and relationships between phone usage and productivity.
-   **Productivity Prediction**: A Random Forest model that predicts `Work_Productivity_Score` based on user inputs like age, gender, daily phone hours, and more.
-   **Interactive Dashboard**: Built with Streamlit for easy interaction.

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
    streamlit run app.py
    ```

## Project Structure

-   `app.py`: The Streamlit application source code.
-   `train_model.py`: Script used to train the model (optional to run, as model is included).
-   `productivity_model.pkl`: Pre-trained model file.
-   `requirements.txt`: List of Python dependencies.
-   `Smartphone_Usage_Productivity_Dataset_50000.csv`: The dataset used for training and analysis.
