# 🌍 Air Quality Classification App 🍃

> **An interactive end-to-end Machine Learning web application to predict and classify air quality levels based on environmental and demographic features.**
> _Built as a Machine Learning university course project! 🚀_

## 📖 Project Overview

Monitoring air quality is crucial for public health and environmental planning. This project utilizes a dataset of 5000 records containing various environmental metrics (pollutant concentrations, temperature, humidity) to build a robust classification model.

The project is divided into two main parts:

1. **Model Development (`ass.ipynb`):** A comprehensive Jupyter Notebook covering data exploration, preprocessing, and the training of the classification model.
2. **Web Deployment (`deploy.py`):** A dynamic, multi-page Streamlit application that allows users to interact with the data, visualize insights, and test the model in real-time.

## ✨ Key Features (Streamlit Dashboard)

- **🏠 Home:** Project introduction and dataset overview.
- **📊 Exploratory Data Analysis (EDA):** Interactive charts (powered by Plotly) to uncover data distributions and feature correlations.
- **🛠️ Preprocessing:** Options to handle data scaling (StandardScaler, MinMaxScaler, RobustScaler) and split the dataset.
- **⚙️ Model:** Train various algorithms on the fly, including **Logistic Regression, Random Forest, and XGBoost**.
- **📈 Evaluation:** Visual performance metrics including Classification Reports, Confusion Matrices, and ROC-AUC curves.
- **🧪 Testing:** A user-friendly interface to input custom environmental parameters and instantly get an air quality prediction.

## 🛠️ Tech Stack

- **Python 3** 🐍
- **Streamlit** 👑 (For the interactive web interface and custom UI styling)
- **Scikit-Learn & XGBoost** 🤖 (For machine learning pipelines and algorithms)
- **Pandas & NumPy** 📊 (For data manipulation)
- **Plotly** 📈 (For beautiful, interactive data visualizations)

## 🚀 How to Run Locally

Follow these steps to run the Streamlit dashboard on your local machine:

### 1. Clone the Repository

```bash
git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
cd your-repo-name
```
