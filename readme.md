# 🌍 Air Quality Classification App 🍃

> **An interactive end-to-end Machine Learning web application to predict and classify air quality levels based on environmental and demographic features.**
> _Built as a Machine Learning university course project! 🚀_
>
> **Status:** 🏁 Completed
> **Project Type:** Group Project

## 📖 Project Overview

This project provides a comprehensive solution for monitoring and predicting air quality. Using a dataset of 5,000 environmental records, I developed a machine learning pipeline that covers everything from raw data exploration to a fully functional web deployment.

The application allows users to visualize environmental trends, train multiple classification models (Logistic Regression, Random Forest, XGBoost), and perform real-time testing with custom inputs.

## ✨ Key Features

- **📊 Interactive EDA:** Deep dive into pollutant distributions and correlations using Plotly.
- **🛠️ Flexible Preprocessing:** Toggle between different scalers (Standard, MinMax, Robust) to see how they affect model performance.
- **⚙️ Dynamic Training:** Train and compare different algorithms directly from the UI.
- **📈 Comprehensive Evaluation:** View detailed metrics including Confusion Matrices and ROC-AUC curves.
- **🧪 Real-time Prediction:** Input custom parameters to instantly classify air quality levels.

## 🤝 My Role & Contributions

In this group project, I was responsible for the core data science lifecycle, handling everything from **Exploratory Data Analysis (EDA) through to Model Evaluation**. My specific contributions include:
- **Exploratory Data Analysis (EDA):** Analyzing feature distributions, uncovering hidden patterns, and building interactive visualizations to understand environmental correlations.
- **Data Preprocessing:** Engineering features and applying various scaling techniques (Standard, MinMax, Robust) to prepare the data for training.
- **Model Training:** Designing and training multiple classification algorithms (Logistic Regression, Random Forest, XGBoost) to find the best-performing model.
- **Model Evaluation:** Conducting rigorous testing using Classification Reports, Confusion Matrices, and ROC-AUC curves to ensure accurate and reliable air quality predictions.

## 🛠️ Tech Stack

- **Python 3** 🐍
- **Streamlit** 👑 (Web Framework & UI)
- **Scikit-Learn & XGBoost** 🤖 (ML Models)
- **Pandas & NumPy** 📊 (Data Processing)
- **Plotly** 📈 (Interactive Visualizations)

## 📂 Project Structure

```text
├── air_quality.csv             # Raw air quality data
├── ass.ipynb            # Jupyter Notebook for EDA and model development
├── deploy.py            # Main Streamlit application script
├── model.pkl            # Pre-trained serialized classification model
├── requirements.txt     # List of Python dependencies required to run the app
└── README.md            # Project documentation
```

## 🚀 How to Run Locally

1. Clone the Repository

```bash
git clone [https://github.com/yosukeyung/Air-Quality-Classification-App.git](https://github.com/yosukeyung/Air-Quality-Classification-App.git)
cd Air-Quality-Classification-App
```

2. Install Dependencies

```bash
pip install -r requirements.txt
```

3. Run the App

```bash
streamlit run deploy.py
```

## 👨‍💻 Author

Yosuke Yung
_CS Student @ BINUS UNIVERSITY_
