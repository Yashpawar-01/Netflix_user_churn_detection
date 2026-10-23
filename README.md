# Netflix Customer Churn Prediction

## 📌 Project Overview

Netflix Customer Churn Prediction is a Machine Learning project designed to predict whether a customer is likely to stop using a subscription service based on customer-related features.

The project uses a trained Machine Learning model and a Streamlit web interface to make predictions from user-provided customer information.

The goal is to demonstrate how Machine Learning can support customer retention strategies by identifying customers who may be at risk of leaving.

## 🎯 Objectives

- Predict customer churn using a trained Machine Learning model.
- Build an interactive web application using Streamlit.
- Allow users to enter customer information through a simple interface.
- Display churn predictions and probability when supported by the model.
- Demonstrate a practical end-to-end Machine Learning deployment workflow.

## ✨ Features

- Interactive Streamlit user interface.
- Customer information input form.
- Prediction using a previously trained model.
- Churn probability display when available.
- Customer dataset integration.
- Downloadable prediction results, if enabled in the application.
- Cloud deployment through Streamlit Community Cloud.

## 🛠️ Technologies Used

- **Python** — Main programming language.
- **Pandas** — Data manipulation and analysis.
- **NumPy** — Numerical operations.
- **Scikit-learn** — Machine Learning model support.
- **Joblib** — Loading saved model and feature files.
- **Streamlit** — Interactive web application.
- **Git and GitHub** — Version control and project hosting.

## 📂 Project Structure

```text
netflix-customer-churn-prediction/
│
├── app.py
├── netflix_churn_model.pkl
├── netflix_churn_features.pkl
├── netflix_customer_churn.csv
├── requirements.txt
└── README.md
```

### File Description

- `app.py` — Main Streamlit application.
- `netflix_churn_model.pkl` — Previously trained Machine Learning model.
- `netflix_churn_features.pkl` — Saved feature information used by the application.
- `netflix_customer_churn.csv` — Customer dataset used by the project.
- `requirements.txt` — Python dependencies required to run the application.
- `README.md` — Project documentation.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/netflix-customer-churn-prediction.git
```

### 2. Navigate to the project directory

```bash
cd netflix-customer-churn-prediction
```

### 3. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application should open in your browser.

## 🚀 Deployment

This application can be deployed using Streamlit Community Cloud.

1. Push the project files to a GitHub repository.
2. Sign in to Streamlit Community Cloud using GitHub.
3. Create a new app.
4. Select the repository, branch, and `app.py` as the main file.
5. Deploy the application.

## 📊 How It Works

1. The application loads the saved Machine Learning model and feature information.
2. The user enters customer details through the Streamlit interface.
3. The application prepares the input in the format expected by the model.
4. The trained model generates a churn prediction.
5. The result is displayed in the application, along with a probability when available.

## 🔮 Future Improvements

- Add exploratory data analysis and visualizations.
- Compare multiple Machine Learning algorithms.
- Display model evaluation metrics such as precision, recall, F1-score, and ROC-AUC.
- Improve the interface with customer analytics dashboards.
- Add model monitoring and retraining capabilities.
- Explore explainable AI to understand the factors influencing churn predictions.

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes. Predictions depend on the quality of the dataset, model training, and the correctness of the input features. Results should not be treated as guaranteed predictions of customer behavior.

## 👨‍💻 Author

**Yash Ananda Pawar**

Machine Learning | Data Science | Software Development

---

If you find this project useful, consider giving the repository a ⭐ on GitHub.
