# UTS_ML-Prediksi-Harga-Mobil-Bekas

## 🚗 Used Car Price Prediction App

A **Machine Learning web app** built with **Streamlit** that predicts **used car prices** based on vehicle specifications, brand, and other features.  
The backend model is trained using **XGBoost** and **Random Forest** with preprocessing steps including scaling and label encoding.

---

## 🧾 Overview

This project demonstrates a complete workflow:
1. **Data Preprocessing**  
2. **Feature Engineering**  
3. **Model Training & Evaluation**  
4. **Model Saving & Deployment**  
5. **Web Interface via Streamlit**

The goal is to predict the **estimated market price** of used cars in Indonesia using regression models.

---

## 🧰 Tech Stack

| Component | Tools / Libraries |
|------------|------------------|
| Language | Python 3 |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Modeling | XGBoost, RandomForest (Scikit-learn) |
| Model Tracking | Joblib |
| Deployment | Streamlit |

---

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/used-car-price-predictor.git
cd used-car-price-predictor
pip install -r requirements.txt
```

If you don't have a requirements.txt yet, create it:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit joblib
pip freeze > requirements.txt
```
## 🌐 Running the Streamlit App

To launch the web interface locally:
```bash
streamlit run app.py
```
Then open the provided local URL (usually http://localhost:8501).

