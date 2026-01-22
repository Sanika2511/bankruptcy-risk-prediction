# Bankruptcy Risk Prediction

This project focuses on predicting whether a company is likely to go bankrupt based on key risk factors.  
It combines a trained machine learning model with simple NLP logic to make the prediction process more flexible and easier to understand.

The project was built as a practical exercise to connect data, modeling, and user-driven inputs in a clean and transparent way.

---

## What this project does

- Predicts bankruptcy risk using a trained machine learning model  
- Uses structured risk factors such as:
  - Industrial risk  
  - Management risk  
  - Financial flexibility  
  - Credibility  
  - Competitiveness  
  - Operating risk
- Supports short text descriptions of a company, which are:
  - Spell-corrected
  - Interpreted using basic NLP
  - Mapped to structured risk values
- Produces clear and interpretable prediction results

---

## How it works

1. Risk factors are provided directly or inferred from text  
2. Text inputs are processed and converted into numerical features  
3. The trained model evaluates bankruptcy risk  
4. The final prediction is generated based on these inputs  

---

## Tools and technologies

- Python  
- Scikit-learn (Support Vector Classifier)  
- spaCy  
- pyspellchecker  
- Pandas, NumPy  

---

## Dataset

The model was trained using a bankruptcy dataset stored locally in Excel (`.xlsx`) format.  
The dataset is not included in this repository and is not required to use the trained model.

---

## Author

Built by **Sanika Sharma**

