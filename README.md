
# Expresso Churn Prediction App

This project implements a **Churn Prediction** model using machine learning techniques on Expresso Telecom data. The model predicts whether a customer is likely to churn based on various behavioral attributes. A **Streamlit** application has been developed to allow users to input customer features and predict churn probability in real-time.

## Project Overview

Expresso is a telecom service provider in Africa. This project focuses on analyzing data from Expresso's customers and predicting churn. Churn prediction is critical for businesses to identify clients at risk of leaving, enabling proactive retention strategies.

The app provides:
- **User inputs** for customer attributes (e.g., `montant`, `revenue`, `tenure`, etc.)
- **Model selection** (Random Forest in this case)
- **Prediction and probability of churn**

### Dataset

The dataset includes customer information from Expresso's markets in Mauritania and Senegal. The features include customer behavior, usage patterns, and demographic data. Key features include:
REGION- the location of each client

TENURE- duration in the network

MONTANT- top-up amount

FREQUENCE_RECH- number of times the customer refilled

REVENUE- monthly income of each client

ARPU_SEGMENT- income over 90 days / 3

FREQUENCE- number of times the client has made an income

DATA_VOLUME- number of connections

ON_NET- inter expresso call

ORANGE- call to orange

TIGO- call to Tigo

ZONE1- call to zones1

ZONE2- call to zones2

MRG- a client who is going

REGULARITY- number of times the client is active for 90 days

TOP_PACK- the most active packs

FREQ_TOP_PACK- number of times the client has activated the top pack packages

CHURN- variable to predict - Target

## Files in the Repository

- `churn.py`: Main script that runs the Streamlit application.
- `random_forest_model.pkl`: Pre-trained Random Forest model for churn prediction.
- `scaler_balanced.pkl`: Scaler used for data normalization.
- `README.md`: Project documentation.

### Large Files
Due to size constraints, the following files are stored on Google Drive:

- [Expresso_churn_dataset.csv](https://drive.google.com/file/d/1QJsp9NUekDzuldRnRBqs9w5Rfv3NXAbt/view?usp=sharing)

- [random_forest_model.pkl](https://drive.google.com/file/d/1C8G67G_0248UefxbeByijFc0Bh2iaKkc/view?usp=sharing)

- [scaler_balanced.pkl](https://drive.google.com/file/d/1PWMcnKM11NdlRG_K8thGQwdcnBHtHFr5/view?usp=sharing)

Download these files and place them in the appropriate directories before running the project.

## Libraries used


- Python 3.x
- Required libraries: `streamlit`, `scikit-learn`, `joblib`, `pandas`



## Model and Evaluation

The machine learning model used in this project is a **Random Forest Classifier**, chosen for its robustness and performance. The model was trained using the Expresso customer dataset with the goal of minimizing false negatives (i.e., predicting churners as non-churners).

### Model Performance

| Metric       | Precision | Recall | F1-Score | Support  |
|--------------|-----------|--------|----------|----------|
| Churn (1)    | 0.56      | 0.09   | 0.15     | 120,685  |
| No Churn (0) | 0.82      | 0.98   | 0.90     | 525,530  |
| **Accuracy** |           |        | **0.82** | 646,215  |

## Future Enhancements

- **Hyperparameter tuning** to further improve model performance.
- **Balancing the dataset** for better recall on churners.
- **Support for additional models** (e.g., Logistic Regression, Adaboost) with user selection.

## Conclusion

This project showcases a robust approach to **churn prediction** using real-world data and a practical interface through **Streamlit**. With this tool, businesses can make data-driven decisions to reduce churn and improve customer retention.

