# Heart Disease Prediction using Machine Learning

This project uses machine learning algorithms to predict whether a person has heart disease, based on a dataset obtained from Kaggle. The models are trained using **scikit-learn** and evaluated using standard metrics like accuracy, ROC-AUC, and feature importance.

---

## Dataset

- Source: [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets)  
- Features include medical information such as:
  - Age, Sex, Chest Pain Type (`cp`), Resting Blood Pressure (`trestbps`), Cholesterol (`chol`), Fasting Blood Sugar (`fbs`), ECG Results (`restecg`), Max Heart Rate (`thalach`), Exercise Induced Angina (`exang`), ST Depression (`oldpeak`), Slope of ST segment, Number of vessels (`ca`), Thalassemia (`thal`), and more.
- Target: `1` = Heart Disease, `0` = No Heart Disease

---

## Machine Learning Models Used

- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Random Forest Classifier**

All models were trained using **scikit-learn** and evaluated using performance metrics including:

- Accuracy
- Confusion Matrix
- ROC Curve & AUC Score
- Feature Importance (for Logistic Regression)

---

##  Evaluation

- Plotted ROC Curves to compare model performance.
- Visualized feature importance for logistic regression.
- Models show promising performance in distinguishing between healthy and heart-diseased individuals.

---

##  Model Saving (Optional)

Although the trained models were not saved in this version, you can easily save and reload them using:

Using pickle or joblib
```python
import pickle

# Save
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
