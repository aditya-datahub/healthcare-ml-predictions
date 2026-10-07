# ❤️ Heart Disease Prediction

A beginner-level machine learning project that predicts whether a person has a heart problem from basic medical measurements, using **Logistic Regression**.

This was one of my first ML projects, built while learning the end-to-end workflow: loading data, exploring it, splitting it, training a model, and making a prediction on new input.

---

## 🎯 Problem Statement

Given a patient's health details (age, cholesterol, blood pressure, chest pain type, etc.), predict whether the patient has a **defective heart (1)** or a **healthy heart (0)**.

This is a **binary classification** problem.

## 📊 Dataset

| Item | Detail |
|---|---|
| File | `heart_disease_data.csv` |
| Rows | 303 patients |
| Columns | 14 (13 features + 1 target) |
| Missing values | None |
| Target | `target` → `1` = Defective Heart, `0` = Healthy Heart |
| Class balance | 165 defective (54%) vs 138 healthy (46%), fairly balanced |

**Features**

| Column | Meaning |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = yes, 0 = no) |
| `restecg` | Resting ECG result |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels (0–3) |
| `thal` | Thalassemia result |

## 🛠️ Approach

1. **Load data** with Pandas and inspect it (`head`, `tail`, `shape`, `info`, `describe`)
2. **Check data quality**: no missing values found
3. **Check target distribution**: classes are reasonably balanced
4. **Split features and label**: `x` = 13 features, `y` = `target`
5. **Train/test split**: 80% train, 20% test, with `stratify=y` so both sets keep the same class ratio (`random_state=42`)
6. **Train model**: Logistic Regression (scikit-learn)
7. **Evaluate**: accuracy on training and test data
8. **Predictive system**: enter one patient's values and get a Healthy / Defective prediction

## 📈 Results

| Dataset | Accuracy |
|---|---|
| Training data | **83.9%** |
| Test data | **80.3%** |

The small gap between training and test accuracy shows the model is **not heavily overfitting**.

### Sample prediction

```python
input_data = (56, 1, 1, 120, 236, 0, 1, 178, 0, 0.8, 2, 0, 2)
# Output: Defective Heart
```

## 🧰 Tech Stack

Python · Pandas · NumPy · scikit-learn · Jupyter Notebook

## 📁 Project Structure

```
heart-disease-prediction/
├── README.md
├── main.ipynb                 # full notebook (EDA → model → prediction)
└── heart_disease_data.csv     # dataset
```

## ▶️ How to Run

```bash
pip install numpy pandas scikit-learn jupyter
jupyter notebook main.ipynb
```

Keep `heart_disease_data.csv` in the same folder as the notebook.

## ⚠️ Limitations and Next Steps

This project is intentionally simple. Things I would improve now:

- Only **accuracy** is reported. For a medical problem, **recall** matters more (missing a sick patient is costly), so I would add a confusion matrix, precision, recall, F1 and ROC-AUC.
- **Scale the features** (`StandardScaler`) and fix the Logistic Regression convergence warning.
- Compare with other models (Random Forest, SVM, KNN, XGBoost) and use cross-validation.
- Add EDA charts (correlation heatmap, feature distributions) and feature importance.

## 📝 Note

This is a learning project. It is **not** a medical tool and should not be used for real diagnosis.

---

👤 **Aditya Sharma** · [GitHub](https://github.com/aditya-datahub) · [LinkedIn](https://www.linkedin.com/in/aditya-sharma-data-analyst)
