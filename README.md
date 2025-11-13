# End-to-End Credit Risk Modeling

This project demonstrates a complete data science workflow for predicting credit default risk. The goal is to build a machine learning model that can accurately predict the probability of a borrower defaulting on a loan, enabling financial institutions to make data-driven lending decisions and minimize financial losses.

## 📊 Business Problem

Financial institutions face significant risk when lending money. A "default" occurs when a borrower fails to make a loan repayment for over 90 days. Accurately predicting these defaults before a loan is approved is a critical business function.

This model provides a risk score for each applicant, allowing the bank to:

*  Approve low-risk applicants.
*  Manually review medium-risk applicants.
*  Reject high-risk applicants, saving the bank from potential losses.

## ⚙️ Data Science Methodology

This project follows a structured methodology from data exploration to model evaluation.

### 1. Exploratory Data Analysis (EDA)

*See `basic_data_analysis.ipynb`*

The initial step was a deep dive into the dataset to understand its characteristics.

* **Missing Values:** Analyzed and handled missing values, particularly for `MonthlyIncome`.
* **Distributions:** Visualized the distributions of key features like `Age`, `DebtRatio`, and `NumberOfDependents`.
* **Correlations:** Used a correlation heatmap to identify relationships between variables.
* **Key Insight (Imbalance):** The dataset is highly imbalanced. The vast majority of applicants did not default. This discovery was critical, as it means **Accuracy** is a misleading metric. Therefore, **AUC-ROC** was chosen as the primary evaluation metric.

![Correlation Heatmap]()
<img width="552" height="455" alt="image" src="https://github.com/user-attachments/assets/e2a35bf7-c861-48e2-ad9e-f975f7d5f6a2" />


### 2. Feature Engineering

*See `Credit Risk Analysis.ipynb`*

This was the most critical step for improving model performance. Instead of using raw data, new features were created based on domain knowledge.

**Key Engineered Features:**

* **`DebtRatio`:** A more robust `DebtRatio` was calculated. A ratio of 0 indicates no debt, while a high ratio indicates a large debt burden relative to income, which is a strong predictor of default.
* **`MonthlyIncomePerDependent`:** This feature combines two variables to better represent an applicant's financial cushion.
* **`Binned Age`:** Grouped `Age` into categorical bins (e.g., 'Youth', 'Adult', 'Senior') to capture non-linear relationships.

### 3. Modeling and Evaluation

*See `Credit Risk Analysis.ipynb`*

To prove the value of feature engineering, I trained and compared two sets of models:

* **Baseline Model:** Trained on the raw, original data.
* **Improved Model:** Trained on the data with the new engineered features.

**Models Tested:** Logistic Regression, Random Forest, Voting Classifier (Ensemble).

**Evaluation Metric:** **AUC-ROC** (Area Under the Receiver Operating Characteristic Curve). This is the industry-standard metric for imbalanced classification problems, as it measures a model's ability to distinguish between positive and negative classes.

## 📈 Results and Key Findings

The results clearly demonstrate that feature engineering was more impactful than just model selection. The models trained on the enhanced features significantly outperformed the baseline models.

| Model Version | Evaluation Metric (AUC-ROC) | Key Takeaway |
| :--- | :--- | :--- |
| **Baseline Model** (Raw Data) | **0.792** | This is our starting performance with no feature engineering. |
| **Improved Model** (With Feature Engineering) | **0.875** | Achieved a **10.5% performance lift** directly due to creating better features. |

The final ensemble model (`predictions_voting_Feature_transformation.csv`) provides the most robust predictions by combining the strengths of multiple models.

![Final ROC Curve]
<matplotlib.figure.Figure at 0x7f624e4d5250><img width="509" height="514" alt="image" src="https://github.com/user-attachments/assets/f71168df-300d-4191-9c95-5dc8465246d4" />


## 🛠️ Technical Stack

* **Language:** Python
* **Core Libraries:** Pandas, NumPy, Scikit-learn
* **Data Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook

## 🚀 How to Run This Project

1.  Clone the repository:
    ```bash
    git clone [https://github.com/Abhi2009iv/Quantitaive-credit-risk-modelling.git](https://github.com/Abhi2009iv/Quantitaive-credit-risk-modelling.git)
    cd Quantitaive-credit-risk-modelling
    ```
2.  Create a virtual environment (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter
    ```
    (Note: You can create a `requirements.txt` file for a more professional setup)

4.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
5.  Open and run `basic_data_analysis.ipynb` first, followed by `Credit Risk Analysis.ipynb`.

## 🔮 Future Improvements

This project provides a strong foundation. The next steps to enhance it would be:

* **Advanced Models:** Experiment with gradient-boosting models (XGBoost, LightGBM) which often win on tabular data.
* **Model Interpretability:** Use SHAP or LIME to explain *why* the model flags a specific applicant as high-risk. This is crucial for business adoption and regulatory compliance.
* **Deployment:** Containerize the final model using Docker and deploy it as a REST API using Flask or FastAPI, making it accessible for a real-world application.
