import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# 🔹 Load the dataset
data = pd.read_csv(r"/Users/anjalikulkarni/Downloads/ML project/feb/fake_loan_data.csv")

# 🔹 Ensure correct data types
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].astype(str)

for col in data.select_dtypes(include=['int64', 'float64']).columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# 🔹 Handle missing values
num_imputer = SimpleImputer(strategy='median')
data[data.select_dtypes(include=['int64', 'float64']).columns] = num_imputer.fit_transform(
    data.select_dtypes(include=['int64', 'float64'])
)

# 🔹 Ensure 'loan_status' is binary (0 or 1)
data['loan_status'] = data['loan_status'].astype(int)

# 🔹 WoE & IV Calculation
def calculate_woe_iv(data, feature, target):
    if data[feature].dtype != 'object':
        data[feature] = data[feature].astype(str)

    grouped = data.groupby(feature)[target].agg(['count', 'sum']).reset_index()
    grouped.columns = [feature, 'Total', 'Defaults']

    grouped['Total'] = pd.to_numeric(grouped['Total'], errors='coerce')
    grouped['Defaults'] = pd.to_numeric(grouped['Defaults'], errors='coerce')
    grouped.fillna(0, inplace=True)

    grouped['NonDefaults'] = grouped['Total'] - grouped['Defaults']

    value_counts = data[target].value_counts()
    total_good = value_counts.iloc[0] if 0 in value_counts.index else 1
    total_bad = value_counts.iloc[1] if 1 in value_counts.index else 1

    grouped['DistGood'] = grouped['NonDefaults'] / total_good
    grouped['DistBad'] = grouped['Defaults'] / total_bad

    with np.errstate(divide='ignore', invalid='ignore'):
        grouped['WoE'] = np.log(grouped['DistGood'] / grouped['DistBad'].replace(0, np.nan))

    grouped['IV'] = (grouped['DistGood'] - grouped['DistBad']) * grouped['WoE'].fillna(0)
    return grouped['IV'].sum()

# 🔹 Improved binning for numerical features
def calculate_numerical_woe_iv(data, feature, target):
    try:
        unique_values = data[feature].nunique()
        num_bins = min(10, unique_values)
        data[feature] = pd.to_numeric(data[feature], errors='coerce')

        data[feature] = data[feature].fillna(data[feature].median())  # FIXED Warning

        bins = pd.cut(data[feature], bins=num_bins, include_lowest=True, duplicates='drop')
        bins_str = bins.astype(str)

        return calculate_woe_iv(data.assign(**{feature: bins_str}), feature, target)

    except ValueError as e:
        print(f"Error in binning {feature}: {e}")
        return 0

# 🔹 Calculate IV for all features
iv_values = {}
for feature in data.select_dtypes(include=['object']).columns:
    iv_values[feature] = calculate_woe_iv(data, feature, 'loan_status')

for feature in data.select_dtypes(include=['int64', 'float64']).columns:
    iv_values[feature] = calculate_numerical_woe_iv(data, feature, 'loan_status')

iv_df = pd.DataFrame(iv_values.items(), columns=['Feature', 'IV']).sort_values(by='IV', ascending=False)
print(iv_df)

# 🔹 Feature selection & encoding
X = data.drop(columns=['loan_status'])
y = data['loan_status']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# 🔹 Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Logistic Regression pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear'))
])

# 🔹 Train model
model.fit(X_train, y_train)

# 🔹 Predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

# 🔹 Model evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'ROC AUC Score: {roc_auc:.2f}')

# 🔹 Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 🔹 Loan eligibility based on IV
def assess_loan_eligibility(iv_dict):
    eligibility = {}
    for feature, iv in iv_dict.items():
        if iv < 0.02:
            eligibility[feature] = "Useless for prediction"
        elif 0.02 <= iv < 0.1:
            eligibility[feature] = "Weak predictive power"
        elif 0.1 <= iv < 0.3:
            eligibility[feature] = "Medium predictive power"
        elif 0.3 <= iv < 0.5:
            eligibility[feature] = "Strong predictive power"
        else:
            eligibility[feature] = "Very strong predictive power"
    return eligibility

eligibility_results = assess_loan_eligibility(iv_values)
print("\nLoan Eligibility based on IV values:")
for feature, status in eligibility_results.items():
    print(f"{feature}: {status}")
