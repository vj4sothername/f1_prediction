# f1_logistic_prediction.py

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

data_path = os.path.dirname(os.path.abspath(__file__))

# Resolve data directory (env var > ./data > script folder)
data_root_candidates = [
    os.environ.get("F1_DATA_DIR"),
    os.path.join(data_path, "data"),
    data_path,
]
data_root = next((d for d in data_root_candidates if d and os.path.isdir(d)), data_path)

# Load all CSVs from the chosen data directory (case-insensitive keys)
csv_files = [f for f in os.listdir(data_root) if f.lower().endswith('.csv')]
if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in '{data_root}'. Place your Ergast CSVs (e.g., results.csv, races.csv) there "
        f"or set F1_DATA_DIR to the directory containing them."
    )
dfs = {os.path.splitext(f)[0].lower(): pd.read_csv(os.path.join(data_root, f)) for f in csv_files}
print("Loaded CSVs:", sorted(dfs.keys()))

if 'results' not in dfs:
    raise KeyError(
        "Could not find 'results.csv'. Ensure the following files exist in the data folder:\n"
        "- results.csv\n- qualifying.csv\n- races.csv\n- drivers.csv\n- constructors.csv\n- circuits.csv\n"
        f"Looked in: {data_root}\nFound: {', '.join(csv_files)}"
    )

# Start with results as the base
merged = dfs['results']
merged = merged.rename(columns={"positionOrder": "position_order", "grid": "grid_position"})

# Merge in all other DataFrames on likely keys
merge_keys = {
    'qualifying': ['raceId', 'driverId'],
    'races': ['raceId'],
    'drivers': ['driverId'],
    'constructors': ['constructorId'],
    'circuits': ['circuitId']
}

for name, df in dfs.items():
    if name == 'results':
        continue
    keys = merge_keys.get(name)
    if keys:
        missing_in_merged = [k for k in keys if k not in merged.columns]
        missing_in_df = [k for k in keys if k not in df.columns]
        if missing_in_merged or missing_in_df:
            print(f"Cannot merge '{name}':")
            if missing_in_merged:
                print(f"  Missing in merged DataFrame: {missing_in_merged}")
            if missing_in_df:
                print(f"  Missing in '{name}' DataFrame: {missing_in_df}")
            continue
        merged = merged.merge(df, on=keys, how='left', suffixes=('', f'_{name}'))

# Drop rows with missing target or key features
merged.dropna(subset=['position_order', 'grid_position'], inplace=True)

# --- FIND AND REPORT '\\N' VALUES (should be none after replacement, but this is a check) ---
# Replace '\\N' with np.nan in all DataFrames before merging (apply here for merged safeguard too)
merged = merged.replace('\\N', np.nan)
mask = merged.apply(lambda col: col.astype(str).str.contains(r'\\N', na=False)).any()
cols_with_N = merged.columns[mask]
if len(cols_with_N) > 0:
    rows_with_N = merged[cols_with_N][merged[cols_with_N].astype(str).apply(lambda x: x.str.contains(r'\\N', na=False)).any(axis=1)]
    print("Rows containing '\\N':")
    print(rows_with_N)
else:
    print("No '\\N' values found in merged DataFrame.")

# --- CONVERT TIME/DURATION STRINGS TO SECONDS ---
def time_to_seconds(val):
    """Convert a time string like '1:42:11.687' or '1:23.456' to seconds."""
    if pd.isnull(val):
        return np.nan
    if isinstance(val, (int, float)):
        return val
    val = str(val)
    if ':' in val:
        parts = val.split(':')
        try:
            parts = [float(p) for p in parts]
            if len(parts) == 3:
                return parts[0]*3600 + parts[1]*60 + parts[2]
            elif len(parts) == 2:
                return parts[0]*60 + parts[1]
        except Exception:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan

# Detect columns that look like time/duration and convert them
for col in merged.columns:
    if merged[col].dtype == 'object':
        sample = merged[col].dropna().astype(str).head(10)
        if sample.str.contains(r'^\d+:\d+').any():
            merged[col] = merged[col].apply(time_to_seconds)

# Feature engineering
merged['experience'] = merged.groupby('driverId')['raceId'].transform('count')
merged['positions_gained'] = merged['grid_position'] - merged['position_order']
merged['top_10'] = merged['position_order'].apply(lambda x: 1 if x <= 10 else 0)

# Example: Calculate driver age at race if dob and date are present
if 'dob' in merged.columns and 'date' in merged.columns:
    merged['race_year'] = pd.to_datetime(merged['date'], errors='coerce').dt.year
    merged['driver_age'] = merged['race_year'] - pd.to_datetime(merged['dob'], errors='coerce').dt.year

# One-hot encode all categorical columns with few unique values
categorical_cols = [col for col in merged.columns if merged[col].dtype == 'object' and merged[col].nunique() < 20]
merged = pd.get_dummies(merged, columns=categorical_cols, drop_first=True)

# Remove leaky and ID columns
leaky_cols = [
    'position_order', 'top_10', 'raceId', 'driverId', 'constructorId', 'circuitId',
    'date', 'dob', 'url', 'positions_gained', 'points', 'laps', 'fastestLapTime',
    'resultId', 'qualifyId', 'statusId'
]

feature_cols = [
    col for col in merged.columns
    if col not in leaky_cols
    and 'sprint' not in col.lower()
    and not col.lower().endswith('id')
]

X = merged[feature_cols]
y = merged['top_10']

# Drop any remaining non-numeric columns (just in case)
non_numeric_cols = X.select_dtypes(include=['object']).columns
if len(non_numeric_cols) > 0:
    print("Dropping non-numeric columns from features:", list(non_numeric_cols))
    X = X.drop(columns=non_numeric_cols)

print(f"Final feature shape: {X.shape}")

if X.shape[0] == 0:
    raise ValueError("No samples left after cleaning. Please check your data and feature selection.")

# Skipping features without any observed values: ['fp1_time' 'fp2_time' 'fp3_time' 'quali_time' 'sprint_time'].
X = X.loc[:, (X != 0).any(axis=0)]

# Drop columns that are all NaN
X = X.dropna(axis=1, how='all')
print(f"Feature shape after dropping all-NaN columns: {X.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define both pipelines
pipelines = {
    "Logistic Regression": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ]),
    "Random Forest": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),  # Not strictly needed for RF, but keeps pipeline consistent
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
}

for name, pipe in pipelines.items():
    print(f"\n=== {name} ===")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("🏁 Classification Report:\n")
    print(classification_report(y_test, y_pred))
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.3f}")
    if name == "Logistic Regression":
        coefs = pipe.named_steps['clf'].coef_[0]
        importances = pd.Series(coefs, index=X.columns).sort_values(ascending=False)
        print("Top features (coefficients):\n", importances.head(20))
    else:
        importances = pd.Series(pipe.named_steps['clf'].feature_importances_, index=X.columns).sort_values(ascending=False)
        print("Top features (importances):\n", importances.head(20))

# Confusion Matrix
for name, pipe in pipelines.items():
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test)
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

# ROC Curve
for name, pipe in pipelines.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test)
    plt.title(f"{name} - ROC Curve")
    plt.show()

# Precision-Recall Curve
for name, pipe in pipelines.items():
    PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test)
    plt.title(f"{name} - Precision-Recall Curve")
    plt.show()

# Feature Importance (Random Forest)
rf = pipelines["Random Forest"].named_steps['clf']
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.head(10).plot(kind='barh')
plt.title("Random Forest - Top 10 Feature Importances")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.show()

# Feature Importance (Logistic Regression)
lr = pipelines["Logistic Regression"].named_steps['clf']
coefs = pd.Series(lr.coef_[0], index=X.columns).abs().sort_values(ascending=False)
coefs.head(10).plot(kind='barh')
plt.title("Logistic Regression - Top 10 Absolute Coefficients")
plt.xlabel("Absolute Coefficient")
plt.gca().invert_yaxis()
plt.show()

# Cross-validation boxplot
cv_results = []
for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
    cv_results.append(pd.Series(scores, name=name))
pd.concat(cv_results, axis=1).boxplot()
plt.title("Cross-Validation Accuracy Scores")
plt.ylabel("Accuracy")
plt.show()
