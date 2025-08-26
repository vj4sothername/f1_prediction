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

# Replace '\\N' with np.nan (safeguard)
merged = merged.replace('\\N', np.nan)

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
        if sample.str_contains(r'^\d+:\d+', regex=True).any() if hasattr(sample, "str_contains") else sample.str.contains(r'^\d+:\d+').any():
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
    'top_10', 'raceId', 'driverId', 'constructorId', 'circuitId','position_order',
    'date', 'dob', 'url', 'points', 'laps', 'fastestLapTime',
    'resultId', 'qualifyId', 'statusId','fp1_time', 'fp2_time', 'fp3_time', 'q1_time', 'q2_time', 'q3_time',
    'q1_position','q2_position','q3_position','q1','q2','q3','positions_gained','time','round','race_year','year'
    ,'constructorId_qualifying','time_races','quali_time'
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

# Remove all-zero columns
X = X.loc[:, (X != 0).any(axis=0)]

# Drop columns that are all NaN
X = X.dropna(axis=1, how='all')

# NEW: Drop very sparse columns (e.g., <5% non-missing) to avoid imputer warnings and speed up
min_non_null = max(1, int(np.ceil(0.05 * len(X))))
before = X.shape[1]
X = X.dropna(axis=1, thresh=min_non_null)
# Drop constant columns
const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
if const_cols:
    X = X.drop(columns=const_cols)
print(f"Dropped {before - X.shape[1]} sparse/constant columns")

print(f"Final feature shape: {X.shape}")
if X.shape[0] == 0:
    raise ValueError("No samples left after cleaning. Please check your data and feature selection.")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define both pipelines (faster RF, parallel where safe)
pipelines = {
    "Logistic Regression": Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, n_jobs=None))
    ]),
    "Random Forest": Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),  # harmless for RF
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42))
    ])
}

cv_scores_by_model = {}

for name, pipe in pipelines.items():
    print(f"\n=== {name} ===")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("🏁 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Lighter and parallel CV
    scores = cross_val_score(pipe, X, y, cv=3, scoring='accuracy', n_jobs=-1)
    cv_scores_by_model[name] = scores
    print(f"Cross-validation accuracy scores: {scores}")
    print(f"Mean CV accuracy: {scores.mean():.3f}")

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

# Cross-validation boxplot (reuse computed scores to avoid extra work)
cv_results = [pd.Series(scores, name=name) for name, scores in cv_scores_by_model.items()]
pd.concat(cv_results, axis=1).boxplot()
plt.title("Cross-Validation Accuracy Scores")
plt.ylabel("Accuracy")
plt.show()
