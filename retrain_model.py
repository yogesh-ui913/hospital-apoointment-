import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings("ignore")

print("Loading data...")
df = pd.read_csv("hospital_appointment.csv")

print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())

# Rename columns
df.rename(columns={'No-show': 'No_show', 'Hipertension': 'Hypertension', 'Handcap': 'Handicap'}, inplace=True)

# Target variable
target = "No_show"
df[target] = df[target].map({'No': 0, 'Yes': 1})

print("Unique values in target:", df[target].unique())

# Drop unnecessary columns
df = df.drop(columns=["PatientId", "AppointmentID", "ScheduledDay", "AppointmentDay"], errors='ignore')

# Age handling
df['Age'] = df['Age'].replace(0, df['Age'].mean())
df['Age'] = df['Age'].abs()

# Remove age outliers
Q3 = df['Age'].quantile(0.75)
Q1 = df['Age'].quantile(0.25)
IQR = Q3 - Q1
lower_limit = Q1 - (1.5 * IQR)
upper_limit = Q3 + (1.5 * IQR)
df = df[(df['Age'] >= lower_limit) & (df['Age'] <= upper_limit)]

print("Data shape after cleanup:", df.shape)

# Select features
feature_cols = ['Gender', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received', 'Age']
X = df[feature_cols].copy()
y = df[target].copy()

# Encode categorical variables
le = LabelEncoder()
for col in ['Gender', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received']:
    if col in X.columns:
        X[col] = le.fit_transform(X[col].astype(str))

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Reset index
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Train model
print("Training model...")
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training Accuracy: {train_score:.4f}")
print(f"Testing Accuracy: {test_score:.4f}")

# Save model
print("Saving model...")
joblib.dump({
    "model": model,
    "columns": X.columns.tolist()
}, "hospital_model.pkl")

print("✅ Model saved successfully as hospital_model.pkl")
