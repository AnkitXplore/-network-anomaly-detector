
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from sklearn.utils import resample

# Load the dataset
data_path = "data/processed/cleaned_KDDTrain.csv"

if not os.path.exists(data_path):
    print(f"❌ File not found: {data_path}")
    exit()

df = pd.read_csv(data_path)

# Split features and label
X = df.drop("label", axis=1)
y = df["label"]

# After loading and preprocessing the data, add:
print('Class distribution before balancing:')
print(df['label'].value_counts())

# Optional: Upsample minority class (anomalies)
majority = df[df['label'] == 0]
minority = df[df['label'] == 1]
if len(minority) > 0 and len(majority) > len(minority):
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    df_balanced = pd.concat([majority, minority_upsampled])
    print('Class distribution after upsampling:')
    print(df_balanced['label'].value_counts())
    # Use df_balanced for training
    X = df_balanced.drop("label", axis=1)
    y = df_balanced["label"]
else:
    print('No upsampling performed.')
    # ... continue with model training using df ...

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/Network_Anomility.joblib")

print("✅ Model trained and saved as 'models/Network_Anomility.joblib'")
print("Training columns:", X.columns.tolist())
