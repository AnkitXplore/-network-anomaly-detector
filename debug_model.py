import joblib
import pandas as pd

# Load the model
model = joblib.load("models/Network_Anomility.joblib")

print("Model type:", type(model))
print("Model parameters:", model.get_params() if hasattr(model, 'get_params') else "No get_params method")

# Try to get feature names if the model has them
if hasattr(model, 'feature_names_in_'):
    print("Expected feature names:", model.feature_names_in_)
    print("Number of expected features:", len(model.feature_names_in_))
    print("All expected features:", list(model.feature_names_in_))

# Check if it's a pipeline
if hasattr(model, 'steps'):
    print("This is a pipeline with steps:", [step[0] for step in model.steps])
    # Get the actual estimator
    estimator = model.steps[-1][1]
    if hasattr(estimator, 'feature_names_in_'):
        print("Pipeline estimator feature names:", estimator.feature_names_in_)

print("\nModel loaded successfully!") 