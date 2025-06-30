import joblib
import os

input_model_path = "classifier_bal_1.pkl"
output_model_path = "compressed_classifier_bal_1.pkl"

model = joblib.load(input_model_path)
# os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
joblib.dump(model, output_model_path, compress=9)

print(f"Original size: {os.path.getsize(input_model_path) / (1024 * 1024):.2f} MB")
print(f"Compressed size: {os.path.getsize(output_model_path) / (1024 * 1024):.2f} MB")