import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import onnx
import skl2onnx
from skl2onnx.common.data_types import StringTensorType

# 1️ Load Data
with open("data/sentiment.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 2️ Encode Sentiment Labels
label_mapping = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0}
df["sentiment_label"] = df["sentiment_label"].map(label_mapping)

# 3️ Convert Messages to TF-IDF Features
vectorizer = TfidfVectorizer(max_features=500)  # Limit to 500 most frequent words
X = vectorizer.fit_transform(df["message"])
y = df["sentiment_label"]

# 4️ Split Data into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️ Train a Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6️ Evaluate Model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 7️ Save Model & Vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# 8️ Convert Model to ONNX
initial_type = [("input", StringTensorType([None]))]
onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_type)
with open("sentiment_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✅ Model saved as sentiment_model.onnx")
