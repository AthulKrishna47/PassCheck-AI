import pandas as pd
import joblib
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

start = time.time()


df = pd.read_csv(
    "Password.csv",          # Contains 620k examples of password and strength, 
    encoding="latin-1",      
    on_bad_lines="skip"
)


df = df.dropna(subset=["password", "strength"])
df["password"] = df["password"].astype(str)         # Converting password into string because '1234' could also be a password

df = df.sample(
    n=min(50000, len(df)),              # Taking into account of 50000 records to save/reduce training time
    random_state=42                 
)


X = df["password"]
y = df["strength"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


model = LogisticRegression(
    solver="saga",
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)


pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print(f"\nAccuracy: {accuracy:.4f}")


joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully!")

end = time.time()

print(f"Training Time: {round(end - start, 2)} seconds")