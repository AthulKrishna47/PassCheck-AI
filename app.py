from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

strength_map = {
    0: "Weak 🔴",
    1: "Medium 🟡",
    2: "Strong 🟢"
}

@app.route("/", methods=["GET", "POST"])
def home():
    # 1. Initialize ALL variables right away so they always exist
    result = None
    confidence = None
    strength_slug = None  # <-- Fixed! Now it's safe for GET requests

    # 2. Only compute values if the form was actually submitted
    if request.method == "POST":
        password = request.form["password"]
        vector = vectorizer.transform([password])
        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]

        confidence = round(max(probabilities) * 100, 2)
        result = strength_map[prediction]
        
        # Split out the emoji and lowercase the text ("weak", "medium", or "strong")
        strength_slug = result.split()[0].lower()

    # 3. This will now return safely on both GET and POST requests
    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        strength_slug=strength_slug
    )

if __name__ == "__main__":
    app.run(debug=True)