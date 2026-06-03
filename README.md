# 🔐 PassCheck AI

A modern, responsive full-stack web application implementing an embedded Machine Learning classification pipeline to evaluate password safety profiles dynamically.

## 🧠 Core Architecture
* **Vectorization Layer:** Implements a Character-Level n-gram TF-IDF Vectorizer extracting structural micro-patterns (ngram_range 1 to 2).
* **Classification Engine:** Built on a Logistic Regression variant configured with a fast `saga` solver optimized for larger sample feature tracking.
* **UI Interface:** An animated, glassmorphic UI layout using hardware-accelerated ambient glows, micro-interactions, and real-time state manipulation matching the inferred ML structural profile.

## 🛠️ Step-by-Step Setup
1. Clone the project locally:

   ```
   git clone <your-repo-link>
   ```
   ```
   cd PassCheck-AI
   ```
2. Install framework dependencies:

    ```
    pip install -r requirements.txt
    ```
3. Run the model optimization script (Optional):

    ```
    python train_model.py
    ```
4. Start the server:

    ```
    python app.py
    ```