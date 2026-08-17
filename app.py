import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained SVM model
MODEL_PATH = "svm_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# HTML and CSS layout styled internally
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVM Prediction Service</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --input-bg: #0f172a;
            --border: #334155;
            --success-bg: rgba(16, 185, 129, 0.15);
            --success-border: #10b981;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 24px;
        }

        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 32px;
            width: 100%;
            max-width: 460px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4);
        }

        h1 {
            font-size: 1.6rem;
            font-weight: 700;
            margin-bottom: 6px;
            text-align: center;
        }

        p.subtitle {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.875rem;
            margin-bottom: 28px;
        }

        .form-group {
            margin-bottom: 16px;
        }

        label {
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 6px;
            color: var(--text-main);
        }

        input {
            width: 100%;
            padding: 12px 14px;
            background-color: var(--input-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-main);
            font-size: 0.95rem;
            transition: all 0.2s ease;
        }

        input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
        }

        button {
            width: 100%;
            padding: 13px;
            background-color: var(--accent);
            color: #ffffff;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            margin-top: 12px;
            transition: background-color 0.2s ease;
        }

        button:hover {
            background-color: var(--accent-hover);
        }

        .result-box {
            margin-top: 24px;
            padding: 16px;
            background: var(--success-bg);
            border: 1px solid var(--success-border);
            border-radius: 8px;
            text-align: center;
        }

        .result-label {
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }

        .result-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #ffffff;
            text-transform: uppercase;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>SVM Predictor</h1>
        <p class="subtitle">Enter numerical feature values for prediction</p>
        
        <form method="POST" action="/">
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" step="any" id="age" name="Age" placeholder="e.g., 35" required value="{{ request.form.get('Age', '') }}">
            </div>
            
            <div class="form-group">
                <label for="gender">Gender (Encoded Numeric Value)</label>
                <input type="number" step="any" id="gender" name="Gender" placeholder="e.g., 0 or 1" required value="{{ request.form.get('Gender', '') }}">
            </div>

            <div class="form-group">
                <label for="region">Region (Encoded Numeric Value)</label>
                <input type="number" step="any" id="region" name="Region" placeholder="e.g., 1" required value="{{ request.form.get('Region', '') }}">
            </div>

            <div class="form-group">
                <label for="occupation">Occupation (Encoded Numeric Value)</label>
                <input type="number" step="any" id="occupation" name="Occupation" placeholder="e.g., 2" required value="{{ request.form.get('Occupation', '') }}">
            </div>

            <div class="form-group">
                <label for="income">Income</label>
                <input type="number" step="any" id="income" name="Income" placeholder="e.g., 55000" required value="{{ request.form.get('Income', '') }}">
            </div>

            <button type="submit">Run Prediction</button>
        </form>

        {% if prediction is not none %}
        <div class="result-box">
            <div class="result-label">Prediction Result</div>
            <div class="result-value">{{ prediction }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            features = [
                float(request.form.get("Age")),
                float(request.form.get("Gender")),
                float(request.form.get("Region")),
                float(request.form.get("Occupation")),
                float(request.form.get("Income"))
            ]
            pred = model.predict([features])[0]
            prediction = str(pred)
        except Exception as e:
            prediction = f"Error: {str(e)}"
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
