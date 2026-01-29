import sys
import json
import pandas as pd
import joblib
import traceback

# === Load trained models ONCE ===
try:
    ensemble_model = joblib.load("ensemble_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
except Exception as e:
    print(json.dumps({
        "error": "Failed to load model files",
        "details": str(e)
    }))
    sys.exit(1)

try:
    # === Read JSON input ===
    input_json = sys.stdin.read()
    if not input_json:
        print(json.dumps({"error": "Empty input"}))
        sys.exit(1)

    data = json.loads(input_json)

    # 🔹 Log received data for debugging
    print(f"Received input: {data}", file=sys.stderr)


    # === Build DataFrame safely ===
    df = pd.DataFrame([data])

    # === Ensure all expected features exist (categorical or numeric) ===
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0  # default value

    df = df[feature_cols]

    # === Numeric columns ===
    num_cols = ['Age', 'Sexual Partners', 'First_Sexual_Activity_Age', 'Risk_Score']

    # 🔹 Ensure all numeric columns exist and are type-safe
    for col in num_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 🔹 Scale numeric columns
    df[num_cols] = scaler.transform(df[num_cols])

    # === Predict risk ===
    prob = ensemble_model.predict_proba(df)[:, 1][0]
    score = round(prob * 10)

    riskLevel = (
        "low" if score < 4
        else "medium" if score < 7
        else "high"
    )

    # === Output JSON ===
    print(json.dumps({
        "riskLevel": riskLevel,
        "score": score
    }))

except Exception as e:
    # ALWAYS return JSON on failure
    print(json.dumps({
        "error": "Prediction failed",
        "details": str(e),
        "trace": traceback.format_exc()
    }))
    sys.exit(1)
