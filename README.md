# HEMA Backend Integration (Flutter → Node.js → Python ML)

This repository documents the architecture, setup, known issues, and debugging steps for the **HEMA (Her Early Medical Assessment)** risk prediction backend. It is intended to help contributors and reviewers understand how the system works and how to troubleshoot common errors such as **504 Gateway Timeout** and **502 Bad Gateway**.

---

## 📌 Project Overview

HEMA is a mobile-based cervical cancer and HPV risk assessment system. The application collects questionnaire data from users via a Flutter mobile app, processes it through a Node.js backend, and performs risk prediction using a Python machine learning model.

The ML model uses **derived (engineered) features**, meaning the frontend does **not** send all model features directly. Instead, the backend computes them before inference.

---

## 🏗️ System Architecture

```
Flutter Mobile App
        ↓ (HTTP JSON)
Node.js (Express API)
        ↓ (stdin/stdout via child_process)
Python ML Script
        ↓
Prediction Result (JSON)
```

### Components

* **Frontend:** Flutter (Dart)
* **Backend API:** Node.js + Express
* **ML Layer:** Python (scikit-learn)
* **Model Artifacts:**

  * `ensemble_model.pkl`
  * `scaler.pkl`
  * `feature_cols.pkl`
* **Local Tunneling:** ngrok
* **Target Hosting:** Render

---

## 📊 Machine Learning Model Notes

* The model was trained using **both raw and derived features**.
* Only **raw features** are collected from the user.
* Derived features (e.g. risk scores, binary flags, interaction terms) are computed **inside the Python script**.

### Example Features in `feature_cols.pkl`

```
Age
Age_Group
Sexual Partners
High_Sexual_Partners
First_Sexual_Activity_Age
Early_Sexual_Activity
Smoker
STDs_History
Pap_Result_Positive
Risk_Score
HighRisk_Combo
HighRisk_Sexual_Age
Smoking_STDs
HPV_Pap
```

⚠️ **Important:** The frontend is **not expected** to send all of these. Only base inputs are required.

---

## 📝 Data Sent from Flutter

Flutter collects questionnaire responses dynamically and sends them as a Dart `Map<String, dynamic>`.

Example payload:

```json
{
  "age": 20,
  "sexual_partners": 4,
  "first_sex_age": 16,
  "hpv_test": "Yes",
  "pap_smear": "No",
  "smoking": "Yes",
  "std_history": "No",
  "screening_type": "HPV DNA Test"
}
```

* JSON serialization is handled using Dart’s `jsonEncode`, ensuring valid JSON formatting.
* No manual string manipulation is performed in the frontend.

---

## 🔌 Backend Communication (Node.js → Python)

Node.js invokes the Python ML script using `child_process.spawn` and communicates via stdin/stdout.

### Node.js Pattern

```js
const py = spawn('python', ['predict_risk.py']);
py.stdin.write(JSON.stringify(data));
py.stdin.end();
```

### Python Pattern

```python
input_json = sys.stdin.read()
data = json.loads(input_json)

# feature engineering

print(json.dumps({
    "riskLevel": risk_level,
    "score": score
}))
```

---

## ❗ Known Issues

### 1. 504 Gateway Timeout

Occurs when:

* Python blocks on `sys.stdin.read()`
* Python fails to return output
* Node waits indefinitely for stdout
* ngrok / Render times out the HTTP request

### 2. 502 Bad Gateway

Occurs when:

* Python process exits without valid stdout
* Node attempts to respond after timeout
* Multiple response attempts are triggered

---

## 🧠 Root Cause Analysis

The primary issue is **inter-process communication reliability** between Node.js and Python:

* `sys.stdin.read()` is blocking and waits for EOF
* If stdin is not closed correctly, Python hangs
* If Python errors before printing JSON, Node receives nothing
* Node’s timeout handler may fire before Python responds

This results in gateway errors even when frontend JSON is valid.

---

## 🛠️ Debugging Steps

### Frontend

* Log outgoing JSON using:

  ```dart
  print(jsonEncode(answers));
  ```

### Node.js

* Log received payload before spawning Python
* Log stdout and stderr from Python
* Ensure only **one** HTTP response is sent

### Python

* Wrap logic in `try/except`
* Always print a JSON response (even on error)
* Log to `stderr` for debugging

---

## 📦 Environment

* Node.js ≥ 18
* Python 3.10
* scikit-learn
* pandas
* joblib
* Flutter (stable channel)
* Windows (local dev)

---

## 🤝 Contributing

If you are helping debug or improve this system, please focus on:

* Non-blocking stdin handling
* Safer subprocess communication
* ML inference latency reduction
* Robust error handling between layers

---

## 📬 Support

If you need additional context, logs, or sample files, open an issue and request:

* Full Node.js route
* Full Python script
* Sample payload
* Error logs

---

Thank you for contributing to HEMA.
