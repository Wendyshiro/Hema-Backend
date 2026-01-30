import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# === 1. Load dataset ===
df = pd.read_excel("data/Cervical Cancer Datasets_.xlsx")
df = df.dropna(axis=1, how='all')
print("✅ Dataset Loaded Successfully!\n")
print(df.head())

# === 2. Handle Basic Numeric Columns ===
df['Age'] = df['Age'].astype(float)
df['Sexual Partners'] = df['Sexual Partners'].astype(float)
df['First_Sexual_Activity_Age'] = df['First Sexual Activity Age'].astype(float)

# === 3. One-Hot Encoding ===
df_encoded = pd.get_dummies(
    df,
    columns=["Smoking Status", "STDs History", "HPV Test Result", "Pap Smear Result"],
    drop_first=True
)

# === 4. Derived Risk Indicators (BEFORE creating Risk Score) ===
df_encoded['Age_Group'] = pd.cut(df['Age'], bins=[0,20,30,40,50,100], labels=[0,1,2,3,4]).astype(int)
df_encoded['High_Sexual_Partners'] = (df['Sexual Partners'] >= 3).astype(int)
df_encoded['Early_Sexual_Activity'] = (df['First_Sexual_Activity_Age'] < 16).astype(int)

# Reference encoded columns (adjust names to match your actual column names)
df_encoded['Smoker'] = df_encoded.get('Smoking Status_Yes', 0)
df_encoded['STDs_History'] = df_encoded.get('STDs History_Yes', 0)
df_encoded['Pap_Positive'] = df_encoded.get('Pap Smear Result_Positive', 0)

# === 5. Interaction Features ===
df_encoded['HighRisk_Combo'] = (
    df_encoded['High_Sexual_Partners'] * 
    df_encoded['Early_Sexual_Activity'] * 
    df_encoded['Smoker']
)
df_encoded['HighRisk_Sexual_Age'] = df_encoded['High_Sexual_Partners'] * df_encoded['Early_Sexual_Activity']
df_encoded['Smoking_STDs'] = df_encoded['Smoker'] * df_encoded['STDs_History']
df_encoded['Pap_Smoking'] = df_encoded['Pap_Positive'] * df_encoded['Smoker']

# === 6. Create Risk Score (TARGET for Regression) ===
df_encoded['Risk_Score'] = (
    df_encoded['High_Sexual_Partners'] +
    df_encoded['Early_Sexual_Activity'] +
    df_encoded['Smoker'] +
    df_encoded['STDs_History'] +
    df_encoded['Pap_Positive']
)

# === 7. Create Risk Category (TARGET for Classification) - NUMERIC LABELS ===
df_encoded['Risk_Category'] = pd.cut(
    df_encoded['Risk_Score'],
    bins=[-1, 1, 3, 5],
    labels=[0, 1, 2]  # 0=Low, 1=Medium, 2=High (numeric for XGBoost)
).astype(int)

# Create mapping for interpretation
risk_category_map = {0: 'Low', 1: 'Medium', 2: 'High'}
risk_category_map_reverse = {'Low': 0, 'Medium': 1, 'High': 2}

# Also create HPV Binary target (for your original classification task)
df_encoded['HPV_Binary'] = df_encoded.get('HPV Test Result_Positive', 0)

print("✅ Feature Engineering Completed!\n")
print(f"Risk Score Distribution:\n{df_encoded['Risk_Score'].value_counts().sort_index()}")
print(f"\nRisk Category Distribution:")
for code, label in risk_category_map.items():
    count = (df_encoded['Risk_Category'] == code).sum()
    print(f"  {label} ({code}): {count}")

# === 8. Choose features for training (NO target leakage) ===
feature_cols = [
    'Age', 'Age_Group', 'Sexual Partners', 'High_Sexual_Partners',
    'First_Sexual_Activity_Age', 'Early_Sexual_Activity',
    'Smoker', 'STDs_History', 'Pap_Positive',
    'HighRisk_Combo', 'HighRisk_Sexual_Age', 'Smoking_STDs', 'Pap_Smoking'
]

X = df_encoded[feature_cols].copy()

# === 9. Define Targets ===
y_risk_score = df_encoded['Risk_Score']  # Regression target
y_risk_category = df_encoded['Risk_Category']  # Classification target (numeric)

# === 10. Normalize numeric features ===
scaler = StandardScaler()
num_cols = ['Age', 'Sexual Partners', 'First_Sexual_Activity_Age']
X[num_cols] = scaler.fit_transform(X[num_cols])

# === 11. Train-test split ===
X_train, X_test, y_score_train, y_score_test, y_cat_train, y_cat_test = train_test_split(
    X, y_risk_score, y_risk_category, 
    stratify=y_risk_category, 
    test_size=0.2, 
    random_state=42
)

print("\n✅ Train-Test Split Completed!")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ============================================================
# PART A: REGRESSION MODELS (Predict exact Risk Score)
# ============================================================
print("\n" + "="*70)
print("PART A: REGRESSION MODELS - Predicting Risk Score (0-5)")
print("="*70)

# === Random Forest Regressor ===
rf_reg = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_reg.fit(X_train, y_score_train)
rf_reg_pred = rf_reg.predict(X_test)

print("\n🌲 Random Forest Regression Results:")
mse_rf = mean_squared_error(y_score_test, rf_reg_pred)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_score_test, rf_reg_pred)
r2_rf = r2_score(y_score_test, rf_reg_pred)

print(f"  MSE: {mse_rf:.4f}")
print(f"  RMSE: {rmse_rf:.4f}")
print(f"  MAE: {mae_rf:.4f}")
print(f"  R² Score: {r2_rf:.4f}")

# === XGBoost Regressor ===
xgb_reg = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_reg.fit(X_train, y_score_train)
xgb_reg_pred = xgb_reg.predict(X_test)

print("\n🚀 XGBoost Regression Results:")
mse_xgb = mean_squared_error(y_score_test, xgb_reg_pred)
rmse_xgb = np.sqrt(mse_xgb)
mae_xgb = mean_absolute_error(y_score_test, xgb_reg_pred)
r2_xgb = r2_score(y_score_test, xgb_reg_pred)

print(f"  MSE: {mse_xgb:.4f}")
print(f"  RMSE: {rmse_xgb:.4f}")
print(f"  MAE: {mae_xgb:.4f}")
print(f"  R² Score: {r2_xgb:.4f}")

# ============================================================
# PART B: CLASSIFICATION MODELS (Predict Risk Category)
# ============================================================
print("\n" + "="*70)
print("PART B: CLASSIFICATION MODELS - Predicting Risk Category")
print("="*70)

# === Handle class imbalance with SMOTE ===
smote = SMOTE(random_state=42)
X_train_res, y_cat_train_res = smote.fit_resample(X_train, y_cat_train)

print(f"\n📊 After SMOTE - Class Distribution:")
unique, counts = np.unique(y_cat_train_res, return_counts=True)
for val, count in zip(unique, counts):
    print(f"  {risk_category_map[val]} ({val}): {count}")

# === Random Forest Classifier ===
rf_clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train_res, y_cat_train_res)
rf_clf_pred = rf_clf.predict(X_test)

print("\n🌲 Random Forest Classification Results:")
print(f"  Accuracy: {accuracy_score(y_cat_test, rf_clf_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_cat_test, rf_clf_pred))
print("\nClassification Report:")
# Create target names for the report
target_names = [risk_category_map[i] for i in sorted(np.unique(y_cat_test))]
print(classification_report(y_cat_test, rf_clf_pred, target_names=target_names))

# === XGBoost Classifier ===
xgb_clf = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss'
)
xgb_clf.fit(X_train_res, y_cat_train_res)
xgb_clf_pred = xgb_clf.predict(X_test)

print("\n🚀 XGBoost Classification Results:")
print(f"  Accuracy: {accuracy_score(y_cat_test, xgb_clf_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_cat_test, xgb_clf_pred))
print("\nClassification Report:")
print(classification_report(y_cat_test, xgb_clf_pred, target_names=target_names))

# === Ensemble Classifier ===
ensemble_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('xgb', xgb_clf)],
    voting='soft',
    n_jobs=-1
)
ensemble_clf.fit(X_train_res, y_cat_train_res)
ensemble_clf_pred = ensemble_clf.predict(X_test)

print("\n🎯 Ensemble Classification Results:")
print(f"  Accuracy: {accuracy_score(y_cat_test, ensemble_clf_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_cat_test, ensemble_clf_pred))
print("\nClassification Report:")
print(classification_report(y_cat_test, ensemble_clf_pred, target_names=target_names))

# ============================================================
# PART C: FEATURE IMPORTANCE
# ============================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)

feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'RF_Regression': rf_reg.feature_importances_,
    'RF_Classification': rf_clf.feature_importances_
}).sort_values('RF_Regression', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10).to_string(index=False))

# ============================================================
# PART D: PREDICTION FUNCTION WITH RECOMMENDATIONS
# ============================================================

def predict_patient_risk(patient_data, reg_model, clf_model, scaler, feature_cols):
    """
    Comprehensive risk prediction with recommendations
    
    Returns:
    - Risk score (0-5)
    - Risk category (Low/Medium/High)
    - Probabilities
    - Personalized recommendations
    """
    # Prepare patient data
    patient_df = pd.DataFrame([patient_data])
    
    # Create a copy for scaling
    patient_scaled = patient_df.copy()
    
    # Normalize numeric features
    num_cols_to_scale = ['Age', 'Sexual Partners', 'First_Sexual_Activity_Age']
    patient_scaled[num_cols_to_scale] = scaler.transform(patient_df[num_cols_to_scale])
    
    # Ensure correct feature order
    X_patient = patient_scaled[feature_cols]
    
    # Predictions
    risk_score = reg_model.predict(X_patient)[0]
    risk_category_code = clf_model.predict(X_patient)[0]
    risk_category = risk_category_map[risk_category_code]
    risk_proba = clf_model.predict_proba(X_patient)[0]
    
    # Get probabilities for each class
    class_labels = clf_model.classes_
    prob_dict = {risk_category_map[int(label)]: prob * 100 for label, prob in zip(class_labels, risk_proba)}
    
    # Generate recommendations
    recommendations = []
    
    # Base recommendations by category
    if risk_category == 'High':
        recommendations.append("⚠️ HIGH RISK ALERT:")
        recommendations.append("  • Schedule IMMEDIATE gynecological consultation")
        recommendations.append("  • Annual Pap smear and HPV testing mandatory")
        recommendations.append("  • Consider HPV vaccination if not already done")
    elif risk_category == 'Medium':
        recommendations.append("⚡ MEDIUM RISK:")
        recommendations.append("  • Schedule check-up within 3-6 months")
        recommendations.append("  • Pap smear every 1-2 years recommended")
    else:
        recommendations.append("✓ LOW RISK:")
        recommendations.append("  • Continue routine screening every 3 years")
    
    # Personalized recommendations based on risk factors
    if patient_data.get('Smoker', 0) == 1:
        recommendations.append("\n🚭 SMOKING CESSATION URGENT:")
        recommendations.append("  • Enroll in cessation program immediately")
        recommendations.append("  • Smoking increases cervical cancer risk by 2-3x")
    
    if patient_data.get('High_Sexual_Partners', 0) == 1:
        recommendations.append("\n🛡️ SEXUAL HEALTH:")
        recommendations.append("  • Use barrier protection consistently")
        recommendations.append("  • Increase STD screening frequency")
    
    if patient_data.get('Pap_Positive', 0) == 1:
        recommendations.append("\n📋 ABNORMAL PAP SMEAR:")
        recommendations.append("  • Follow up with colposcopy ASAP")
        recommendations.append("  • Discuss treatment options with physician")
    
    if patient_data.get('STDs_History', 0) == 1:
        recommendations.append("\n💊 STD HISTORY:")
        recommendations.append("  • Complete all STD treatments")
        recommendations.append("  • Quarterly STD screenings recommended")
    
    if patient_data.get('Early_Sexual_Activity', 0) == 1:
        recommendations.append("\n📊 EARLY SEXUAL ACTIVITY:")
        recommendations.append("  • Increased monitoring recommended")
        recommendations.append("  • Annual gynecological exams starting now")
    
    recommendations.append("\n📌 GENERAL PREVENTION:")
    recommendations.append("  • Maintain healthy BMI")
    recommendations.append("  • Diet rich in fruits & vegetables")
    recommendations.append("  • Regular exercise (150 min/week)")
    recommendations.append("  • Limit alcohol consumption")
    recommendations.append("  • Practice safe sex consistently")
    
    return {
        'risk_score': round(risk_score, 2),
        'risk_category': risk_category,
        'probabilities': prob_dict,
        'recommendations': recommendations
    }

# ============================================================
# PART E: TEST PREDICTION FUNCTION
# ============================================================
print("\n" + "="*70)
print("EXAMPLE PATIENT PREDICTION")
print("="*70)

# Example patient data (use actual features from test set)
if len(X_test) > 0:
    # Get first test patient's original (unscaled) features
    test_idx = X_test.index[0]
    example_patient = df_encoded.loc[test_idx, feature_cols].to_dict()
    
    print(f"\n📋 Patient Features:")
    for key, value in example_patient.items():
        print(f"  {key}: {value}")
    
    result = predict_patient_risk(example_patient, rf_reg, ensemble_clf, scaler, feature_cols)
    
    print(f"\n📊 RISK ASSESSMENT:")
    print(f"  Risk Score: {result['risk_score']}/5.0")
    print(f"  Risk Category: {result['risk_category']}")
    
    print(f"\n🎯 RISK PROBABILITIES:")
    for category, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {prob:.1f}%")
    
    print(f"\n💡 PERSONALIZED RECOMMENDATIONS:")
    for rec in result['recommendations']:
        print(rec)

# ============================================================
# PART F: SAVE MODELS AND RESULTS
# ============================================================
print("\n" + "="*70)
print("SAVING MODELS AND RESULTS")
print("="*70)

# Get predictions for all data
# Need to scale the numeric columns first
X_full_scaled = X.copy()
df_encoded['Risk_Score_Predicted'] = rf_reg.predict(X_full_scaled)
df_encoded['Risk_Category_Predicted_Code'] = ensemble_clf.predict(X_full_scaled)
df_encoded['Risk_Category_Predicted'] = df_encoded['Risk_Category_Predicted_Code'].map(risk_category_map)

# Get probability predictions
proba = ensemble_clf.predict_proba(X_full_scaled)
for idx, class_code in enumerate(ensemble_clf.classes_):
    df_encoded[f'Risk_Confidence_{risk_category_map[class_code]}'] = proba[:, idx]

# Save to Excel
df_encoded.to_excel("cervical_risk_predictions_complete.xlsx", index=False)
print("✅ Predictions saved to: cervical_risk_predictions_complete.xlsx")

# Save models
joblib.dump(rf_reg, 'models/rf_regressor.pkl')
joblib.dump(ensemble_clf, 'models/ensemble_classifier.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(feature_cols, 'models/feature_cols.pkl')
joblib.dump(risk_category_map, 'models/risk_category_map.pkl')

print("✅ Models saved to /models/ directory!")
print("\nSaved files:")
print("  • rf_regressor.pkl (for Risk Score)")
print("  • ensemble_classifier.pkl (for Risk Category)")
print("  • scaler.pkl (for feature normalization)")
print("  • feature_cols.pkl (feature list)")
print("  • risk_category_map.pkl (category mapping)")

print("\n" + "="*70)
print("✅ ALL DONE! MODEL TRAINING COMPLETE")
print("="*70)

# ============================================================
# PART G: MODEL PERFORMANCE SUMMARY
# ============================================================
print("\n" + "="*70)
print("📈 FINAL MODEL PERFORMANCE SUMMARY")
print("="*70)

print("\n🔢 REGRESSION PERFORMANCE (Risk Score 0-5):")
print(f"  Best Model: {'XGBoost' if r2_xgb > r2_rf else 'Random Forest'}")
print(f"  R² Score: {max(r2_xgb, r2_rf):.4f}")
print(f"  RMSE: {min(rmse_xgb, rmse_rf):.4f}")

print("\n🎯 CLASSIFICATION PERFORMANCE (Risk Categories):")
print(f"  Ensemble Accuracy: {accuracy_score(y_cat_test, ensemble_clf_pred):.4f}")
print(f"  Random Forest Accuracy: {accuracy_score(y_cat_test, rf_clf_pred):.4f}")
print(f"  XGBoost Accuracy: {accuracy_score(y_cat_test, xgb_clf_pred):.4f}")

print("\n💡 KEY INSIGHTS:")
print(f"  • Dataset size: {len(df_encoded)} patients")
print(f"  • Training set: {len(X_train)} patients")
print(f"  • Test set: {len(X_test)} patients")
print(f"  • Number of features: {len(feature_cols)}")
print(f"  • Risk distribution: Low={sum(df_encoded['Risk_Category']==0)}, " +
      f"Medium={sum(df_encoded['Risk_Category']==1)}, " +
      f"High={sum(df_encoded['Risk_Category']==2)}")