import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# === 1. Load dataset ===
df = pd.read_excel("data/Cervical Cancer Datasets_.xlsx")
df = df.dropna(axis=1, how='all')  # remove any completely empty columns
print("✅ Dataset Loaded Successfully!\n")

# === 2. Create Target Column (the thing we want to predict) ===
df['HPV_Binary'] = df['HPV Test Result'].apply(lambda x: 1 if str(x).strip().upper() == 'POSITIVE' else 0)
y = df['HPV_Binary']

# === 3. Handle Basic Numeric Columns ===
df['Age'] = df['Age'].astype(float)
df['Sexual Partners'] = df['Sexual Partners'].astype(float)
df['First_Sexual_Activity_Age'] = df['First Sexual Activity Age'].astype(float)

# === 4. Binary Encodings (convert Y/N into 1/0) ===
binary_map = {'Y': 1, 'N': 0}
df['Smoking_Status_Binary'] = df['Smoking Status'].map(binary_map)
df['STDs_History_Binary'] = df['STDs History'].map(binary_map)
df['HPV_Positive'] = (df['HPV Test Result'].str.upper() == 'POSITIVE').astype(int)
df['Pap_Positive'] = (df['Pap Smear Result'].str.upper() == 'Y').astype(int)

# === 5. Derived Risk Indicators ===
df['Age_Group'] = pd.cut(df['Age'], bins=[0,20,30,40,50,100], labels=[0,1,2,3,4]).astype(int)
df['High_Sexual_Partners'] = (df['Sexual Partners'] >= 3).astype(int)
df['Early_Sexual_Activity'] = (df['First_Sexual_Activity_Age'] < 16).astype(int)
df['Smoker'] = df['Smoking_Status_Binary']
df['STDs_History'] = df['STDs_History_Binary']
df['Pap_Result_Positive'] = df['Pap_Positive']

# === 6. Extra Interaction Features (combine risk factors) ===
df['HighRisk_Combo'] = df['High_Sexual_Partners'] * df['Early_Sexual_Activity'] * df['Smoker']
df['HighRisk_Sexual_Age'] = df['High_Sexual_Partners'] * df['Early_Sexual_Activity']
df['Smoking_STDs'] = df['Smoking_Status_Binary'] * df['STDs_History_Binary']
df['HPV_Pap'] = df['HPV_Positive'] * df['Pap_Positive']

# === 8. Create Risk Score ===
df['Risk_Score'] = (
    df['High_Sexual_Partners'] +
    df['Early_Sexual_Activity'] +
    df['Smoker'] +
    df['STDs_History'] +
    df['Pap_Result_Positive']
)

# === 9. Choose features for training ===
feature_cols = [
    'Age','Age_Group','Sexual Partners','High_Sexual_Partners',
    'First_Sexual_Activity_Age','Early_Sexual_Activity','Smoker',
    'STDs_History','Pap_Result_Positive','Risk_Score',
    'HighRisk_Combo','HighRisk_Sexual_Age','Smoking_STDs','HPV_Pap'
]


X = df[feature_cols].astype(float)
print("✅ Feature Engineering Completed!\n")

# === 10. Normalize numeric features ===
scaler = StandardScaler()
num_cols = ['Age','Sexual Partners','First_Sexual_Activity_Age','Risk_Score']
X[num_cols] = scaler.fit_transform(X[num_cols])

# === 11. Train-test split (80% train, 20% test) ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === 12. Handle class imbalance with SMOTE ===
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# === 13. Train Random Forest ===
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train_res, y_train_res)
rf_pred = rf.predict(X_test)

print("Random Forest Results:")
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# === 14. Train XGBoost ===
xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)
xgb.fit(X_train_res, y_train_res)
xgb_pred = xgb.predict(X_test)

print("XGBoost Results:")
print(confusion_matrix(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))

# === 15. Ensemble (combine both models) ===
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    voting='soft',
    n_jobs=-1
)
ensemble.fit(X_train_res, y_train_res)
ensemble_pred = ensemble.predict(X_test)

print("Ensemble Results:")
print(confusion_matrix(y_test, ensemble_pred))
print(classification_report(y_test, ensemble_pred))

# === 16. Save results ===
df['RF_Predicted'] = rf.predict(X)
df['RF_Confidence'] = rf.predict_proba(X)[:,1]
df['XGB_Predicted'] = xgb.predict(X)
df['XGB_Confidence'] = xgb.predict_proba(X)[:,1]
df['Ensemble_Predicted'] = ensemble.predict(X)
df['Ensemble_Confidence'] = ensemble.predict_proba(X)[:,1]
df.to_excel("cervical_predictions_final.xlsx", index=False)
print("✅ Results exported to cervical_predictions_final.xlsx")
import joblib

# Save trained ensemble model
joblib.dump(ensemble, 'ensemble_model.pkl')

# Save StandardScaler
joblib.dump(scaler, 'scaler.pkl')

# Save list of feature columns
joblib.dump(feature_cols, 'feature_cols.pkl')

print("✅ .pkl files saved!")
