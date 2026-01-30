"""
Cervical Cancer Risk Prediction Module

This module provides functions to predict cervical cancer risk
based on patient features using trained machine learning models.
"""

import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Union


class RiskPredictor:
    """
    Cervical Cancer Risk Predictor
    
    Loads trained models and provides risk prediction functionality.
    """
    
    def __init__(self, models_dir='models'):
        """
        Initialize the risk predictor by loading models.
        
        Args:
            models_dir (str): Directory containing the trained model files
        """
        self.models_dir = models_dir
        self._load_models()
    
    def _load_models(self):
        """Load all required models and components"""
        try:
            self.rf_regressor = joblib.load(f'{self.models_dir}/rf_regressor.pkl')
            self.ensemble_classifier = joblib.load(f'{self.models_dir}/ensemble_classifier.pkl')
            self.scaler = joblib.load(f'{self.models_dir}/scaler.pkl')
            self.feature_cols = joblib.load(f'{self.models_dir}/feature_cols.pkl')
            self.risk_category_map = joblib.load(f'{self.models_dir}/risk_category_map.pkl')
            
            print("✅ Models loaded successfully!")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not find model files in '{self.models_dir}' directory. "
                f"Please ensure all .pkl files exist. Error: {e}"
            )
        except Exception as e:
            raise Exception(f"Error loading models: {e}")
    
    def validate_input(self, patient_data: Dict) -> bool:
        """
        Validate that patient data contains all required features.
        
        Args:
            patient_data (Dict): Dictionary containing patient features
            
        Returns:
            bool: True if valid, raises ValueError if not
        """
        required_features = set(self.feature_cols)
        provided_features = set(patient_data.keys())
        
        missing_features = required_features - provided_features
        
        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}. "
                f"Required features: {self.feature_cols}"
            )
        
        # Validate age range
        if 'Age' in patient_data:
            age = patient_data['Age']
            if not (0 < age < 120):
                raise ValueError(f"Invalid age: {age}. Must be between 0 and 120.")
        
        # Validate binary features
        binary_features = [
            'High_Sexual_Partners', 'Early_Sexual_Activity', 'Smoker',
            'STDs_History', 'Pap_Positive', 'HighRisk_Combo',
            'HighRisk_Sexual_Age', 'Smoking_STDs', 'Pap_Smoking'
        ]
        
        for feature in binary_features:
            if feature in patient_data:
                if patient_data[feature] not in [0, 1]:
                    raise ValueError(
                        f"Feature '{feature}' must be 0 or 1, got {patient_data[feature]}"
                    )
        
        return True
    
    def predict(self, patient_data: Dict) -> Dict:
        """
        Predict cervical cancer risk for a patient.
        
        Args:
            patient_data (Dict): Dictionary containing patient features
            
        Returns:
            Dict: Prediction results with risk score, category, probabilities, and recommendations
            
        Example:
            >>> predictor = RiskPredictor()
            >>> patient = {
            ...     'Age': 35, 'Age_Group': 2, 'Sexual Partners': 3,
            ...     'High_Sexual_Partners': 1, 'First_Sexual_Activity_Age': 17,
            ...     'Early_Sexual_Activity': 0, 'Smoker': 1, 'STDs_History': 0,
            ...     'Pap_Positive': 0, 'HighRisk_Combo': 0, 'HighRisk_Sexual_Age': 0,
            ...     'Smoking_STDs': 0, 'Pap_Smoking': 0
            ... }
            >>> result = predictor.predict(patient)
            >>> print(result['risk_score'])
            2.5
        """
        # Validate input
        self.validate_input(patient_data)
        
        # Prepare patient data
        patient_df = pd.DataFrame([patient_data])
        
        # Create a copy for scaling
        patient_scaled = patient_df.copy()
        
        # Normalize numeric features
        num_cols_to_scale = ['Age', 'Sexual Partners', 'First_Sexual_Activity_Age']
        patient_scaled[num_cols_to_scale] = self.scaler.transform(patient_df[num_cols_to_scale])
        
        # Ensure correct feature order
        X_patient = patient_scaled[self.feature_cols]
        
        # Make predictions
        risk_score = self.rf_regressor.predict(X_patient)[0]
        risk_category_code = self.ensemble_classifier.predict(X_patient)[0]
        risk_category = self.risk_category_map[risk_category_code]
        risk_proba = self.ensemble_classifier.predict_proba(X_patient)[0]
        
        # Get probabilities for each class
        class_labels = self.ensemble_classifier.classes_
        prob_dict = {
            self.risk_category_map[int(label)]: round(prob * 100, 2)
            for label, prob in zip(class_labels, risk_proba)
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(patient_data, risk_category)
        
        return {
            'risk_score': round(risk_score, 2),
            'risk_category': risk_category,
            'probabilities': prob_dict,
            'recommendations': recommendations,
            'confidence': max(prob_dict.values())  # Highest probability as confidence
        }
    
    def _generate_recommendations(self, patient_data: Dict, risk_category: str) -> List[str]:
        """
        Generate personalized recommendations based on patient data and risk level.
        
        Args:
            patient_data (Dict): Patient features
            risk_category (str): Risk category (Low/Medium/High)
            
        Returns:
            List[str]: List of recommendation strings
        """
        recommendations = []
        
        # Base recommendations by category
        if risk_category == 'High':
            recommendations.append("⚠️ HIGH RISK ALERT:")
            recommendations.append("  • Schedule IMMEDIATE gynecological consultation")
            recommendations.append("  • Annual Pap smear and HPV testing mandatory")
            recommendations.append("  • Consider HPV vaccination if not already done")
            recommendations.append("  • Discuss aggressive screening schedule with physician")
        elif risk_category == 'Medium':
            recommendations.append("⚡ MEDIUM RISK:")
            recommendations.append("  • Schedule check-up within 3-6 months")
            recommendations.append("  • Pap smear every 1-2 years recommended")
            recommendations.append("  • Monitor risk factors closely")
        else:
            recommendations.append("✓ LOW RISK:")
            recommendations.append("  • Continue routine screening every 3 years")
            recommendations.append("  • Maintain current healthy practices")
        
        # Personalized recommendations based on specific risk factors
        if patient_data.get('Smoker', 0) == 1:
            recommendations.append("\n🚭 SMOKING CESSATION URGENT:")
            recommendations.append("  • Enroll in smoking cessation program immediately")
            recommendations.append("  • Smoking increases cervical cancer risk by 2-3x")
            recommendations.append("  • Consider nicotine replacement therapy")
            recommendations.append("  • Join support groups for better success rates")
        
        if patient_data.get('High_Sexual_Partners', 0) == 1:
            recommendations.append("\n🛡️ SEXUAL HEALTH:")
            recommendations.append("  • Use barrier protection consistently")
            recommendations.append("  • Increase STD screening frequency (every 6 months)")
            recommendations.append("  • Discuss HPV vaccination status with doctor")
            recommendations.append("  • Practice safe sex consistently")
        
        if patient_data.get('Pap_Positive', 0) == 1:
            recommendations.append("\n📋 ABNORMAL PAP SMEAR:")
            recommendations.append("  • Follow up with colposcopy ASAP")
            recommendations.append("  • Discuss treatment options with physician")
            recommendations.append("  • Do not delay follow-up appointments")
            recommendations.append("  • May require biopsy or further testing")
        
        if patient_data.get('STDs_History', 0) == 1:
            recommendations.append("\n💊 STD HISTORY:")
            recommendations.append("  • Complete all STD treatments as prescribed")
            recommendations.append("  • Quarterly STD screenings recommended")
            recommendations.append("  • Inform sexual partners about screening importance")
            recommendations.append("  • Consider STD counseling")
        
        if patient_data.get('Early_Sexual_Activity', 0) == 1:
            recommendations.append("\n📊 EARLY SEXUAL ACTIVITY:")
            recommendations.append("  • Increased monitoring recommended")
            recommendations.append("  • Annual gynecological exams starting now")
            recommendations.append("  • More frequent Pap smears may be needed")
        
        if patient_data.get('HighRisk_Combo', 0) == 1:
            recommendations.append("\n⚠️ MULTIPLE RISK FACTORS:")
            recommendations.append("  • Combination of risk factors detected")
            recommendations.append("  • More aggressive screening protocol needed")
            recommendations.append("  • Consider genetic counseling if family history present")
        
        # General prevention recommendations
        recommendations.append("\n📌 GENERAL PREVENTION:")
        recommendations.append("  • Maintain healthy BMI (18.5-24.9)")
        recommendations.append("  • Diet rich in fruits, vegetables, and whole grains")
        recommendations.append("  • Regular exercise (150 minutes/week minimum)")
        recommendations.append("  • Limit alcohol consumption (≤1 drink/day)")
        recommendations.append("  • Practice safe sex consistently")
        recommendations.append("  • Stay up to date with vaccinations")
        recommendations.append("  • Manage stress through healthy coping mechanisms")
        recommendations.append("  • Get adequate sleep (7-9 hours/night)")
        
        return recommendations
    
    def predict_batch(self, patients: List[Dict]) -> List[Dict]:
        """
        Predict risk for multiple patients.
        
        Args:
            patients (List[Dict]): List of patient data dictionaries
            
        Returns:
            List[Dict]: List of prediction results
        """
        return [self.predict(patient) for patient in patients]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the regression model.
        
        Returns:
            pd.DataFrame: DataFrame with features and their importance scores
        """
        importance_df = pd.DataFrame({
            'Feature': self.feature_cols,
            'Importance': self.rf_regressor.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return importance_df


# Convenience function for quick predictions
def predict_risk(patient_data: Dict, models_dir: str = 'models') -> Dict:
    """
    Convenience function to predict risk without instantiating RiskPredictor.
    
    Args:
        patient_data (Dict): Patient features
        models_dir (str): Directory containing model files
        
    Returns:
        Dict: Prediction results
    """
    predictor = RiskPredictor(models_dir=models_dir)
    return predictor.predict(patient_data)


# Example usage
if __name__ == "__main__":
    # Example patient data
    example_patient = {
        'Age': 35,
        'Age_Group': 2,
        'Sexual Partners': 4,
        'High_Sexual_Partners': 1,
        'First_Sexual_Activity_Age': 15,
        'Early_Sexual_Activity': 1,
        'Smoker': 1,
        'STDs_History': 0,
        'Pap_Positive': 1,
        'HighRisk_Combo': 1,
        'HighRisk_Sexual_Age': 1,
        'Smoking_STDs': 0,
        'Pap_Smoking': 1
    }
    
    try:
        # Initialize predictor
        predictor = RiskPredictor()
        
        # Make prediction
        result = predictor.predict(example_patient)
        
        # Display results
        print("\n" + "="*70)
        print("CERVICAL CANCER RISK ASSESSMENT")
        print("="*70)
        
        print(f"\n📊 RISK SCORE: {result['risk_score']}/5.0")
        print(f"📈 RISK CATEGORY: {result['risk_category']}")
        print(f"🎯 CONFIDENCE: {result['confidence']:.1f}%")
        
        print(f"\n🎲 RISK PROBABILITIES:")
        for category, prob in sorted(result['probabilities'].items(), 
                                     key=lambda x: x[1], reverse=True):
            print(f"  {category}: {prob:.1f}%")
        
        print(f"\n💡 PERSONALIZED RECOMMENDATIONS:")
        for rec in result['recommendations']:
            print(rec)
        
        print("\n" + "="*70)
        
        # Show feature importance
        print("\n📊 FEATURE IMPORTANCE:")
        importance = predictor.get_feature_importance()
        print(importance.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error: {e}")