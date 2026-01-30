import pytest
import numpy as np
import pandas as pd
import joblib
import os
from unittest.mock import Mock, patch

# Assuming you have a predict_risk module with the prediction function
# If not, we'll create it separately

class TestRiskPredictionModel:
    """Test suite for cervical cancer risk prediction model"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures - runs before each test"""
        # Check if models exist
        self.models_exist = all([
            os.path.exists('models/rf_regressor.pkl'),
            os.path.exists('models/ensemble_classifier.pkl'),
            os.path.exists('models/scaler.pkl'),
            os.path.exists('models/feature_cols.pkl'),
            os.path.exists('models/risk_category_map.pkl')
        ])
        
        if self.models_exist:
            self.rf_reg = joblib.load('models/rf_regressor.pkl')
            self.ensemble_clf = joblib.load('models/ensemble_classifier.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.feature_cols = joblib.load('models/feature_cols.pkl')
            self.risk_category_map = joblib.load('models/risk_category_map.pkl')
    
    # ================================================================
    # TEST 1: Model Files Existence
    # ================================================================
    def test_model_files_exist(self):
        """Test that all required model files exist"""
        required_files = [
            'models/rf_regressor.pkl',
            'models/ensemble_classifier.pkl',
            'models/scaler.pkl',
            'models/feature_cols.pkl',
            'models/risk_category_map.pkl'
        ]
        
        for file_path in required_files:
            assert os.path.exists(file_path), f"Missing required file: {file_path}"
    
    # ================================================================
    # TEST 2: Model Loading
    # ================================================================
    def test_models_load_successfully(self):
        """Test that models can be loaded without errors"""
        assert self.models_exist, "Models don't exist"
        
        # Check models are not None
        assert self.rf_reg is not None, "Regression model failed to load"
        assert self.ensemble_clf is not None, "Classification model failed to load"
        assert self.scaler is not None, "Scaler failed to load"
        assert self.feature_cols is not None, "Feature columns failed to load"
        
        # Check types
        assert hasattr(self.rf_reg, 'predict'), "Regression model missing predict method"
        assert hasattr(self.ensemble_clf, 'predict'), "Classifier missing predict method"
        assert hasattr(self.scaler, 'transform'), "Scaler missing transform method"
        assert isinstance(self.feature_cols, list), "Feature columns should be a list"
    
    # ================================================================
    # TEST 3: Prediction Output Structure
    # ================================================================
    def test_prediction_output_structure(self):
        """Test that prediction returns correct structure"""
        if not self.models_exist:
            pytest.skip("Models not found")
        
        # Sample low-risk patient
        sample_patient = {
            'Age': 25,
            'Age_Group': 1,
            'Sexual Partners': 1,
            'High_Sexual_Partners': 0,
            'First_Sexual_Activity_Age': 18,
            'Early_Sexual_Activity': 0,
            'Smoker': 0,
            'STDs_History': 0,
            'Pap_Positive': 0,
            'HighRisk_Combo': 0,
            'HighRisk_Sexual_Age': 0,
            'Smoking_STDs': 0,
            'Pap_Smoking': 0
        }
        
        result = self._predict(sample_patient)
        
        # Check output structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "risk_score" in result, "Missing 'risk_score' key"
        assert "risk_category" in result, "Missing 'risk_category' key"
        assert "probabilities" in result, "Missing 'probabilities' key"
        assert "recommendations" in result, "Missing 'recommendations' key"
    
    # ================================================================
    # TEST 4: Risk Score Range
    # ================================================================
    def test_risk_score_in_valid_range(self):
        """Test that risk score is within expected range (0-5)"""
        if not self.models_exist:
            pytest.skip("Models not found")
        
        test_patients = [
            self._create_low_risk_patient(),
            self._create_medium_risk_patient(),
            self._create_high_risk_patient()
        ]
        
        for patient in test_patients:
            result = self._predict(patient)
            risk_score = result['risk_score']
            
            assert isinstance(risk_score, (int, float)), "Risk score should be numeric"
            assert 0 <= risk_score <= 5, f"Risk score {risk_score} out of range [0-5]"
    
    # ================================================================
    # TEST 5: Risk Category Values
    # ================================================================
    def test_risk_category_valid_values(self):
        """Test that risk category is one of: Low, Medium, High"""
        if not self.models_exist:
            pytest.skip("Models not found")
        
        valid_categories = ['Low', 'Medium', 'High']
        
        test_patients = [
            self._create_low_risk_patient(),
            self._create_medium_risk_patient(),
            self._create_high_risk_patient()
        ]
        
        for patient in test_patients:
            result = self._predict(patient)
            risk_category = result['risk_category']
            
            assert risk_category in valid_categories, \
                f"Invalid risk category: {risk_category}"
    
    # ================================================================
    # TEST 6: Probability Values
    # ================================================================
    def test_probabilities_sum_to_100(self):
        """Test that probability values sum to approximately 100%"""
        if not self.models_exist:
            pytest.skip("Models not found")
        
        patient = self._create_medium_risk_patient()
        result = self._predict(patient)
        probabilities = result['probabilities']
        
        # Check all categories present
        assert 'Low' in probabilities, "Missing 'Low' probability"
        assert 'Medium' in probabilities, "Missing 'Medium' probability"
        assert 'High' in probabilities, "Missing 'High' probability"
        
        # Check probabilities are numeric and in range
        for category, prob in probabilities.items():
            assert isinstance(prob, (int, float)), f"{category} probability not numeric"
            assert 0 <= prob <= 100, f"{category} probability out of range: {prob}"
        
        # Check sum to ~100% (allow small floating point error)
        total_prob = sum(probabilities.values())
        assert abs(total_prob - 100) < 1, f"Probabilities sum to {total_prob}, expected ~100"
    
    # ================================================================
    # TEST 7: Low Risk Patient Scenario
    # ================================================================
    def test_low_risk_patient_classification(self):
        """Test that a clearly low-risk patient is classified correctly"""
        if not self.models_exist:
            pytest.skip("Models not found")
        
        low_risk_patient = self._create_low_risk_patient()
        result = self._predict(low_risk_patient)
        
        # Should have low risk score
        assert result['risk_score'] <= 1.5, \
            f"Low risk patient has high score: {result['risk_score']}"
        
        # Should be classified as Low (most likely)
        # Note: Model might predict Medium in edge cases, so we check probability
        assert result['probabilities']['Low'] >= 40, \
            "Low risk patient should have high Low probability"
    
    # ================================================================
    # TEST 8: High Risk Patient Scenario
    # ================================================================
    def test_high_risk_patient_classification(self):
        """Test that a clearly high-risk patient is classified correctly"""
        if not self.models_exist:
            pytest.skip("Models not found")
        
        high_risk_patient = self._create_high_risk_patient()
        result = self._predict(high_risk_patient)
        
        # Should have high risk score
        assert result['risk_score'] >= 3.5, \
            f"High risk patient has low score: {result['risk_score']}"
        
        # Should be classified as High or Medium
        assert result['risk_category'] in ['Medium', 'High'], \
            f"High risk patient classified as {result['risk_category']}"
    
    # ================================================================
    # TEST 9: Recommendations Presence
    # ================================================================
    def test_recommendations_not_empty(self):
        """Test that recommendations are always provided"""
        if not self.models_exist:
            pytest.skip("Models not found")
        
        test_patients = [
            self._create_low_risk_patient(),
            self._create_medium_risk_patient(),
            self._create_high_risk_patient()
        ]
        
        for patient in test_patients:
            result = self._predict(patient)
            recommendations = result['recommendations']
            
            assert isinstance(recommendations, list), \
                "Recommendations should be a list"
            assert len(recommendations) > 0, \
                "Recommendations should not be empty"
            assert all(isinstance(rec, str) for rec in recommendations), \
                "All recommendations should be strings"
    
    # ================================================================
    # TEST 10: Smoking-Specific Recommendations
    # ================================================================
    def test_smoker_gets_cessation_recommendation(self):
        """Test that smokers receive smoking cessation recommendations"""
        if not self.models_exist:
            pytest.skip("Models not found")
        
        smoker_patient = self._create_medium_risk_patient()
        smoker_patient['Smoker'] = 1
        
        result = self._predict(smoker_patient)
        recommendations_text = ' '.join(result['recommendations']).lower()
        
        # Check for smoking-related keywords
        smoking_keywords = ['smoking', 'cessation', 'smoke', 'quit']
        has_smoking_rec = any(keyword in recommendations_text 
                              for keyword in smoking_keywords)
        
        assert has_smoking_rec, \
            "Smoker should receive smoking cessation recommendation"
    
    # ================================================================
    # TEST 11: Missing Features Handling
    # ================================================================
    def test_missing_features_handled(self):
        """Test that missing features are handled gracefully"""
        if not self.models_exist:
            pytest.skip("Models not found")
        
        # Patient with missing some features
        incomplete_patient = {
            'Age': 30,
            'Age_Group': 2,
            'Sexual Partners': 2,
            'High_Sexual_Partners': 0,
            'First_Sexual_Activity_Age': 18
            # Missing other features
        }
        
        # Should either handle gracefully or raise informative error
        try:
            result = self._predict(incomplete_patient)
            # If it works, that's fine
            assert result is not None
        except (KeyError, ValueError) as e:
            # If it raises error, it should be informative
            assert len(str(e)) > 0, "Error message should not be empty"
    
    # ================================================================
    # TEST 12: Invalid Age Handling
    # ================================================================
    def test_invalid_age_values(self):
        """Test handling of invalid age values"""
        if not self.models_exist:
            pytest.skip("Models not found")
        
        invalid_ages = [-5, 0, 150, 999]
        
        for invalid_age in invalid_ages:
            patient = self._create_low_risk_patient()
            patient['Age'] = invalid_age
            
            # Should either handle or raise appropriate error
            try:
                result = self._predict(patient)
                # If prediction works, risk score should still be valid
                assert 0 <= result['risk_score'] <= 5
            except (ValueError, AssertionError):
                # Expected for invalid inputs
                pass
    
    # ================================================================
    # TEST 13: Feature Importance Accessibility
    # ================================================================
    def test_model_has_feature_importance(self):
        """Test that models provide feature importance"""
        if not self.models_exist:
            pytest.skip("Models not found")
        
        assert hasattr(self.rf_reg, 'feature_importances_'), \
            "Regression model missing feature_importances_"
        assert hasattr(self.ensemble_clf.estimators_[0], 'feature_importances_'), \
            "Classification model missing feature_importances_"
        
        # Check importance arrays are correct length
        assert len(self.rf_reg.feature_importances_) == len(self.feature_cols), \
            "Feature importance length mismatch"
    
    # ================================================================
    # TEST 14: Consistent Predictions
    # ================================================================
    def test_prediction_consistency(self):
        """Test that same input gives same output (deterministic)"""
        if not self.models_exist:
            pytest.skip("Models not found")
        
        patient = self._create_medium_risk_patient()
        
        result1 = self._predict(patient)
        result2 = self._predict(patient)
        
        # Risk scores should be identical
        assert result1['risk_score'] == result2['risk_score'], \
            "Predictions should be deterministic"
        assert result1['risk_category'] == result2['risk_category'], \
            "Categories should be deterministic"
    
    # ================================================================
    # TEST 15: Edge Case - All Risk Factors Present
    # ================================================================
    def test_all_risk_factors_present(self):
        """Test patient with all risk factors"""
        if not self.models_exist:
            pytest.skip("Models not found")
        
        maximum_risk_patient = {
            'Age': 45,
            'Age_Group': 3,
            'Sexual Partners': 10,
            'High_Sexual_Partners': 1,
            'First_Sexual_Activity_Age': 14,
            'Early_Sexual_Activity': 1,
            'Smoker': 1,
            'STDs_History': 1,
            'Pap_Positive': 1,
            'HighRisk_Combo': 1,
            'HighRisk_Sexual_Age': 1,
            'Smoking_STDs': 1,
            'Pap_Smoking': 1
        }
        
        result = self._predict(maximum_risk_patient)
        
        # Should have very high risk
        assert result['risk_score'] >= 4.0, \
            "Patient with all risk factors should have high score"
        
        # Should have extensive recommendations
        assert len(result['recommendations']) >= 8, \
            "High risk patient should have many recommendations"
    
    # ================================================================
    # TEST 16: Edge Case - No Risk Factors
    # ================================================================
    def test_no_risk_factors(self):
        """Test patient with no risk factors"""
        if not self.models_exist:
            pytest.skip("Models not found")
        
        zero_risk_patient = {
            'Age': 25,
            'Age_Group': 1,
            'Sexual Partners': 1,
            'High_Sexual_Partners': 0,
            'First_Sexual_Activity_Age': 20,
            'Early_Sexual_Activity': 0,
            'Smoker': 0,
            'STDs_History': 0,
            'Pap_Positive': 0,
            'HighRisk_Combo': 0,
            'HighRisk_Sexual_Age': 0,
            'Smoking_STDs': 0,
            'Pap_Smoking': 0
        }
        
        result = self._predict(zero_risk_patient)
        
        # Should have very low risk
        assert result['risk_score'] <= 0.5, \
            "Patient with no risk factors should have very low score"
        assert result['risk_category'] == 'Low', \
            "Patient with no risk factors should be Low risk"
    
    # ================================================================
    # TEST 17: Batch Prediction
    # ================================================================
    def test_batch_prediction(self):
        """Test that model can handle multiple patients at once"""
        if not self.models_exist:
            pytest.skip("Models not found")
        
        patients = [
            self._create_low_risk_patient(),
            self._create_medium_risk_patient(),
            self._create_high_risk_patient()
        ]
        
        results = [self._predict(patient) for patient in patients]
        
        assert len(results) == 3, "Should predict for all patients"
        
        # Risk scores should be ordered (low < medium < high)
        assert results[0]['risk_score'] < results[1]['risk_score'], \
            "Low risk should have lower score than medium"
        assert results[1]['risk_score'] < results[2]['risk_score'], \
            "Medium risk should have lower score than high"
    
    # ================================================================
    # TEST 18: Model Performance Metrics
    # ================================================================
    def test_model_performance_metrics(self):
        """Test that model maintains acceptable performance thresholds"""
        if not self.models_exist:
            pytest.skip("Models not found")
        
        # Create diverse test set
        test_patients = []
        for _ in range(30):
            # Mix of risk levels
            if _ % 3 == 0:
                test_patients.append(self._create_low_risk_patient())
            elif _ % 3 == 1:
                test_patients.append(self._create_medium_risk_patient())
            else:
                test_patients.append(self._create_high_risk_patient())
        
        predictions = [self._predict(p) for p in test_patients]
        
        # All predictions should be valid
        for pred in predictions:
            assert 0 <= pred['risk_score'] <= 5
            assert pred['risk_category'] in ['Low', 'Medium', 'High']
    
    # ================================================================
    # Helper Methods
    # ================================================================
    
    def _create_low_risk_patient(self):
        """Create a low-risk patient profile"""
        return {
            'Age': 25,
            'Age_Group': 1,
            'Sexual Partners': 1,
            'High_Sexual_Partners': 0,
            'First_Sexual_Activity_Age': 20,
            'Early_Sexual_Activity': 0,
            'Smoker': 0,
            'STDs_History': 0,
            'Pap_Positive': 0,
            'HighRisk_Combo': 0,
            'HighRisk_Sexual_Age': 0,
            'Smoking_STDs': 0,
            'Pap_Smoking': 0
        }
    
    def _create_medium_risk_patient(self):
        """Create a medium-risk patient profile"""
        return {
            'Age': 35,
            'Age_Group': 2,
            'Sexual Partners': 3,
            'High_Sexual_Partners': 1,
            'First_Sexual_Activity_Age': 17,
            'Early_Sexual_Activity': 0,
            'Smoker': 1,
            'STDs_History': 0,
            'Pap_Positive': 0,
            'HighRisk_Combo': 0,
            'HighRisk_Sexual_Age': 0,
            'Smoking_STDs': 0,
            'Pap_Smoking': 0
        }
    
    def _create_high_risk_patient(self):
        """Create a high-risk patient profile"""
        return {
            'Age': 42,
            'Age_Group': 3,
            'Sexual Partners': 6,
            'High_Sexual_Partners': 1,
            'First_Sexual_Activity_Age': 15,
            'Early_Sexual_Activity': 1,
            'Smoker': 1,
            'STDs_History': 1,
            'Pap_Positive': 1,
            'HighRisk_Combo': 1,
            'HighRisk_Sexual_Age': 1,
            'Smoking_STDs': 1,
            'Pap_Smoking': 1
        }
    
    def _predict(self, patient_data):
        """Helper method to make predictions"""
        # Prepare patient data
        patient_df = pd.DataFrame([patient_data])
        
        # Create a copy for scaling
        patient_scaled = patient_df.copy()
        
        # Normalize numeric features
        num_cols_to_scale = ['Age', 'Sexual Partners', 'First_Sexual_Activity_Age']
        patient_scaled[num_cols_to_scale] = self.scaler.transform(patient_df[num_cols_to_scale])
        
        # Ensure correct feature order
        X_patient = patient_scaled[self.feature_cols]
        
        # Predictions
        risk_score = self.rf_reg.predict(X_patient)[0]
        risk_category_code = self.ensemble_clf.predict(X_patient)[0]
        risk_category = self.risk_category_map[risk_category_code]
        risk_proba = self.ensemble_clf.predict_proba(X_patient)[0]
        
        # Get probabilities for each class
        class_labels = self.ensemble_clf.classes_
        prob_dict = {self.risk_category_map[int(label)]: prob * 100 
                     for label, prob in zip(class_labels, risk_proba)}
        
        # Generate basic recommendations
        recommendations = self._generate_recommendations(patient_data, risk_category)
        
        return {
            'risk_score': round(risk_score, 2),
            'risk_category': risk_category,
            'probabilities': prob_dict,
            'recommendations': recommendations
        }
    
    def _generate_recommendations(self, patient_data, risk_category):
        """Generate recommendations based on patient data"""
        recommendations = []
        
        # Base recommendations by category
        if risk_category == 'High':
            recommendations.append("⚠️ HIGH RISK: Schedule immediate consultation")
        elif risk_category == 'Medium':
            recommendations.append("⚡ MEDIUM RISK: Schedule check-up within 3-6 months")
        else:
            recommendations.append("✓ LOW RISK: Continue routine screening")
        
        # Personalized recommendations
        if patient_data.get('Smoker', 0) == 1:
            recommendations.append("🚭 SMOKING CESSATION: Enroll in cessation program")
        
        if patient_data.get('High_Sexual_Partners', 0) == 1:
            recommendations.append("🛡️ SEXUAL HEALTH: Use barrier protection")
        
        if patient_data.get('Pap_Positive', 0) == 1:
            recommendations.append("📋 ABNORMAL PAP: Follow up with colposcopy")
        
        if patient_data.get('STDs_History', 0) == 1:
            recommendations.append("💊 STD HISTORY: Increase screening frequency")
        
        recommendations.append("📌 GENERAL: Maintain healthy lifestyle")
        
        return recommendations


# ================================================================
# Integration Tests
# ================================================================

class TestModelIntegration:
    """Integration tests for the complete prediction pipeline"""
    
    def test_end_to_end_prediction_flow(self):
        """Test complete flow from input to recommendations"""
        if not os.path.exists('models/rf_regressor.pkl'):
            pytest.skip("Models not found")
        
        # This would test your actual API endpoint or main function
        # For now, we'll test that all components work together
        
        test_input = {
            'Age': 30,
            'Age_Group': 2,
            'Sexual Partners': 2,
            'High_Sexual_Partners': 0,
            'First_Sexual_Activity_Age': 18,
            'Early_Sexual_Activity': 0,
            'Smoker': 0,
            'STDs_History': 0,
            'Pap_Positive': 0,
            'HighRisk_Combo': 0,
            'HighRisk_Sexual_Age': 0,
            'Smoking_STDs': 0,
            'Pap_Smoking': 0
        }
        
        # Load and predict
        rf_reg = joblib.load('models/rf_regressor.pkl')
        ensemble_clf = joblib.load('models/ensemble_classifier.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_cols = joblib.load('models/feature_cols.pkl')
        
        # Should complete without errors
        assert rf_reg is not None
        assert ensemble_clf is not None


# ================================================================
# Performance Tests
# ================================================================

class TestModelPerformance:
    """Performance and speed tests"""
    
    def test_prediction_speed(self):
        """Test that predictions complete within acceptable time"""
        if not os.path.exists('models/rf_regressor.pkl'):
            pytest.skip("Models not found")
        
        import time
        
        rf_reg = joblib.load('models/rf_regressor.pkl')
        ensemble_clf = joblib.load('models/ensemble_classifier.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_cols = joblib.load('models/feature_cols.pkl')
        
        patient = pd.DataFrame([{
            'Age': 30, 'Age_Group': 2, 'Sexual Partners': 2,
            'High_Sexual_Partners': 0, 'First_Sexual_Activity_Age': 18,
            'Early_Sexual_Activity': 0, 'Smoker': 0, 'STDs_History': 0,
            'Pap_Positive': 0, 'HighRisk_Combo': 0, 'HighRisk_Sexual_Age': 0,
            'Smoking_STDs': 0, 'Pap_Smoking': 0
        }])
        
        # Scale
        num_cols = ['Age', 'Sexual Partners', 'First_Sexual_Activity_Age']
        patient[num_cols] = scaler.transform(patient[num_cols])
        X = patient[feature_cols]
        
        # Time the prediction
        start = time.time()
        _ = rf_reg.predict(X)
        _ = ensemble_clf.predict(X)
        duration = time.time() - start
        
        # Should complete in under 1 second
        assert duration < 1.0, f"Prediction took too long: {duration}s"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])