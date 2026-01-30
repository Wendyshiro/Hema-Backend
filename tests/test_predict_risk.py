"""
Simple test file for cervical cancer risk prediction model.
Run with: pytest test_predict_risk.py -v
"""

import pytest
from predict_risk import predict_risk, RiskPredictor


class TestPredictRiskBasic:
    """Basic tests for the predict_risk function"""
    
    def test_predict_risk_output_structure(self):
        """Test that predict_risk returns expected structure"""
        sample_input = {
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
        
        result = predict_risk(sample_input)
        
        # Check it returns a dict with expected keys
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "risk_score" in result, "Missing 'risk_score' key"
        assert "risk_category" in result, "Missing 'risk_category' key"
        assert "probabilities" in result, "Missing 'probabilities' key"
        assert "recommendations" in result, "Missing 'recommendations' key"
    
    def test_risk_score_range(self):
        """Test that risk score is within valid range"""
        sample_input = {
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
        
        result = predict_risk(sample_input)
        risk_score = result['risk_score']
        
        assert isinstance(risk_score, (int, float)), "Risk score should be numeric"
        assert 0 <= risk_score <= 5, f"Risk score {risk_score} out of range [0-5]"
    
    def test_risk_category_values(self):
        """Test that risk category is valid"""
        sample_input = {
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
        
        result = predict_risk(sample_input)
        risk_category = result['risk_category']
        
        valid_categories = ['Low', 'Medium', 'High']
        assert risk_category in valid_categories, \
            f"Invalid risk category: {risk_category}"
    
    def test_probabilities_present(self):
        """Test that probabilities are provided for all categories"""
        sample_input = {
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
        
        result = predict_risk(sample_input)
        probabilities = result['probabilities']
        
        assert 'Low' in probabilities, "Missing 'Low' probability"
        assert 'Medium' in probabilities, "Missing 'Medium' probability"
        assert 'High' in probabilities, "Missing 'High' probability"
        
        # Check probabilities sum to approximately 100%
        total_prob = sum(probabilities.values())
        assert abs(total_prob - 100) < 1, f"Probabilities sum to {total_prob}, expected ~100"
    
    def test_recommendations_not_empty(self):
        """Test that recommendations are always provided"""
        sample_input = {
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
        
        result = predict_risk(sample_input)
        recommendations = result['recommendations']
        
        assert isinstance(recommendations, list), "Recommendations should be a list"
        assert len(recommendations) > 0, "Recommendations should not be empty"


class TestPredictRiskScenarios:
    """Test different patient risk scenarios"""
    
    def test_low_risk_patient(self):
        """Test prediction for low-risk patient"""
        low_risk_patient = {
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
        
        result = predict_risk(low_risk_patient)
        
        # Should have low risk score
        assert result['risk_score'] <= 2.0, \
            f"Low risk patient has high score: {result['risk_score']}"
        
        # Should likely be classified as Low
        assert result['probabilities']['Low'] >= 30, \
            "Low risk patient should have significant Low probability"
    
    def test_high_risk_patient(self):
        """Test prediction for high-risk patient"""
        high_risk_patient = {
            'Age': 45,
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
        
        result = predict_risk(high_risk_patient)
        
        # Should have high risk score
        assert result['risk_score'] >= 3.0, \
            f"High risk patient has low score: {result['risk_score']}"
        
        # Should be classified as Medium or High
        assert result['risk_category'] in ['Medium', 'High'], \
            f"High risk patient classified as {result['risk_category']}"
    
    def test_smoker_gets_recommendation(self):
        """Test that smokers receive smoking cessation recommendations"""
        smoker_patient = {
            'Age': 30,
            'Age_Group': 2,
            'Sexual Partners': 2,
            'High_Sexual_Partners': 0,
            'First_Sexual_Activity_Age': 18,
            'Early_Sexual_Activity': 0,
            'Smoker': 1,  # Smoker
            'STDs_History': 0,
            'Pap_Positive': 0,
            'HighRisk_Combo': 0,
            'HighRisk_Sexual_Age': 0,
            'Smoking_STDs': 0,
            'Pap_Smoking': 0
        }
        
        result = predict_risk(smoker_patient)
        recommendations_text = ' '.join(result['recommendations']).lower()
        
        # Check for smoking-related keywords
        smoking_keywords = ['smoking', 'cessation', 'smoke', 'quit']
        has_smoking_rec = any(keyword in recommendations_text 
                              for keyword in smoking_keywords)
        
        assert has_smoking_rec, \
            "Smoker should receive smoking cessation recommendation"


class TestPredictRiskEdgeCases:
    """Test edge cases and error handling"""
    
    def test_missing_features_raises_error(self):
        """Test that missing features raise appropriate error"""
        incomplete_patient = {
            'Age': 30,
            'Sexual Partners': 2,
            # Missing other required features
        }
        
        with pytest.raises(ValueError) as exc_info:
            predict_risk(incomplete_patient)
        
        assert "Missing required features" in str(exc_info.value)
    
    def test_invalid_age_raises_error(self):
        """Test that invalid age raises error"""
        invalid_patient = {
            'Age': -5,  # Invalid age
            'Age_Group': 0,
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
        
        with pytest.raises(ValueError) as exc_info:
            predict_risk(invalid_patient)
        
        assert "Invalid age" in str(exc_info.value)
    
    def test_invalid_binary_feature_raises_error(self):
        """Test that invalid binary feature values raise error"""
        invalid_patient = {
            'Age': 30,
            'Age_Group': 2,
            'Sexual Partners': 2,
            'High_Sexual_Partners': 0,
            'First_Sexual_Activity_Age': 18,
            'Early_Sexual_Activity': 0,
            'Smoker': 5,  # Invalid: should be 0 or 1
            'STDs_History': 0,
            'Pap_Positive': 0,
            'HighRisk_Combo': 0,
            'HighRisk_Sexual_Age': 0,
            'Smoking_STDs': 0,
            'Pap_Smoking': 0
        }
        
        with pytest.raises(ValueError) as exc_info:
            predict_risk(invalid_patient)
        
        assert "must be 0 or 1" in str(exc_info.value)


class TestRiskPredictorClass:
    """Test the RiskPredictor class"""
    
    def test_predictor_initialization(self):
        """Test that RiskPredictor initializes correctly"""
        predictor = RiskPredictor()
        
        assert predictor.rf_regressor is not None
        assert predictor.ensemble_classifier is not None
        assert predictor.scaler is not None
        assert predictor.feature_cols is not None
    
    def test_batch_prediction(self):
        """Test batch prediction functionality"""
        predictor = RiskPredictor()
        
        patients = [
            {
                'Age': 25, 'Age_Group': 1, 'Sexual Partners': 1,
                'High_Sexual_Partners': 0, 'First_Sexual_Activity_Age': 20,
                'Early_Sexual_Activity': 0, 'Smoker': 0, 'STDs_History': 0,
                'Pap_Positive': 0, 'HighRisk_Combo': 0, 'HighRisk_Sexual_Age': 0,
                'Smoking_STDs': 0, 'Pap_Smoking': 0
            },
            {
                'Age': 40, 'Age_Group': 3, 'Sexual Partners': 5,
                'High_Sexual_Partners': 1, 'First_Sexual_Activity_Age': 16,
                'Early_Sexual_Activity': 1, 'Smoker': 1, 'STDs_History': 1,
                'Pap_Positive': 1, 'HighRisk_Combo': 1, 'HighRisk_Sexual_Age': 1,
                'Smoking_STDs': 1, 'Pap_Smoking': 1
            }
        ]
        
        results = predictor.predict_batch(patients)
        
        assert len(results) == 2, "Should predict for all patients"
        assert results[0]['risk_score'] < results[1]['risk_score'], \
            "Low risk patient should have lower score"
    
    def test_feature_importance(self):
        """Test feature importance retrieval"""
        predictor = RiskPredictor()
        importance = predictor.get_feature_importance()
        
        assert isinstance(importance, pd.DataFrame)
        assert 'Feature' in importance.columns
        assert 'Importance' in importance.columns
        assert len(importance) > 0


class TestPredictionConsistency:
    """Test prediction consistency"""
    
    def test_same_input_same_output(self):
        """Test that same input always gives same output"""
        patient = {
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
        
        result1 = predict_risk(patient)
        result2 = predict_risk(patient)
        
        assert result1['risk_score'] == result2['risk_score'], \
            "Predictions should be deterministic"
        assert result1['risk_category'] == result2['risk_category'], \
            "Categories should be deterministic"


# Additional helper for pytest
import pandas as pd

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])