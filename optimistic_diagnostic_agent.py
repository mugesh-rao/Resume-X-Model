import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

class OptimisticDiagnosticAgent:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.conditions = ['Heart Failure', 'Arrhythmia', 'Coronary Artery Disease']
        self.treatment_options = {
            'Heart Failure': ['ACE inhibitors', 'Beta blockers', 'Diuretics'],
            'Arrhythmia': ['Antiarrhythmic drugs', 'Blood thinners', 'Calcium channel blockers'],
            'Coronary Artery Disease': ['Statins', 'Aspirin', 'Nitrates']
        }
        
    def preprocess_data(self, patient_data):
        """Preprocess and structure patient data"""
        try:
            # Extract numerical data
            vitals = np.array(patient_data['vitals'])
            
            # Create feature vector
            features = np.concatenate([
                vitals.reshape(1, -1),
                np.array([[patient_data['age']]])
            ], axis=1)
            
            # Normalize features
            normalized_data = self.scaler.fit_transform(features)
            
            return normalized_data
            
        except Exception as e:
            print(f"Error in data preprocessing: {str(e)}")
            return None

    def analyze_patient_condition(self, patient_data):
        """Analyze patient condition"""
        try:
            # Process data
            processed_data = self.preprocess_data(patient_data)
            if processed_data is None:
                return None
                
            # Train classifier on synthetic data
            X_train = np.random.rand(100, processed_data.shape[1])
            y_train = np.random.choice(self.conditions, 100)
            self.rf_classifier.fit(X_train, y_train)
            
            # Generate predictions
            predictions = self.rf_classifier.predict_proba(processed_data)
            
            # Calculate confidence scores
            confidence_scores = np.max(predictions, axis=1)
            
            # Extract findings
            findings = self._extract_key_findings(patient_data)
            
            return {
                'predictions': predictions,
                'confidence': confidence_scores[0],
                'findings': findings,
                'condition': self.conditions[np.argmax(predictions[0])]
            }
            
        except Exception as e:
            print(f"Error in patient analysis: {str(e)}")
            return None

    def optimize_treatment_plan(self, patient_analysis, historical_data):
        """Optimize treatment recommendations"""
        try:
            if patient_analysis is None:
                return None
                
            condition = patient_analysis['condition']
            available_treatments = self.treatment_options.get(condition, [])
            
            # Simple treatment selection based on condition
            selected_treatments = available_treatments[:2]  # Select top 2 treatments
            
            return {
                'primary_treatment': selected_treatments[0],
                'secondary_treatment': selected_treatments[1],
                'condition': condition,
                'confidence': patient_analysis['confidence']
            }
            
        except Exception as e:
            print(f"Error in treatment optimization: {str(e)}")
            return None

    def visualize_diagnostics(self, patient_data, analysis_results):
        """Generate diagnostic visualizations"""
        try:
            if analysis_results is None:
                return
                
            plt.figure(figsize=(15, 10))
            
            # 1. Vitals Heatmap
            plt.subplot(2, 2, 1)
            self._plot_vitals_heatmap(patient_data)
            
            # 2. Confidence Gauge
            plt.subplot(2, 2, 2)
            self._plot_confidence_gauge(analysis_results['confidence'])
            
            # 3. Condition Probabilities
            plt.subplot(2, 2, 3)
            self._plot_condition_probabilities(analysis_results['predictions'][0])
            
            # 4. Risk Factors
            plt.subplot(2, 2, 4)
            self._plot_risk_factors(patient_data)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")

    def generate_diagnostic_report(self, patient_data, analysis_results, treatment_plan):
        """Generate diagnostic report"""
        try:
            if analysis_results is None or treatment_plan is None:
                return None
                
            report = {
                'patient_summary': {
                    'age': patient_data['age'],
                    'gender': patient_data['gender'],
                    'symptoms': patient_data['symptoms'],
                    'vitals': patient_data['vitals']
                },
                'diagnosis': {
                    'condition': analysis_results['condition'],
                    'confidence': f"{analysis_results['confidence']*100:.1f}%",
                    'findings': analysis_results['findings']
                },
                'treatment_plan': {
                    'primary': treatment_plan['primary_treatment'],
                    'secondary': treatment_plan['secondary_treatment']
                }
            }
            
            return report
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            return None

    # Helper Methods
    def _extract_key_findings(self, patient_data):
        """Extract key findings from patient data"""
        return [
            f"Age: {patient_data['age']}",
            f"Symptoms: {patient_data['symptoms']}",
            f"Medical History: {patient_data['medical_history']}",
            f"Lab Results: {patient_data['lab_results']}"
        ]

    def _plot_vitals_heatmap(self, patient_data):
        """Plot vitals heatmap"""
        vitals = np.array(patient_data['vitals'])
        vitals_df = pd.DataFrame(
            vitals,
            columns=['Systolic BP', 'Diastolic BP', 'Heart Rate', 'O2 Saturation']
        )
        sns.heatmap(vitals_df, annot=True, cmap='YlOrRd')
        plt.title('Patient Vitals Analysis')

    def _plot_confidence_gauge(self, confidence):
        """Plot confidence gauge"""
        plt.pie([confidence, 1-confidence], 
                colors=['green', 'lightgray'],
                labels=[f'{confidence*100:.1f}%', ''])
        plt.title('Diagnostic Confidence')

    def _plot_condition_probabilities(self, probabilities):
        """Plot condition probabilities"""
        plt.bar(self.conditions, probabilities)
        plt.title('Condition Probabilities')
        plt.xticks(rotation=45)

    def _plot_risk_factors(self, patient_data):
        """Plot risk factors"""
        risk_factors = ['Age > 50', 'Hypertension', 'Family History']
        risk_values = [1, 1, 1]  # Example values
        plt.bar(risk_factors, risk_values)
        plt.title('Risk Factors')
        plt.xticks(rotation=45)

# Usage Example
if __name__ == "__main__":
    # Initialize the diagnostic agent
    agent = OptimisticDiagnosticAgent()
    
    # Sample patient data
    patient_data = {
        'patient_id': 'P001',
        'age': 55,
        'gender': 'Male',
        'symptoms': 'Fatigue, Shortness of Breath, Irregular Heart Rate',
        'vitals': [[120, 80, 90, 98]],  # [Systolic BP, Diastolic BP, Heart Rate, O2 Saturation]
        'medical_history': 'Hypertension, Family History of Heart Disease',
        'lab_results': {
            'blood_work': 'Anemia',
            'ecg': 'Left Ventricular Dysfunction'
        },
        'medications': ['Beta Blockers', 'ACE Inhibitors']
    }
    
    # Run diagnostic process
    processed_data = agent.preprocess_data(patient_data)
    analysis_results = agent.analyze_patient_condition(patient_data)
    treatment_plan = agent.optimize_treatment_plan(analysis_results, None)
    
    # Generate visualizations and report
    agent.visualize_diagnostics(patient_data, analysis_results)
    diagnostic_report = agent.generate_diagnostic_report(
        patient_data, 
        analysis_results, 
        treatment_plan
    )
    
    # Print diagnostic report
    if diagnostic_report:
        print("\nDiagnostic Report")
        print("=" * 50)
        for section, content in diagnostic_report.items():
            print(f"\n{section.replace('_', ' ').title()}:")
            print("-" * 30)
            for key, value in content.items():
                print(f"{key}: {value}") 