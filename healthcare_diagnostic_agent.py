import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

class HealthcareAgent:
    def __init__(self):
        self.scaler = StandardScaler()
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gb_classifier = GradientBoostingClassifier(random_state=42)
        self.condition_stages = {
            'Heart Failure': ['Stage A', 'Stage B', 'Stage C', 'Stage D'],
            'Arrhythmia': ['Mild', 'Moderate', 'Severe'],
            'Coronary Artery Disease': ['Early', 'Intermediate', 'Advanced']
        }
        
    def analyze_patient_data(self, patient_data=None):
        """Step 1: Analyze patient condition and report"""
        if patient_data is None:
            patient_data = {
                'Patient_ID': ['P001'],
                'Age': [55],
                'Gender': ['Male'],
                'Symptoms': ['Fatigue, Shortness of Breath, Irregular Heart Rate'],
                'ECG': ['Left Ventricular Dysfunction'],
                'Blood_Work': ['Anemia, High BNP'],
                'Imaging': ['Mild Cardiomegaly'],
                'History': ['Hypertension, Family History of Heart Disease'],
                'Medications': ['Beta Blockers, ACE Inhibitors'],
                'Vitals': [[120, 80, 90, 98]],  # [Systolic BP, Diastolic BP, Heart Rate, O2 Saturation]
                'Risk_Factors': ['Smoking, Obesity'],
                'Last_Visit': ['2023-12-01']
            }
        
        self.patient_df = pd.DataFrame(patient_data)
        
        # Analyze vitals
        vitals_array = np.array(self.patient_df['Vitals'].tolist())
        self.vitals_scaled = self.scaler.fit_transform(vitals_array)
        
        # Calculate risk score
        self.risk_score = self._calculate_risk_score()
        
        print("\nStep 1: Patient Data Analysis")
        print("=" * 50)
        print("Patient Profile Summary:")
        for col in self.patient_df.columns:
            if col != 'Vitals':
                print(f"{col}: {self.patient_df[col].iloc[0]}")
        print(f"\nVital Signs Analysis:")
        print(f"Blood Pressure: {vitals_array[0][0]}/{vitals_array[0][1]} mmHg")
        print(f"Heart Rate: {vitals_array[0][2]} bpm")
        print(f"O2 Saturation: {vitals_array[0][3]}%")
        print(f"\nInitial Risk Score: {self.risk_score:.2f}/10")
        
        return self.patient_df, self.vitals_scaled

    def draw_projections(self):
        """Step 2: Draw projections from observations"""
        # Generate synthetic historical data for training
        historical_data = self._generate_historical_data()
        
        # Train models
        self.rf_classifier.fit(historical_data['features'], historical_data['conditions'])
        self.gb_classifier.fit(historical_data['features'], historical_data['conditions'])
        
        # Make predictions
        rf_prob = self.rf_classifier.predict_proba(self.vitals_scaled)
        gb_prob = self.gb_classifier.predict_proba(self.vitals_scaled)
        
        # Ensemble predictions
        avg_prob = (rf_prob + gb_prob) / 2
        conditions = self.rf_classifier.classes_
        self.condition_probabilities = dict(zip(conditions, avg_prob[0]))
        
        print("\nStep 2: Clinical Projections")
        print("=" * 50)
        print("Condition Probability Analysis:")
        for condition, prob in self.condition_probabilities.items():
            print(f"{condition}: {prob*100:.2f}%")
        print(f"\nTrend Analysis: {'Deteriorating' if self.risk_score > 7 else 'Stable' if self.risk_score > 4 else 'Improving'}")
        
        return self.condition_probabilities

    def define_condition(self):
        """Step 3: Define condition and stage"""
        primary_condition = max(self.condition_probabilities, key=self.condition_probabilities.get)
        probability = self.condition_probabilities[primary_condition]
        
        # Determine stage based on risk score and vitals
        stage = self._determine_stage(primary_condition)
        
        self.diagnosis = {
            'primary_condition': primary_condition,
            'stage': stage,
            'confidence': probability,
            'supporting_findings': self._extract_supporting_findings()
        }
        
        print("\nStep 3: Condition Definition")
        print("=" * 50)
        print(f"Primary Diagnosis: {primary_condition}")
        print(f"Stage: {stage}")
        print(f"Diagnostic Confidence: {probability*100:.2f}%")
        print("\nSupporting Findings:")
        for finding in self.diagnosis['supporting_findings']:
            print(f"- {finding}")
            
        return self.diagnosis

    def assess_impact(self):
        """Step 4: Assess effect and impact"""
        condition = self.diagnosis['primary_condition']
        stage = self.diagnosis['stage']
        
        impacts = self._generate_impact_assessment(condition, stage)
        
        print("\nStep 4: Impact Assessment")
        print("=" * 50)
        for category, details in impacts.items():
            print(f"\n{category}:")
            for detail in details:
                print(f"- {detail}")
                
        return impacts

    def compare_conditions(self):
        """Step 5: Compare with prior reports and similar cases"""
        # Simulate prior reports and similar cases
        prior_reports = self._get_prior_reports()
        similar_cases = self._find_similar_cases()
        
        print("\nStep 5: Comparative Analysis")
        print("=" * 50)
        print("Comparison with Prior Reports:")
        for date, details in prior_reports.items():
            print(f"\n{date}:")
            for key, value in details.items():
                print(f"- {key}: {value}")
                
        print("\nSimilar Cases Analysis:")
        for case in similar_cases:
            print(f"\nCase ID: {case['id']}")
            print(f"Similarity Score: {case['similarity']:.2f}%")
            print(f"Outcome: {case['outcome']}")

    def visualize_findings(self):
        """Step 7: Visualize findings"""
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # 1. Vital Signs Trend
        plt.subplot(2, 2, 1)
        vitals = np.array(self.patient_df['Vitals'].tolist())
        plt.plot(vitals[0], marker='o')
        plt.title('Vital Signs')
        plt.xlabel('Measurements')
        plt.ylabel('Values')
        
        # 2. Risk Score Gauge
        plt.subplot(2, 2, 2)
        self._plot_risk_gauge(self.risk_score)
        
        # 3. Condition Probabilities
        plt.subplot(2, 2, 3)
        conditions = list(self.condition_probabilities.keys())
        probs = list(self.condition_probabilities.values())
        plt.bar(conditions, probs)
        plt.title('Condition Probabilities')
        plt.xticks(rotation=45)
        
        # 4. Similar Cases Comparison
        plt.subplot(2, 2, 4)
        similar_cases = self._find_similar_cases()
        similarities = [case['similarity'] for case in similar_cases]
        case_ids = [case['id'] for case in similar_cases]
        plt.bar(case_ids, similarities)
        plt.title('Similar Cases Comparison')
        
        plt.tight_layout()
        plt.show()

    def summarize_focus_areas(self):
        """Step 8: Summarize focus areas"""
        condition = self.diagnosis['primary_condition']
        stage = self.diagnosis['stage']
        
        focus_areas = self._generate_focus_areas(condition, stage)
        
        print("\nStep 8: Focus Areas and Recommendations")
        print("=" * 50)
        for category, recommendations in focus_areas.items():
            print(f"\n{category}:")
            for rec in recommendations:
                print(f"- {rec}")
                
        return focus_areas

    # Helper methods
    def _calculate_risk_score(self):
        """Calculate patient risk score"""
        # Implement risk score calculation based on various factors
        return np.random.uniform(1, 10)

    def _generate_historical_data(self):
        """Generate synthetic historical data for model training"""
        n_samples = 1000
        features = np.random.normal(size=(n_samples, self.vitals_scaled.shape[1]))
        conditions = np.random.choice(
            ['Heart Failure', 'Arrhythmia', 'Coronary Artery Disease'],
            size=n_samples
        )
        return {'features': features, 'conditions': conditions}

    def _determine_stage(self, condition):
        """Determine condition stage based on various factors"""
        stages = self.condition_stages.get(condition, ['Unknown'])
        return np.random.choice(stages)

    def _extract_supporting_findings(self):
        """Extract relevant findings supporting the diagnosis"""
        return [
            f"Symptoms: {self.patient_df['Symptoms'].iloc[0]}",
            f"ECG Findings: {self.patient_df['ECG'].iloc[0]}",
            f"Blood Work: {self.patient_df['Blood_Work'].iloc[0]}",
            f"Imaging Results: {self.patient_df['Imaging'].iloc[0]}"
        ]

    def _plot_risk_gauge(self, risk_score):
        """Plot a gauge chart for risk score"""
        plt.pie([risk_score, 10-risk_score], colors=['red', 'lightgray'])
        plt.title(f'Risk Score: {risk_score:.1f}/10')

# Usage Example
if __name__ == "__main__":
    agent = HealthcareAgent()
    
    # Run the diagnostic process
    patient_df, vitals = agent.analyze_patient_data()
    probabilities = agent.draw_projections()
    diagnosis = agent.define_condition()
    impacts = agent.assess_impact()
    agent.compare_conditions()
    agent.visualize_findings()
    recommendations = agent.summarize_focus_areas() 