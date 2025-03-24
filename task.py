import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor  # Replace LSTM with MLPRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Step 1: Analysis of Patient Condition and Report
def analyze_patient_data():
    # Simulated patient data (structured and unstructured)
    patient_data = {
        'Age': [55],
        'Symptoms': ['Fatigue, Shortness of Breath, Irregular Heart Rate'],
        'ECG': ['Left Ventricular Dysfunction'],
        'Blood_Work': ['Anemia'],
        'Imaging': ['Mild Cardiomegaly'],
        'History': ['Hypertension, Family History of Heart Disease'],
        'Vitals': [[120, 80, 90]],  # [Systolic BP, Diastolic BP, Heart Rate]
    }
    patient_df = pd.DataFrame(patient_data)
    
    # Numerical features for PCA
    numerical_features = np.array(patient_df['Vitals'].tolist())
    
    # Standardize the features
    scaler = StandardScaler()
    numerical_features_scaled = scaler.fit_transform(numerical_features)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=1)
    reduced_features = pca.fit_transform(numerical_features_scaled)
    
    print("Step 1: Patient Data Analysis")
    print("Summarized Patient Profile:")
    print(patient_df[['Age', 'Symptoms', 'ECG', 'Blood_Work', 'Imaging', 'History']])
    print("PCA Reduced Features:", reduced_features)
    return patient_df, reduced_features

# Step 2: Draw Projections from the Observation
def draw_projections(patient_df, reduced_features):
    # Simulated historical data
    historical_data = np.random.uniform(0.08395, 0.08425, (50, 3))  # Simulated vitals
    historical_labels = np.array([0, 1, 0, 1, 0] * 10)  # Simulated binary labels

    # Build MLPRegressor model instead of LSTM
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(50,),
        activation='relu',
        solver='adam',
        max_iter=500
    )
    mlp_model.fit(historical_data, historical_labels)

    # Predict using MLPRegressor
    current_vitals = np.array(patient_df['Vitals'].tolist())
    mlp_prediction = mlp_model.predict(current_vitals)

    # Random Forest for condition classification
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Use the original vitals instead of reduced features for Random Forest
    X_train = np.random.rand(100, 3)  # Match the number of features in vitals
    y_train = np.random.choice(['Heart Failure', 'Arrhythmia', 'Coronary Artery Disease'], 100)
    rf_model.fit(X_train, y_train)
    
    # Predict condition using the original vitals
    rf_prediction = rf_model.predict_proba(current_vitals)
    conditions = ['Heart Failure', 'Arrhythmia', 'Coronary Artery Disease']
    probabilities = dict(zip(conditions, rf_prediction[0]))

    print("\nStep 2: Projections from Observations")
    print("MLP Prediction (Abnormality Likelihood):", mlp_prediction[0])
    print("Condition Probabilities:", probabilities)
    return probabilities

# Step 3: Condition Definition by Listing Findings with Condition and Stage Specification
def define_condition(patient_df, probabilities):
    # Identify most likely condition
    condition = max(probabilities, key=probabilities.get)
    
    # Simulated data for Euclidean Distance comparison
    known_cases = np.random.rand(10, 3)  # Simulated features for known heart failure cases
    current_case = np.array(patient_df['Vitals'].tolist())
    
    # Calculate Euclidean Distance
    distances = pairwise_distances(current_case, known_cases, metric='euclidean')
    closest_case_idx = np.argmin(distances)
    
    # Simulated staging based on distance
    stage = 'Stage C' if distances[0][closest_case_idx] < 10 else 'Stage B'
    
    print("\nStep 3: Condition Definition")
    print(f"Condition: {condition}, {stage}")
    print("Findings:")
    print(f"- Symptoms: {patient_df['Symptoms'][0]}")
    print(f"- ECG: {patient_df['ECG'][0]}")
    print(f"- Blood Work: {patient_df['Blood_Work'][0]}")
    print(f"- Imaging: {patient_df['Imaging'][0]}")
    print(f"- Contributing Factors: {patient_df['History'][0]}")
    return condition, stage

# Step 4: Effect and Impact Created by the Condition Defined
def assess_impact(condition, stage):
    print("\nStep 4: Effect and Impact of the Condition")
    print(f"Condition: {condition}, {stage}")
    print("Impacts:")
    print("- Physical: Reduced cardiac output leading to fatigue and shortness of breath.")
    print("- Systemic: Anemia exacerbates symptoms by reducing oxygen delivery.")
    print("- Risks: Increased risk of hospitalization, arrhythmias, or progression to Stage D.")

# Step 5: Comparison with Prior Test Report and Similar Conditions
def compare_conditions(patient_df):
    # Simulated prior test report and other patients' data
    prior_report = {
        'ECG': 'Normal',
        'Blood_Work': 'Normal',
        'Imaging': 'Normal'
    }
    other_patients = pd.DataFrame({
        'ECG': ['Left Ventricular Dysfunction', 'Normal', 'Right Ventricular Dysfunction'],
        'Blood_Work': ['Anemia', 'Normal', 'Normal'],
        'Imaging': ['Mild Cardiomegaly', 'Normal', 'Severe Cardiomegaly']
    })
    
    print("\nStep 5: Comparison with Prior Report and Other Patients")
    print("Prior Report Comparison:")
    print(f"Current ECG: {patient_df['ECG'][0]} vs Prior: {prior_report['ECG']}")
    print(f"Current Blood Work: {patient_df['Blood_Work'][0]} vs Prior: {prior_report['Blood_Work']}")
    print(f"Current Imaging: {patient_df['Imaging'][0]} vs Prior: {prior_report['Imaging']}")
    print("\nComparison with Other Patients:")
    print(other_patients)

# Step 6: Defining the Observation and Effect
def define_observation_effect(condition, stage):
    print("\nStep 6: Observation and Effect")
    print(f"Observation: The patient exhibits symptoms and test results consistent with {condition}, {stage}.")
    print("Effect: The condition limits physical activity, increases cardiac workload, and poses risks of further complications.")

# Step 7: Visualizing the Findings
def visualize_findings(patient_df, other_patients):
    # Combine patient data with other patients for visualization
    combined_data = other_patients.copy()
    combined_data.loc[len(combined_data)] = patient_df[['ECG', 'Blood_Work', 'Imaging']].iloc[0]
    
    # Convert categorical data to numerical for clustering
    combined_data_encoded = pd.get_dummies(combined_data)
    
    # Hierarchical clustering
    Z = linkage(combined_data_encoded, method='ward')
    
    # Plot dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(Z, labels=['Patient 1', 'Patient 2', 'Patient 3', 'Current Patient'])
    plt.title("Hierarchical Clustering of Patient Data")
    plt.xlabel("Patients")
    plt.ylabel("Distance")
    plt.show()
    
    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(combined_data_encoded, annot=True, cmap='coolwarm', cbar=True)
    plt.title("Heatmap of Patient Findings")
    plt.show()

# Step 8: Summarizing Areas or Effects to Concentrate On
def summarize_focus_areas(condition, stage):
    print("\nStep 8: Summary of Areas to Concentrate On")
    print(f"1. Manage {condition} ({stage}): Initiate treatment to improve cardiac function.")
    print("2. Address Anemia: Treat underlying causes to reduce cardiac workload.")
    print("3. Monitor for Complications: Regular follow-ups to prevent progression to Stage D.")
    print("4. Lifestyle Modifications: Recommend diet and exercise adjustments.")

# Main function to run the diagnostic process
def run_diagnostic_process():
    print("HealthInsight AI: Diagnostic Process\n")
    
    # Step 1
    patient_df, reduced_features = analyze_patient_data()
    
    # Step 2
    probabilities = draw_projections(patient_df, reduced_features)
    
    # Step 3
    condition, stage = define_condition(patient_df, probabilities)
    
    # Step 4
    assess_impact(condition, stage)
    
    # Step 5
    compare_conditions(patient_df)
    
    # Step 6
    define_observation_effect(condition, stage)
    
    # Step 7
    other_patients = pd.DataFrame({
        'ECG': ['Left Ventricular Dysfunction', 'Normal', 'Right Ventricular Dysfunction'],
        'Blood_Work': ['Anemia', 'Normal', 'Normal'],
        'Imaging': ['Mild Cardiomegaly', 'Normal', 'Severe Cardiomegaly']
    })
    visualize_findings(patient_df, other_patients)
    
    # Step 8
    summarize_focus_areas(condition, stage)

# Run the diagnostic process
if __name__ == "__main__":
    run_diagnostic_process()