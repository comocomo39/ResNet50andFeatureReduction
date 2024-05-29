import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load training and validation data
train_data = pd.read_csv('train_features_labels_new.csv')
validation_data = pd.read_csv('validation_features_labels_new.csv')

# Prepare training data
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']

# Prepare validation data
X_test = validation_data.drop('label', axis=1)
y_test = validation_data['label']

# Encode labels into numeric values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate mutual information between features and target
mi_scores = mutual_info_classif(X_train_scaled, y_train_encoded)

# Create a DataFrame to visualize MI scores
mi_df = pd.DataFrame({'Feature': range(X_train_scaled.shape[1]), 'MI': mi_scores})
mi_df = mi_df.sort_values(by='MI', ascending=False)
print(mi_df)

# Define a threshold based on the 75th percentile of the MI scores
mi_threshold = np.percentile(mi_scores, 75)
print(f'Mutual Information threshold (75th percentile): {mi_threshold:.4f}')

# Select features that exceed the MI threshold
selected_features = mi_df[mi_df['MI'] > mi_threshold]['Feature'].values
X_train_selected = X_train_scaled[:, selected_features]
X_test_selected = X_test_scaled[:, selected_features]

# Print selected features
print(f'Selected feature indices: {selected_features}')
print(f'Selected feature mutual information scores: {mi_df[mi_df["MI"] > mi_threshold]["MI"].values}')
print(f'Selected feature names: {X_train.columns[selected_features]}')

print(f'Number of selected features: {len(selected_features)}')

# Initialize and train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_selected, y_train_encoded)

# Make predictions on the test set
y_pred = svm_model.predict(X_test_selected)

# Evaluate the model
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test_encoded, y_pred, target_names=[str(cls) for cls in label_encoder.classes_]))

# Save the SVM model
joblib.dump(svm_model, "SVM_model_mi_threshold.pkl")
np.save("top_feature_indices.npy", selected_features)
