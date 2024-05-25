import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

# Carica i dati di training e validation
train_data = pd.read_csv('train_features_labels.csv')
validation_data = pd.read_csv('validation_features_labels.csv')

# Prepara i dati di training
X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values

# Prepara i dati di validation
X_test = validation_data.drop('label', axis=1).values
y_test = validation_data['label'].values

# Codifica le label in valori numerici
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Standardizza le caratteristiche
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Applica la PCA per analizzare la varianza spiegata
pca = PCA()
pca.fit(X_train_scaled)

# Calcola la varianza spiegata cumulativa
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Visualizza il grafico della varianza spiegata cumulativa
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Number of Principal Components')
plt.axhline(y=0.95, color='r', linestyle='--')  # Linea orizzontale al 95% della varianza spiegata
plt.show()

# Trova il numero di componenti che spiegano almeno il 95% della varianza
optimal_n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f'Optimal number of components: {optimal_n_components}')

# Applica la PCA con il numero ottimale di componenti
pca = PCA(n_components=optimal_n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Inizializza e addestra il modello SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_pca, y_train_encoded)

# Fai previsioni sul test set
y_pred = svm_model.predict(X_test_pca)

# Valuta il modello
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test_encoded, y_pred, target_names=[str(cls) for cls in label_encoder.classes_]))

# Salva il modello SVM e il modello PCA
joblib.dump(svm_model, "SVM_model_pca_elbow.pkl")
joblib.dump(pca, "pca_model_elbow.pkl")
