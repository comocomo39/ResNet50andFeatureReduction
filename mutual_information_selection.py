import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Carica i dati di training e validation
train_data = pd.read_csv('train_features_labels_new.csv')
validation_data = pd.read_csv('validation_features_labels_new.csv')

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

# Calcola la mutual information tra le caratteristiche e il target
mi_scores = mutual_info_classif(X_train_scaled, y_train_encoded)

# Crea un DataFrame per visualizzare i punteggi di MI
mi_df = pd.DataFrame({'Feature': range(X_train_scaled.shape[1]), 'MI': mi_scores})
mi_df = mi_df.sort_values(by='MI', ascending=False)
print(mi_df)

# Seleziona le top N caratteristiche in base alla MI
N = 50  # Numero di caratteristiche da selezionare
top_features = mi_df['Feature'].head(N).values
X_train_selected = X_train_scaled[:, top_features]
X_test_selected = X_test_scaled[:, top_features]

# Inizializza e addestra il modello SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_selected, y_train_encoded)

# Fai previsioni sul test set
y_pred = svm_model.predict(X_test_selected)

# Valuta il modello
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test_encoded, y_pred, target_names=[str(cls) for cls in label_encoder.classes_]))

# Salva il modello SVM
joblib.dump(svm_model, "SVM_model_mi.pkl")
