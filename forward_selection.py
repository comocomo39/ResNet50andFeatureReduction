import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

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

# Inizializza il modello SVM
svm_model = SVC(kernel='linear')

# Applica la forward selection
sfs = SFS(svm_model,
          k_features=1000,
          forward=True,
          floating=False,
          scoring='accuracy',
          cv=5)

sfs = sfs.fit(X_train, y_train_encoded)

# Indici delle caratteristiche selezionate
selected_features = sfs.k_feature_idx_

# Filtra le caratteristiche selezionate
X_train_sfs = X_train[:, selected_features]
X_test_sfs = X_test[:, selected_features]

# Riaddestra il modello SVM con le caratteristiche selezionate
svm_model.fit(X_train_sfs, y_train_encoded)

# Fai previsioni sul test set
y_pred = svm_model.predict(X_test_sfs)

# Valuta il modello
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test_encoded, y_pred, target_names=[str(cls) for cls in label_encoder.classes_]))

# Salva il nuovo modello SVM
joblib.dump(svm_model, "SVM_model_forward_selection.pkl")
