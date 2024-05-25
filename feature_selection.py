import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import pandas as pd

# Carica il modello SVM addestrato
svm_model = joblib.load("SVM_model.pkl")

# Estrai i coefficienti delle caratteristiche
coef = svm_model.coef_[0]

# Calcola l'importanza delle caratteristiche (valore assoluto dei coefficienti)
importance = np.abs(coef)

# Ordina le caratteristiche per importanza decrescente
sorted_indices = np.argsort(importance)[::-1]

# Seleziona le prime 1000 caratteristiche
top_1000_indices = sorted_indices[:1000]

# Salva gli indici delle prime 1000 caratteristiche
np.save("top_1000_feature_indices.npy", top_1000_indices)

# Visualizza l'importanza delle prime 1000 caratteristiche
plt.bar(range(1000), importance[top_1000_indices])
plt.xlabel('Indice della Caratteristica')
plt.ylabel('Importanza (Valore Assoluto dei Coefficienti)')
plt.title('Importanza delle Prime 1000 Caratteristiche in un Modello SVM Lineare')
plt.show()

# Carica i dati di training e validation
train_data = pd.read_csv('train_features_labels.csv')
validation_data = pd.read_csv('validation_features_labels.csv')

# Prepara i dati di training
X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values

# Prepara i dati di validation
X_test = validation_data.drop('label', axis=1).values
y_test = validation_data['label'].values

# Filtra le prime 1000 caratteristiche per i dati di training e validation
X_train_top_1000 = X_train[:, top_1000_indices]
X_test_top_1000 = X_test[:, top_1000_indices]

# Codifica le label in valori numerici
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)  # Usa lo stesso encoder per mantenere la consistenza

# Riaddestra un nuovo classificatore SVM con le prime 1000 caratteristiche
new_svm_model = SVC(kernel='linear')
new_svm_model.fit(X_train_top_1000, y_train_encoded)

# Fai previsioni sul test set
y_pred = new_svm_model.predict(X_test_top_1000)

# Valuta il modello
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test_encoded, y_pred, target_names=[str(cls) for cls in label_encoder.classes_]))

# Salva il nuovo modello SVM
joblib.dump(new_svm_model, "new_SVM_model_top_1000.pkl")
