import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
y_test_encoded = label_encoder.transform(y_test)  # Usa lo stesso encoder per mantenere la consistenza

# Inizializza i modelli
models = {
    "SVM": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
    "AdaBoost": AdaBoostClassifier(n_estimators=100)
}

# Nel loop di previsione dei modelli
for name, model in models.items():
    model.fit(X_train, y_train_encoded)
    y_pred = model.predict(X_test)

    # Inverti la codifica numerica per ottenere i nomi originali delle classi
    y_pred_original = label_encoder.inverse_transform(y_pred)

    print(f"Performance of {name}:")
    print(f"Accuracy: {accuracy_score(y_test_encoded, y_pred)}")
    print(classification_report(y_test_encoded, y_pred, target_names=[str(cls) for cls in label_encoder.classes_]))
    print("\n")
    joblib.dump(model, f"{name}_model.pkl")



