import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tkinter import Tk, Button, Label, filedialog
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Carica il modello per l'estrazione delle features
feature_extraction_model = load_model("feature_extraction_model_new.h5")

# Carica il modello SVM addestrato
svm_model = joblib.load("SVM_model_mi_threshold.pkl")

# Carica le caratteristiche selezionate da .npy
selected_features = np.load('top_feature_indices.npy')

coef = svm_model.coef_[0]
importance = np.abs(coef)

plt.bar(range(len(importance)), importance)
plt.xlabel('Indice della Caratteristica')
plt.ylabel('Importanza (Valore Assoluto dei Coefficienti)')
plt.title('Importanza delle Caratteristiche in un Modello SVM Lineare')
plt.show()

def classify_image(image_path):
    # Carica e preprocessa l'immagine
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Estrai le features dell'immagine utilizzando il modello per l'estrazione delle features
    features = feature_extraction_model.predict(img_array)

    # Seleziona solo le caratteristiche desiderate
    features_selected = features[:, selected_features]

    # Classifica le features selezionate utilizzando il modello SVM
    prediction = svm_model.predict(features_selected)

    return prediction[0]

def update_ui_with_image(image_path):
    # Aggiorna l'interfaccia utente con l'immagine e la classificazione
    img = Image.open(image_path)
    img = img.resize((250, 250), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    prediction = classify_image(image_path)

    result_label.config(text=f"Risultato della classificazione: {prediction}")

def select_and_classify_image():
    # Funzione per selezionare l'immagine e classificarla
    file_path = filedialog.askopenfilename(title="Seleziona un'immagine",
                                           filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if file_path:
        update_ui_with_image(file_path)

# Setup della finestra principale
root = Tk()
root.title("Classificatore di immagini")
root.geometry("300x300")
root.resizable(False, False)

# Elementi UI
select_button = Button(root, text="Seleziona Immagine", command=select_and_classify_image)
select_button.pack()

image_label = Label(root)  # Inizializzazione dell'etichetta per l'immagine
image_label.pack()

result_label = Label(root, text="Risultato della classificazione:")
result_label.pack()

# Immagine di default
default_image_path = 'C:/Users/casac/Desktop/mushrooms/interfaccia.jpg'  # Aggiorna con il percorso dell'immagine di default
update_ui_with_image(default_image_path)  # Assicurati che questa chiamata sia dopo la definizione di image_label

root.mainloop()
