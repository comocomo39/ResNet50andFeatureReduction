from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.image import resize
import numpy as np
import pandas as pd
import tensorflow as tf

data_dir = 'C:/Users/casac/Desktop/funghi/'

# Dimensioni desiderate delle immagini
image_size = (224, 224)

# Funzione per il preprocessing delle immagini
def preprocess_images(image):
    image_resized = resize(image, image_size)
    return preprocess_input(image_resized)

# Data generator con preprocessing_function per preprocessare le immagini
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_images,
    validation_split=0.2)

from PIL import Image
import os

# Percorso alla directory contenente le immagini
data_dir = 'C:/Users/casac/Desktop/funghi/'

# Lista per tenere traccia dei file problematici
invalid_files = []

# Esamina tutti i file nelle sottodirectory di data_dir
for subdir, dirs, files in os.walk(data_dir):
    for file in files:
        file_path = os.path.join(subdir, file)
        try:
            # Tenta di aprire l'immagine
            with Image.open(file_path) as img:
                # Opzionalmente, puoi tentare di fare un'operazione sull'immagine per assicurarti che non sia corrotta
                img.verify()
        except (IOError, SyntaxError) as e:
            print(f"File non valido: {file_path}")
            invalid_files.append(file_path)

# Stampa il numero di file non validi trovati
print(f"Trovati {len(invalid_files)} file non validi.")

# Opzionalmente, stampa i percorsi dei file non validi
for path in invalid_files:
    print(path)

# Qui puoi decidere se rimuovere i file non validi dal filesystem o gestirli in altro modo


batch_size = 32  # Puoi aggiustare questo in base alla memoria disponibile

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# Carica ResNet50 con i pesi pre-addestrati su ImageNet, senza l'ultimo layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congela i layers del modello base
for layer in base_model.layers:
    layer.trainable = False

# Aggiunge i layer personalizzati
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compila il modello
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Allena il modello
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Crea un modello per l'estrazione delle features dal penultimo layer
penultimate_layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)

# Funzione per estrarre le features usando il modello appena creato
# Funzione per estrarre le features e le label (con nomi di classe) usando il modello
def extract_features_and_labels_with_class_names(generator, sample_count):
    features = np.zeros(shape=(sample_count, 2048))  # Dimensione in base all'output del penultimo layer
    labels = np.empty(shape=(sample_count), dtype=object)  # Modifica qui per usare dtype object
    i = 0
    class_indices = {v: k for k, v in generator.class_indices.items()}  # Inverte il dizionario class_indices
    for inputs_batch, labels_batch in generator:
        features_batch = penultimate_layer_model.predict(inputs_batch)
        batch_size = inputs_batch.shape[0]
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels_batch_indices = np.argmax(labels_batch, axis=1)
        labels[i * batch_size: (i + 1) * batch_size] = [class_indices[idx] for idx in labels_batch_indices]  # Mappa gli indici alle classi
        i += 1
        if (i + 1) * batch_size >= sample_count:
            break
    return features, labels

# Aggiornamento del numero di campioni per corrispondere al dataset
num_train_samples = 70912  # Numero esatto di immagini di training
num_validation_samples = 17726  # Numero esatto di immagini di validazione

# Estrazione delle features e delle labels aggiornate
train_features, train_labels = extract_features_and_labels_with_class_names(train_generator, num_train_samples)
validation_features, validation_labels = extract_features_and_labels_with_class_names(validation_generator, num_validation_samples)

# Funzione per salvare le features e le labels in un file CSV
def save_features_labels_to_csv(features, labels, file_name):
    features_df = pd.DataFrame(features)
    labels_df = pd.DataFrame(labels, columns=['label'])
    data_df = pd.concat([features_df, labels_df], axis=1)
    data_df.to_csv(file_name, index=False)

# Salvataggio dei dati estratti in file CSV
save_features_labels_to_csv(train_features, train_labels, 'train_features_labels.csv')
save_features_labels_to_csv(validation_features, validation_labels, 'validation_features_labels.csv')
