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
