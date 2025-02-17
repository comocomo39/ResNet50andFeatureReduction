Ecco il README pronto per il copia e incolla:

```markdown
# 🍄 Classificazione di Funghi con ResNet50 e Riduzione delle Feature

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Deep Learning](https://img.shields.io/badge/Deep%20Learning-ResNet50-brightgreen) ![Feature Selection](https://img.shields.io/badge/Feature%20Selection-PCA%2C%20Mutual%20Information-orange)

## 📖 Descrizione

Questo progetto si focalizza sulla **classificazione di immagini di funghi** utilizzando l'architettura **ResNet50** combinata con tecniche di **riduzione e selezione delle feature**. L'obiettivo principale è migliorare l'accuratezza della classificazione riducendo la dimensionalità dei dati e selezionando le feature più rilevanti.

---

## 🎯 Obiettivi

- **Implementare l'architettura ResNet50** per l'estrazione di feature dalle immagini di funghi.
- **Applicare tecniche di riduzione delle feature**, come la **PCA (Principal Component Analysis)**, per diminuire la dimensionalità dei dati.
- **Utilizzare metodi di selezione delle feature**, tra cui la **Mutual Information**, per identificare le feature più significative.
- **Confrontare le performance** dei modelli con diverse combinazioni di feature selezionate e ridotte.

---

## 🛠️ Tecnologie Utilizzate

Il progetto è stato sviluppato utilizzando **Python** e include i seguenti script chiave:

- `classification.py`: script per la classificazione delle immagini utilizzando il modello ResNet50.
- `preprocessing.py`: script per il preprocessing delle immagini e l'estrazione delle feature iniziali.
- `pca_selection.py`: implementazione della riduzione delle feature tramite PCA.
- `mutual_information_selection.py`: selezione delle feature basata sulla Mutual Information.
- `forward_selection.py` e `backward_selection.py`: metodi di selezione sequenziale delle feature.

---

## 📂 Struttura della Repository

- `main.py`: script principale per l'esecuzione del workflow completo di preprocessing, selezione/riduzione delle feature e classificazione.
- `test.py`: script per il testing e la valutazione del modello addestrato.
- `DocumentationDMML.pdf`: documentazione dettagliata del progetto.
- `Mushroom_Classification_Project.pptx`: presentazione del progetto con risultati e analisi.

---

## 🚀 Come Iniziare

1. **Clona la repository:**

   ```bash
   git clone https://github.com/comocomo39/ResNet50andFeatureReduction.git
   cd ResNet50andFeatureReduction
   ```

2. **Installa le dipendenze richieste:**

   Assicurati di avere installato `torch`, `torchvision`, `numpy`, `scikit-learn` e altre librerie necessarie.

   ```bash
   pip install -r requirements.txt
   ```

3. **Esegui lo script principale:**

   ```bash
   python main.py
   ```

   Questo script eseguirà l'intero processo di preprocessing, selezione/riduzione delle feature e classificazione.

---

## 📜 Documentazione

Per una comprensione dettagliata del progetto, inclusi i risultati ottenuti e le metodologie utilizzate, consulta il documento `DocumentationDMML.pdf` e la presentazione `Mushroom_Classification_Project.pptx` presenti nella repository.

---

## 👥 Collaboratori

- **comocomo39**: [GitHub Profile](https://github.com/comocomo39)
- **GiacomoMoro**: [GitHub Profile](https://github.com/GiacomoMoro)

---
