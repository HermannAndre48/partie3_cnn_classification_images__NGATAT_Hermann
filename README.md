# Classification d'Images : Chats vs Chiens avec CNN

Classification binaire d'images utilisant un réseau de neurones convolutifs (CNN) construit from scratch avec TensorFlow/Keras.

## 📋 Vue d'ensemble

Ce projet implémente un pipeline complet de classification d'images en 4 phases :

### **Phase 1.1 : Création et Préparation du Dataset**
- ✅ Génération de 5000 images synthétiques (80% train, 20% validation)
- ✅ Structure : 4000 images train + 1000 images validation
- ✅ Classes équilibrées : 2000/500 chats + 2000/500 chiens
- ✅ Résolution : 224×224 pixels, format JPEG (qualité 85)

### **Phase 1.2 : Prétraitement et Normalisation**
- ✅ Chargement avec `image_dataset_from_directory`
- ✅ Normalisation : rescaling [0-255] → [0.0-1.0]
- ✅ Batching : taille 32
- ✅ Caching + Prefetch pour performance optimale
- ✅ Validation : shape (32, 128, 128, 3) et (32, 1)

### **Phase 1.3 : Architecture CNN**
```
Sequential Model
├── Conv2D(32, 3×3, relu) → MaxPooling2D
├── Conv2D(64, 3×3, relu) → MaxPooling2D
├── Conv2D(128, 3×3, relu) → MaxPooling2D
├── Flatten
├── Dense(128, relu)
└── Dense(1, sigmoid)  [binary classification]

Total Parameters : 4,287,809
```

### **Phase 1.4 : Entraînement et Diagnostic**
- ✅ 20 epochs avec convergence rapide
- ✅ Callbacks : TensorBoard + EarlyStopping(patience=5)
- ✅ Optimiseur : Adam | Loss : binary_crossentropy
- ✅ **Résultats finaux** :
  - Train Accuracy : **100%**
  - Val Accuracy : **100%**
  - Temps : ~1252 secondes (CPU)
  - Diagnostic : Généralisation acceptable (gap = 0.0)

## 📁 Structure du Projet

```
partie3_cnn_classification_images__NGATAT_Hermann/
├── CNN_Classification_Phase1-3.ipynb  [Notebook principal - 14 cells]
├── curves_cnn_scratch.png            [Courbes d'apprentissage]
├── README.md                          [Documentation]
├── .gitignore                         [Exclusions git]
└── data/                              [Dataset - non versionné]
    ├── train/
    │   ├── chat/ (2000 images)
    │   └── chien/ (2000 images)
    └── val/
        ├── chat/ (500 images)
        └── chien/ (500 images)
```

## 🚀 Utilisation

### Prérequis
```bash
tensorflow>=2.10
pillow>=9.0
matplotlib>=3.5
numpy>=1.21
```

### Exécution du Notebook

1. **Clonez le repo** :
```bash
git clone https://github.com/HermannAndre48/partie3_cnn_classification_images__NGATAT_Hermann.git
cd partie3_cnn_classification_images__NGATAT_Hermann
```

2. **Installez les dépendances** :
```bash
pip install tensorflow pillow matplotlib numpy
```

3. **Exécutez le notebook Jupyter** :
```bash
jupyter notebook CNN_Classification_Phase1-3.ipynb
```

Le notebook exécutera automatiquement les 4 phases dans l'ordre.

## 📊 Résultats et Visualisation

Les courbes d'apprentissage sont sauvegardées dans `curves_cnn_scratch.png` :
- **Loss** : converge de 0.1874 → 8.94e-08
- **Accuracy** : converge de 0.914 → 1.0 en 2 epochs
- **Val Loss** : converge de 6.27e-05 → 1.35e-07

## 🔧 Détails Techniques

| Paramètre | Valeur |
|-----------|--------|
| Framework | TensorFlow 2.x / Keras |
| Model API | Sequential |
| Image Size | 128×128 (après redimensionnement) |
| Batch Size | 32 |
| Epochs | 20 |
| Optimizer | Adam (lr=0.001 par défaut) |
| Loss Function | binary_crossentropy |
| Metrics | accuracy |
| Train/Val Split | 80/20 |

## ✅ Checklist d'Exécution

- [x] Phase 1.1 : Dataset créé et validé
- [x] Phase 1.2 : Pipeline de prétraitement configuré
- [x] Phase 1.3 : Architecture CNN compilée
- [x] Phase 1.4 : Entraînement complété avec diagnostic
- [x] Visualisations générées
- [x] Code poussé vers GitHub

## 📝 Notes

- Le dataset est **synthétique** (généré par PIL) pour éviter les dépendances réseau/Kaggle
- La performance parfaite (100%) est attendue sur données synthétiques avec features claires
- Pour un dataset réel, une data augmentation serait recommandée
- Les fichiers `/data` et `/logs` sont exclus de git via `.gitignore`

## 📧 Auteur

Hermann NGATAT - Juillet 2026
