# mrxs_to_omezarr.zip[README.md](https://github.com/user-attachments/files/24416338/README.md)
# OME-Zarr Pathology Toolkit

Suite d'outils pour la pathologie numérique : conversion, visualisation et annotation de lames virtuelles au format OME-Zarr.

**Projet open source — Pathologie numérique**

---

## 📋 Sommaire

- [Fonctionnalités](#-fonctionnalités)
- [Installation](#-installation)
- [Outils disponibles](#-outils-disponibles)
- [Utilisation](#-utilisation)
- [Presets de conversion](#-presets-de-conversion)
- [Format OME-Zarr](#-format-ome-zarr)
- [Dépannage](#-dépannage)

---

## ✨ Fonctionnalités

- **Conversion MRXS → OME-Zarr** : Conversion par lots avec file d'attente, estimation du temps, compression JPEG/JPEG-XL/Blosc
- **Visualisation pyramidale** : Navigation fluide multi-niveaux avec cache de tuiles
- **Multi-lames** : Ouverture simultanée de plusieurs lames en onglets
- **Annotations hiérarchiques** : Système d'annotation multi-niveaux avec classes personnalisables
- **Export GeoJSON** : Annotations compatibles avec les standards géospatiaux
- **Compression ZIP** : Archivage optimisé pour le transfert

---

## 🔧 Installation

### Prérequis système

**Ubuntu/Debian :**
```bash
sudo apt update
sudo apt install openslide-tools libopenslide-dev python3-tk
```

**macOS :**
```bash
brew install openslide
# tkinter est inclus avec Python de Homebrew
```

**Windows :**
1. Télécharger OpenSlide depuis [openslide.org/download](https://openslide.org/download/)
2. Ajouter le dossier `bin` au PATH système

### Installation Python

```bash
# Cloner ou télécharger le projet
cd omezarr-pathology-toolkit

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Pour la compression JPEG-XL optimale (optionnel)
pip install imagecodecs[all]
```

---

## 🛠 Outils disponibles

| Outil | Description | Fichier |
|-------|-------------|---------|
| **Convertisseur** | Conversion MRXS → OME-Zarr avec file d'attente | `mrxszarr6.py` |
| **Viewer Multi** | Visualisation multi-lames avec onglets | `omezarr_viewer_multi.py` |
| **Annotateur** | Viewer avec annotations hiérarchiques | `omezarr_annotator2.py` |

---

## 🚀 Utilisation

### Convertisseur MRXS → OME-Zarr

```bash
python mrxszarr6.py
```

**Interface :**
1. **Ajouter des fichiers** : Glisser-déposer ou bouton "Ajouter"
2. **Choisir un preset** : Défaut, Haute qualité, Archivage, Web rapide
3. **Lancer la conversion** : Les fichiers sont traités séquentiellement

**Fonctionnalités clés :**
- File d'attente avec statut par fichier
- Estimation du temps de conversion
- Validation automatique des fichiers MRXS
- Vignettes de prévisualisation
- Compression ZIP optionnelle après conversion

### Viewer Multi-lames

```bash
python omezarr_viewer_multi.py
```

**Navigation :**
- **Molette** : Zoom avant/arrière
- **Clic gauche + glisser** : Déplacer la vue
- **Double-clic** sur l'arborescence : Ouvrir une lame

**Fonctionnalités :**
- Arborescence des fichiers .zarr
- Onglets pour plusieurs lames simultanées
- Cache de tuiles partagé (200 tuiles)

### Annotateur

```bash
python omezarr_annotator2.py
```

**Modes :**
- **Navigation** : Parcourir la lame
- **Dessin (D)** : Créer des annotations polygonales

**Annotations :**
- Niveaux hiérarchiques (Macroscopique, Tissulaire, Cellulaire)
- Classes personnalisables avec couleurs
- Raccourcis clavier 1-9 pour les classes
- Sauvegarde intégrée dans le Zarr ou export GeoJSON

---

## 📦 Presets de conversion

| Preset | Compression | Qualité | Downscale | Usage |
|--------|-------------|---------|-----------|-------|
| **Défaut** | Blosc/ZSTD | 85 | ×2.0 | Usage quotidien |
| **Haute qualité** | JPEG-XL | 95 | ×2.0 | Archivage haute fidélité |
| **Archivage** | JPEG | 60 | ×6.67 | Transfert/stockage longue durée |
| **Web rapide** | JPEG | 75 | ×2.0 | Affichage web optimisé |

### Preset Archivage

Le preset "Archivage" est optimisé pour réduire drastiquement la taille des fichiers :
- Démarre du niveau natif 3 du MRXS (~×8 de l'original)
- Applique un downscale supplémentaire de ×6.67
- Compression JPEG qualité 60
- Résultat : fichiers ~50× plus petits, idéaux pour le transfert réseau

---

## 📁 Format OME-Zarr

Structure d'un fichier OME-Zarr généré :

```
lame.ome.zarr/
├── .zattrs              # Métadonnées OME-NGFF
├── .zgroup              # Marqueur de groupe Zarr
├── 0/                   # Niveau 0 (pleine résolution)
│   ├── .zarray
│   └── [chunks...]
├── 1/                   # Niveau 1 (×2 downscale)
├── 2/                   # Niveau 2 (×4 downscale)
└── ...
```

**Métadonnées incluses :**
- Version OME-NGFF 0.4
- Axes (Y, X, C) avec unités
- Transformations d'échelle par niveau
- Source et méthode de conversion

---

## 🔍 Dépannage

### "OpenSlide not found"

```bash
# Vérifier l'installation
python -c "import openslide; print(openslide.__version__)"

# Si erreur, réinstaller la librairie système
sudo apt install libopenslide0  # Ubuntu/Debian
```

### "JPEG-XL non disponible"

```bash
# Installer imagecodecs avec tous les codecs
pip install --upgrade imagecodecs[all]

# Vérifier
python -c "from imagecodecs.numcodecs import JpegXl; print('OK')"
```

### Fichier MRXS invalide

Le convertisseur vérifie automatiquement :
- Présence du fichier `.mrxs`
- Présence du dossier de données associé (même nom sans extension)
- Lisibilité par OpenSlide

### Performances lentes

- Utiliser un SSD pour les fichiers source et destination
- Réduire la taille de tuile (256 au lieu de 512)
- Utiliser le preset "Archivage" pour partir d'un niveau natif

---

## 📊 Performances typiques

| Configuration | Vitesse | Fichier 2GB MRXS |
|--------------|---------|------------------|
| SSD + Blosc | ~35 ms/tuile | ~15 min |
| SSD + JPEG | ~40 ms/tuile | ~18 min |
| SSD + JPEG-XL | ~55 ms/tuile | ~25 min |
| Archivage (niveau 3) | ~30 ms/tuile | ~3 min |

---

## 📄 Licence

Projet open source développé sur le temps libre de l'auteur, qui luttait contre l'attrait de son chat pour le clavier. 🐱⌨️

MIT License - Utilisation libre.

---

## 🤝 Contributions

Les contributions sont bienvenues ! Pour signaler un bug ou proposer une amélioration, ouvrir une issue sur le dépôt.
