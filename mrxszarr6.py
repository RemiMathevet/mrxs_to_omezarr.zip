"""
MRXS → OME-Zarr pyramidal (lecture par tuiles, économe en RAM)
GUI Tkinter – compression JPEG ou JPEG-XL selon disponibilité

Dépendances :
pip install openslide-python ome-zarr zarr dask numpy numcodecs pillow
pip install imagecodecs  # optionnel, pour JPEG-XL
"""

import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
from typing import Callable, Optional
import math
import time

import openslide
import numpy as np
import zarr
import dask.array as da

# Détecter la version de Zarr
ZARR_V3 = zarr.__version__.startswith("3")

# Compresseur - on utilisera numcodecs qui fonctionne avec les deux versions
try:
    from numcodecs import Blosc, register_codec
    COMPRESSOR_BLOSC = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    NUMCODECS_AVAILABLE = True
except ImportError:
    COMPRESSOR_BLOSC = None
    NUMCODECS_AVAILABLE = False

# JPEG-XL optionnel
JPEGXL_AVAILABLE = False
if NUMCODECS_AVAILABLE:
    try:
        from imagecodecs.numcodecs import JpegXl
        register_codec(JpegXl)
        JPEGXL_AVAILABLE = True
    except ImportError:
        pass

# JPEG standard optionnel
JPEG_AVAILABLE = False
if NUMCODECS_AVAILABLE:
    try:
        from imagecodecs.numcodecs import Jpeg
        register_codec(Jpeg)
        JPEG_AVAILABLE = True
    except ImportError:
        pass


# Presets de configuration
PRESETS = {
    "Défaut": {
        "description": "Paramètres par défaut, bon compromis qualité/taille",
        "tile_size": 512,
        "downscale": 2.0,
        "max_levels": 0,  # Auto
        "quality": 85,
        "compression": "blosc",  # blosc, jpeg, jpegxl
        "use_native_levels": True,
        "base_level": 0,
    },
    "Haute qualité": {
        "description": "Qualité maximale, fichiers plus volumineux",
        "tile_size": 512,
        "downscale": 2.0,
        "max_levels": 0,
        "quality": 95,
        "compression": "jpegxl",
        "use_native_levels": True,
        "base_level": 0,
    },
    "Archivage": {
        "description": "Fichiers légers pour archivage longue durée (~×6.6 depuis niveau 3)",
        "tile_size": 512,
        "downscale": 6.67,
        "max_levels": 3,
        "quality": 60,
        "compression": "jpeg",
        "use_native_levels": True,
        "base_level": 3,
    },
    "Web rapide": {
        "description": "Optimisé pour affichage web, JPEG standard",
        "tile_size": 256,
        "downscale": 2.0,
        "max_levels": 6,
        "quality": 75,
        "compression": "jpeg",
        "use_native_levels": True,
        "base_level": 0,
    },
}


def format_duration(seconds: float) -> str:
    """Formate une durée en secondes en format lisible (ex: 2h 15min 30s)."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}min {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes:02d}min"


def estimate_conversion_time(
    total_tiles: int,
    tile_size: int,
    compression: str,
    use_native_levels: bool,
    base_level: int
) -> tuple[float, float, float]:
    """
    Estime le temps de conversion basé sur le nombre de tuiles.
    
    Retourne (estimation_basse, estimation_moyenne, estimation_haute) en secondes.
    
    Hypothèses de performance (par tuile 512x512) :
    - Lecture OpenSlide + resize : ~15-30 ms
    - Compression Blosc/ZSTD : ~5-10 ms  
    - Compression JPEG : ~10-20 ms
    - Compression JPEG-XL : ~20-40 ms
    - Écriture Zarr : ~5-10 ms
    
    Pour tuiles plus grandes, le temps augmente ~linéairement avec la surface.
    """
    # Facteur de taille de tuile (référence = 512)
    tile_factor = (tile_size / 512) ** 2
    
    # Temps de base par tuile en ms (pour 512x512)
    if compression == "jpegxl":
        # JPEG-XL est le plus lent
        time_per_tile_low = 35   # ms - cas optimiste (SSD rapide, CPU puissant)
        time_per_tile_mid = 55   # ms - cas moyen
        time_per_tile_high = 80  # ms - cas pessimiste
    elif compression == "jpeg":
        # JPEG est intermédiaire
        time_per_tile_low = 25   # ms
        time_per_tile_mid = 40   # ms
        time_per_tile_high = 60  # ms
    else:
        # Blosc/ZSTD est le plus rapide
        time_per_tile_low = 20   # ms
        time_per_tile_mid = 35   # ms
        time_per_tile_high = 55  # ms
    
    # Bonus si on part d'un niveau natif (moins de données à lire)
    if use_native_levels and base_level > 0:
        native_bonus = 0.85  # 15% plus rapide
    else:
        native_bonus = 1.0
    
    # Calcul des estimations
    est_low = total_tiles * time_per_tile_low * tile_factor * native_bonus / 1000
    est_mid = total_tiles * time_per_tile_mid * tile_factor * native_bonus / 1000
    est_high = total_tiles * time_per_tile_high * tile_factor * native_bonus / 1000
    
    return est_low, est_mid, est_high


# -----------------------------
# Classe d'accès lazy aux tuiles OpenSlide
# -----------------------------
class OpenSlideLazyTiles:
    """
    Permet à Dask de lire les tuiles à la demande sans charger toute l'image.
    """
    def __init__(self, slide: openslide.OpenSlide, level: int = 0):
        self.slide = slide
        self.level = level
        self.width, self.height = slide.level_dimensions[level]
        self.shape = (self.height, self.width, 3)
        self.dtype = np.uint8
        self.ndim = 3

    def __getitem__(self, slices):
        y_slice, x_slice, c_slice = slices

        # Calculer les coordonnées
        y_start = y_slice.start or 0
        x_start = x_slice.start or 0
        y_stop = y_slice.stop or self.height
        x_stop = x_slice.stop or self.width

        h = y_stop - y_start
        w = x_stop - x_start

        # Convertir en coordonnées niveau 0 si nécessaire
        downsample = self.slide.level_downsamples[self.level]
        x0 = int(x_start * downsample)
        y0 = int(y_start * downsample)

        # Lire la région
        region = self.slide.read_region((x0, y0), self.level, (w, h))
        arr = np.asarray(region.convert("RGB"))

        # Gérer le slice sur les canaux
        if isinstance(c_slice, slice):
            return arr[:, :, c_slice]
        return arr


def create_lazy_dask_array(slide: openslide.OpenSlide, tile_size: int, level: int = 0) -> da.Array:
    """
    Crée un Dask array qui lit les tuiles à la demande.
    """
    lazy_tiles = OpenSlideLazyTiles(slide, level)
    
    return da.from_array(
        lazy_tiles,
        chunks=(tile_size, tile_size, 3),
        meta=np.array([], dtype=np.uint8),
        asarray=False,
        fancy=False,
        lock=threading.Lock()  # OpenSlide n'est pas thread-safe
    )


# -----------------------------
# Conversion avec génération manuelle de la pyramide
# -----------------------------
def convert_mrxs_to_omezarr(
    mrxs_path: str,
    out_zarr: str,
    tile_size: int,
    quality: int,
    downscale: float,
    max_levels: int,
    compression: str,  # "blosc", "jpeg", "jpegxl"
    base_level: int,
    use_native_levels: bool,
    log_callback: Callable[[str], None],
    progress_callback: Callable[[int, int], None],
):
    """
    Convertit un fichier MRXS en OME-Zarr pyramidal.
    
    Args:
        compression: Type de compression ("blosc", "jpeg", "jpegxl")
    """
    try:
        log_callback("Ouverture du fichier MRXS…")
        slide = openslide.OpenSlide(mrxs_path)
        
        # Déterminer le niveau de base à utiliser
        if use_native_levels and base_level > 0 and base_level < slide.level_count:
            start_level = base_level
            base_downsample = slide.level_downsamples[base_level]
            w, h = slide.level_dimensions[base_level]
            log_callback(f"Utilisation du niveau natif {base_level} (×{base_downsample:.1f})")
        else:
            start_level = 0
            base_downsample = 1.0
            w, h = slide.dimensions
        
        log_callback(f"Dimensions de travail : {w:,} × {h:,} pixels")

        # Configurer le compresseur selon le type choisi
        compressor = None
        compression_name = "Aucune"
        
        if NUMCODECS_AVAILABLE:
            if compression == "jpegxl" and JPEGXL_AVAILABLE:
                from imagecodecs.numcodecs import JpegXl
                # Distance JPEG-XL : 0 = lossless, 1 = haute qualité, 15 = basse qualité
                distance = max(0.5, (100 - quality) / 10)
                compressor = JpegXl(level=distance, numthreads=4)
                compression_name = f"JPEG-XL (distance={distance:.1f})"
            elif compression == "jpeg" and JPEG_AVAILABLE:
                from imagecodecs.numcodecs import Jpeg
                compressor = Jpeg(level=quality)
                compression_name = f"JPEG (qualité={quality})"
            elif compression == "jpegxl" and not JPEGXL_AVAILABLE:
                log_callback("⚠ JPEG-XL non disponible, utilisation de Blosc/ZSTD")
                compressor = COMPRESSOR_BLOSC
                compression_name = "Blosc/ZSTD (fallback)"
            elif compression == "jpeg" and not JPEG_AVAILABLE:
                log_callback("⚠ JPEG non disponible, utilisation de Blosc/ZSTD")
                compressor = COMPRESSOR_BLOSC
                compression_name = "Blosc/ZSTD (fallback)"
            else:
                # blosc par défaut
                compressor = COMPRESSOR_BLOSC
                compression_name = "Blosc/ZSTD"
        else:
            log_callback("⚠ numcodecs non disponible, pas de compression")
        
        log_callback(f"Compression : {compression_name}")

        # Calculer le nombre de niveaux de pyramide
        min_dim = min(w, h)
        if max_levels <= 0:
            num_levels = max(1, int(math.log(min_dim / tile_size, downscale)) + 1)
        else:
            num_levels = max_levels
        
        log_callback(f"Niveaux de pyramide : {num_levels} (facteur ×{downscale:.2f})")

        # Créer le store Zarr (méthode universelle)
        out_path = Path(out_zarr)
        if out_path.exists():
            import shutil
            shutil.rmtree(out_path)
        
        # zarr.open_group fonctionne avec v2 et v3
        root = zarr.open_group(str(out_path), mode='w')

        # Préparer les métadonnées OME-Zarr
        datasets = []

        for level in range(num_levels):
            scale_factor = downscale ** level
            datasets.append({
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [scale_factor, scale_factor, 1]}
                ]
            })

        # Métadonnées multiscale OME-Zarr
        root.attrs["multiscales"] = [{
            "version": "0.4",
            "name": Path(mrxs_path).stem,
            "axes": [
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
                {"name": "c", "type": "channel"}
            ],
            "datasets": datasets,
            "type": "gaussian",  # méthode de downsampling
            "metadata": {
                "source": str(mrxs_path),
                "method": "openslide"
            }
        }]

        total_tiles = 0
        processed_tiles = 0

        # Calculer le nombre total de tuiles pour la progression
        for lvl in range(num_levels):
            scale_factor = downscale ** lvl
            tw = int(w / scale_factor)
            th = int(h / scale_factor)
            total_tiles += math.ceil(th / tile_size) * math.ceil(tw / tile_size)

        log_callback(f"Tuiles totales à traiter : {total_tiles:,}")

        # Estimation du temps de conversion
        est_low, est_mid, est_high = estimate_conversion_time(
            total_tiles=total_tiles,
            tile_size=tile_size,
            compression=compression,
            use_native_levels=use_native_levels,
            base_level=base_level
        )
        log_callback(f"⏱ Temps estimé : {format_duration(est_mid)} (entre {format_duration(est_low)} et {format_duration(est_high)})")
        
        # Démarrer le chronomètre
        start_time = time.time()

        # Générer chaque niveau
        for level in range(num_levels):
            scale_factor = downscale ** level
            level_w = int(w / scale_factor)
            level_h = int(h / scale_factor)

            log_callback(f"Niveau {level} : {level_w:,} × {level_h:,} (×{scale_factor:.2f})")

            # Trouver le meilleur niveau OpenSlide disponible
            # Le scale_factor est relatif au niveau de base, il faut le convertir en absolu
            absolute_scale = base_downsample * scale_factor
            os_level = slide.get_best_level_for_downsample(absolute_scale)
            os_downsample = slide.level_downsamples[os_level]
            
            log_callback(f"  → Lecture depuis niveau OpenSlide {os_level} (×{os_downsample:.1f})")
            
            # Créer le dataset Zarr pour ce niveau
            # Utiliser zarr.open_array qui fonctionne avec v2 et v3
            array_path = out_path / str(level)
            
            if ZARR_V3:
                # En Zarr v3, créer via le groupe
                try:
                    # Essayer avec compressor (certaines versions)
                    zarr_array = root.create_array(
                        str(level),
                        shape=(level_h, level_w, 3),
                        chunks=(tile_size, tile_size, 3),
                        dtype=np.uint8,
                        overwrite=True
                    )
                except Exception as e:
                    log_callback(f"⚠ Fallback création array: {e}")
                    zarr_array = zarr.open_array(
                        str(array_path),
                        mode='w',
                        shape=(level_h, level_w, 3),
                        chunks=(tile_size, tile_size, 3),
                        dtype=np.uint8
                    )
            else:
                # Zarr v2 classique
                zarr_array = root.create_dataset(
                    str(level),
                    shape=(level_h, level_w, 3),
                    chunks=(tile_size, tile_size, 3),
                    dtype=np.uint8,
                    compressor=compressor,
                    overwrite=True
                )

            # Écrire tuile par tuile
            n_tiles_y = math.ceil(level_h / tile_size)
            n_tiles_x = math.ceil(level_w / tile_size)

            for ty in range(n_tiles_y):
                for tx in range(n_tiles_x):
                    # Coordonnées de la tuile dans ce niveau
                    y0 = ty * tile_size
                    x0 = tx * tile_size
                    y1 = min(y0 + tile_size, level_h)
                    x1 = min(x0 + tile_size, level_w)
                    tile_h = y1 - y0
                    tile_w = x1 - x0

                    # Coordonnées niveau 0 absolu pour OpenSlide
                    # (le niveau de base a déjà un downscale de base_downsample)
                    absolute_scale = base_downsample * scale_factor
                    os_x0 = int(x0 * absolute_scale)
                    os_y0 = int(y0 * absolute_scale)

                    # Taille à lire au niveau OpenSlide optimal
                    read_w = max(1, int(tile_w * absolute_scale / os_downsample))
                    read_h = max(1, int(tile_h * absolute_scale / os_downsample))

                    # Lire et redimensionner si nécessaire
                    region = slide.read_region((os_x0, os_y0), os_level, (read_w, read_h))
                    region = region.convert("RGB")

                    if region.size != (tile_w, tile_h):
                        from PIL import Image
                        region = region.resize((tile_w, tile_h), Image.Resampling.LANCZOS)

                    tile_data = np.asarray(region)

                    # Écrire dans Zarr
                    zarr_array[y0:y1, x0:x1, :] = tile_data

                    processed_tiles += 1
                    progress_callback(processed_tiles, total_tiles)

        slide.close()
        
        # Afficher le temps réel de conversion
        elapsed_time = time.time() - start_time
        time_per_tile = (elapsed_time / total_tiles * 1000) if total_tiles > 0 else 0
        log_callback(f"✓ Conversion terminée avec succès !")
        log_callback(f"⏱ Temps réel : {format_duration(elapsed_time)} ({time_per_tile:.1f} ms/tuile)")
        return True

    except Exception as e:
        log_callback(f"✗ Erreur : {e}")
        import traceback
        log_callback(traceback.format_exc())
        return False


# -----------------------------
# GUI
# -----------------------------
class ConverterGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MRXS → OME-Zarr Converter")
        self.geometry("900x800")
        self.resizable(True, True)

        # Variables de paramètres
        self.tile_size = tk.IntVar(value=512)
        self.quality = tk.IntVar(value=85)
        self.downscale = tk.DoubleVar(value=2.0)
        self.max_levels = tk.IntVar(value=0)
        self.compression = tk.StringVar(value="blosc")  # blosc, jpeg, jpegxl
        
        # Grossissement source et cible pour calcul auto
        self.source_mag = tk.DoubleVar(value=40.0)
        self.target_mag = tk.DoubleVar(value=0.0)  # 0 = désactivé
        
        # Option pour utiliser les niveaux natifs
        self.use_native_levels = tk.BooleanVar(value=True)
        self.base_level = tk.IntVar(value=0)
        
        # Preset sélectionné
        self.preset = tk.StringVar(value="Défaut")
        
        # Option pour compresser en ZIP après conversion
        self.create_zip = tk.BooleanVar(value=False)
        self.zip_level = tk.IntVar(value=0)  # 0 = stockage, 1-9 = compression
        self.delete_folder_after_zip = tk.BooleanVar(value=True)

        # File d'attente : liste de tuples (mrxs_path, out_path, status)
        self.queue_items = []  # [(mrxs_path, out_path), ...]
        
        # État de conversion
        self._converting = False
        self._cancel_requested = False
        self._current_item_index = 0
        
        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(0, weight=1)

        # === File d'attente ===
        queue_frame = ttk.LabelFrame(frm, text="File d'attente", padding=10)
        queue_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        queue_frame.columnconfigure(0, weight=1)
        queue_frame.rowconfigure(1, weight=1)
        frm.rowconfigure(0, weight=1)

        # Boutons de gestion de la file
        queue_btn_frame = ttk.Frame(queue_frame)
        queue_btn_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        ttk.Button(queue_btn_frame, text="➕ Ajouter fichier(s)", command=self._add_files).pack(side="left", padx=(0, 5))
        ttk.Button(queue_btn_frame, text="📁 Ajouter dossier", command=self._add_folder).pack(side="left", padx=(0, 5))
        ttk.Button(queue_btn_frame, text="🗑 Retirer sélection", command=self._remove_selected).pack(side="left", padx=(0, 5))
        ttk.Button(queue_btn_frame, text="🧹 Vider la liste", command=self._clear_queue).pack(side="left", padx=(0, 5))
        
        self.queue_count_label = ttk.Label(queue_btn_frame, text="0 fichier(s)")
        self.queue_count_label.pack(side="right", padx=5)

        # Treeview pour la file d'attente
        tree_frame = ttk.Frame(queue_frame)
        tree_frame.grid(row=1, column=0, sticky="nsew")
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        
        columns = ("status", "input", "output")
        self.queue_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=6)
        self.queue_tree.heading("status", text="État")
        self.queue_tree.heading("input", text="Fichier MRXS")
        self.queue_tree.heading("output", text="Sortie Zarr")
        self.queue_tree.column("status", width=80, minwidth=60)
        self.queue_tree.column("input", width=300, minwidth=150)
        self.queue_tree.column("output", width=300, minwidth=150)
        
        queue_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.queue_tree.yview)
        self.queue_tree.configure(yscrollcommand=queue_scroll.set)
        
        self.queue_tree.grid(row=0, column=0, sticky="nsew")
        queue_scroll.grid(row=0, column=1, sticky="ns")
        
        # Double-clic pour éditer la sortie
        self.queue_tree.bind("<Double-1>", self._edit_output_path)

        # === Presets ===
        preset_frame = ttk.LabelFrame(frm, text="Presets", padding=10)
        preset_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        
        preset_row = ttk.Frame(preset_frame)
        preset_row.pack(fill="x")
        
        ttk.Label(preset_row, text="Profil :").pack(side="left")
        self.preset_combo = ttk.Combobox(
            preset_row, 
            textvariable=self.preset, 
            values=list(PRESETS.keys()), 
            width=15, 
            state="readonly"
        )
        self.preset_combo.pack(side="left", padx=5)
        self.preset_combo.bind("<<ComboboxSelected>>", self._apply_preset)
        
        self.preset_desc_label = ttk.Label(preset_row, text=PRESETS["Défaut"]["description"], foreground="gray")
        self.preset_desc_label.pack(side="left", padx=10)

        # === Paramètres ===
        params = ttk.LabelFrame(frm, text="Paramètres (appliqués à tous les fichiers)", padding=10)
        params.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        # Ligne 1 : Taille tuiles + Downscale
        ttk.Label(params, text="Taille tuiles :").grid(row=0, column=0, sticky="w")
        tile_combo = ttk.Combobox(params, textvariable=self.tile_size, values=[256, 512, 1024, 2048], width=8)
        tile_combo.grid(row=0, column=1, sticky="w", padx=(5, 20))

        ttk.Label(params, text="Downscale :").grid(row=0, column=2, sticky="w")
        self.downscale_combo = ttk.Combobox(
            params, textvariable=self.downscale, 
            values=[1.5, 2.0, 2.5, 2.67, 3.0, 4.0, 5.0, 6.67, 8.0], 
            width=6
        )
        self.downscale_combo.grid(row=0, column=3, sticky="w", padx=5)

        # Ligne 2 : Grossissement source/cible
        ttk.Label(params, text="Grossissement source :").grid(row=1, column=0, sticky="w", pady=(10, 0))
        mag_source_combo = ttk.Combobox(
            params, textvariable=self.source_mag,
            values=[20.0, 40.0, 60.0, 80.0, 100.0],
            width=6
        )
        mag_source_combo.grid(row=1, column=1, sticky="w", padx=5, pady=(10, 0))
        
        ttk.Label(params, text="→ cible (0=off) :").grid(row=1, column=2, sticky="w", pady=(10, 0))
        mag_target_combo = ttk.Combobox(
            params, textvariable=self.target_mag,
            values=[0.0, 5.0, 10.0, 15.0, 20.0, 40.0],
            width=6
        )
        mag_target_combo.grid(row=1, column=3, sticky="w", padx=5, pady=(10, 0))
        
        ttk.Button(params, text="Calc", width=5, command=self._calc_downscale).grid(row=1, column=4, pady=(10, 0))

        # Ligne 3 : Niveaux max + Qualité
        ttk.Label(params, text="Niveaux max (0=auto) :").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Spinbox(params, textvariable=self.max_levels, from_=0, to=20, width=6).grid(row=2, column=1, sticky="w", padx=5, pady=(10, 0))

        ttk.Label(params, text="Qualité :").grid(row=2, column=2, sticky="w", pady=(10, 0))
        quality_frame = ttk.Frame(params)
        quality_frame.grid(row=2, column=3, columnspan=2, sticky="ew", pady=(10, 0))
        
        self.quality_scale = ttk.Scale(quality_frame, from_=30, to=95, variable=self.quality, orient="horizontal", length=120)
        self.quality_scale.pack(side="left")
        self.quality_label = ttk.Label(quality_frame, text="85")
        self.quality_label.pack(side="left", padx=5)
        self.quality.trace_add("write", self._update_quality_label)

        # Ligne 4 : Compression
        compress_frame = ttk.Frame(params)
        compress_frame.grid(row=3, column=0, columnspan=5, sticky="w", pady=(10, 0))
        
        ttk.Label(compress_frame, text="Compression :").pack(side="left")
        
        ttk.Radiobutton(compress_frame, text="Blosc/ZSTD", variable=self.compression, value="blosc").pack(side="left", padx=(5, 10))
        
        jpeg_text = "JPEG" + ("" if JPEG_AVAILABLE else " (non dispo)")
        self.jpeg_radio = ttk.Radiobutton(compress_frame, text=jpeg_text, variable=self.compression, value="jpeg")
        self.jpeg_radio.pack(side="left", padx=(0, 10))
        if not JPEG_AVAILABLE:
            self.jpeg_radio.configure(state="disabled")
        
        jpegxl_text = "JPEG-XL" + ("" if JPEGXL_AVAILABLE else " (non dispo)")
        self.jpegxl_radio = ttk.Radiobutton(compress_frame, text=jpegxl_text, variable=self.compression, value="jpegxl")
        self.jpegxl_radio.pack(side="left", padx=(0, 10))
        if not JPEGXL_AVAILABLE:
            self.jpegxl_radio.configure(state="disabled")

        # Ligne 5 : Niveaux natifs
        native_frame = ttk.Frame(params)
        native_frame.grid(row=4, column=0, columnspan=5, sticky="w", pady=(10, 0))
        
        ttk.Checkbutton(native_frame, text="Utiliser niveaux natifs", variable=self.use_native_levels).pack(side="left", padx=(0, 10))
        
        ttk.Label(native_frame, text="Niveau base:").pack(side="left")
        self.base_level_combo = ttk.Combobox(native_frame, textvariable=self.base_level, values=[0], width=3, state="readonly")
        self.base_level_combo.pack(side="left", padx=5)
        
        ttk.Button(native_frame, text="🔍 Inspecter", command=self._inspect_selected_file).pack(side="left", padx=(5, 10))
        
        self.level_info_label = ttk.Label(native_frame, text="(sélectionner un fichier)", foreground="gray")
        self.level_info_label.pack(side="left", padx=5)
        
        # Ligne 6 : Compression ZIP
        zip_frame = ttk.Frame(params)
        zip_frame.grid(row=5, column=0, columnspan=5, sticky="w", pady=(10, 0))
        
        ttk.Checkbutton(
            zip_frame, 
            text="📦 Créer ZIP après conversion", 
            variable=self.create_zip,
            command=self._toggle_zip_options
        ).pack(side="left")
        
        ttk.Label(zip_frame, text="Niveau:").pack(side="left", padx=(15, 2))
        self.zip_level_spin = ttk.Spinbox(
            zip_frame, 
            textvariable=self.zip_level, 
            from_=0, to=9, 
            width=3,
            state="disabled"
        )
        self.zip_level_spin.pack(side="left")
        
        self.zip_delete_check = ttk.Checkbutton(
            zip_frame, 
            text="Supprimer dossier", 
            variable=self.delete_folder_after_zip,
            state="disabled"
        )
        self.zip_delete_check.pack(side="left", padx=(15, 0))
        
        ttk.Label(zip_frame, text="(niveau 0 recommandé)", foreground="gray").pack(side="left", padx=(10, 0))

        # === Progression ===
        prog_frame = ttk.LabelFrame(frm, text="Progression", padding=10)
        prog_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        prog_frame.columnconfigure(0, weight=1)

        # Progression globale
        ttk.Label(prog_frame, text="Global :").grid(row=0, column=0, sticky="w")
        self.global_progress = ttk.Progressbar(prog_frame, mode="determinate")
        self.global_progress.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        prog_frame.columnconfigure(1, weight=1)
        
        self.global_progress_label = ttk.Label(prog_frame, text="0 / 0 fichiers")
        self.global_progress_label.grid(row=0, column=2, sticky="e", padx=(10, 0))

        # Progression fichier courant
        ttk.Label(prog_frame, text="Fichier :").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.file_progress = ttk.Progressbar(prog_frame, mode="determinate")
        self.file_progress.grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=(5, 0))
        
        self.file_progress_label = ttk.Label(prog_frame, text="En attente…")
        self.file_progress_label.grid(row=1, column=2, sticky="e", padx=(10, 0), pady=(5, 0))

        # === Log ===
        log_frame = ttk.LabelFrame(frm, text="Journal", padding=10)
        log_frame.grid(row=4, column=0, sticky="nsew", pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        frm.rowconfigure(4, weight=1)

        self.log = tk.Text(log_frame, height=8, state="disabled", wrap="word")
        self.log.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

        # === Boutons ===
        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=5, column=0, sticky="ew")

        self.convert_btn = ttk.Button(btn_frame, text="▶ Lancer la conversion", command=self._run_queue)
        self.convert_btn.pack(side="right", padx=(5, 0))
        
        self.cancel_btn = ttk.Button(btn_frame, text="⏹ Annuler", command=self._cancel_conversion, state="disabled")
        self.cancel_btn.pack(side="right")

        ttk.Button(btn_frame, text="Effacer log", command=self._clear_log).pack(side="left")

    # === Gestion de la file d'attente ===
    
    def _add_files(self):
        """Ajoute un ou plusieurs fichiers MRXS à la file"""
        paths = filedialog.askopenfilenames(
            title="Sélectionner des fichiers MRXS",
            filetypes=[("MRXS files", "*.mrxs"), ("All files", "*.*")]
        )
        for path in paths:
            self._add_to_queue(path)
    
    def _add_folder(self):
        """Ajoute tous les fichiers MRXS d'un dossier"""
        folder = filedialog.askdirectory(title="Sélectionner un dossier contenant des fichiers MRXS")
        if folder:
            folder_path = Path(folder)
            mrxs_files = list(folder_path.glob("*.mrxs")) + list(folder_path.glob("*.MRXS"))
            if not mrxs_files:
                messagebox.showinfo("Info", "Aucun fichier MRXS trouvé dans ce dossier.")
                return
            for mrxs_file in sorted(mrxs_files):
                self._add_to_queue(str(mrxs_file))
            self._log(f"Ajouté {len(mrxs_files)} fichier(s) depuis {folder_path.name}")
    
    def _add_to_queue(self, mrxs_path: str):
        """Ajoute un fichier à la file d'attente"""
        # Vérifier si déjà dans la liste
        for item in self.queue_items:
            if item[0] == mrxs_path:
                return  # Déjà présent
        
        # Générer le chemin de sortie automatiquement
        out_path = str(Path(mrxs_path).with_suffix(".ome.zarr"))
        
        self.queue_items.append((mrxs_path, out_path))
        
        # Ajouter à la Treeview
        filename = Path(mrxs_path).name
        out_filename = Path(out_path).name
        self.queue_tree.insert("", "end", values=("⏳ En attente", filename, out_filename))
        
        self._update_queue_count()
    
    def _remove_selected(self):
        """Retire les éléments sélectionnés de la file"""
        selected = self.queue_tree.selection()
        if not selected:
            return
        
        # Obtenir les indices à supprimer
        indices_to_remove = []
        for item in selected:
            idx = self.queue_tree.index(item)
            indices_to_remove.append(idx)
        
        # Supprimer en ordre inverse pour ne pas décaler les indices
        for idx in sorted(indices_to_remove, reverse=True):
            if idx < len(self.queue_items):
                del self.queue_items[idx]
            self.queue_tree.delete(self.queue_tree.get_children()[idx] if idx < len(self.queue_tree.get_children()) else selected[0])
        
        self._update_queue_count()
    
    def _clear_queue(self):
        """Vide la file d'attente"""
        if self._converting:
            messagebox.showwarning("Attention", "Impossible de vider la file pendant une conversion.")
            return
        
        self.queue_items.clear()
        for item in self.queue_tree.get_children():
            self.queue_tree.delete(item)
        self._update_queue_count()
    
    def _edit_output_path(self, event):
        """Permet de modifier le chemin de sortie en double-cliquant"""
        if self._converting:
            return
        
        item = self.queue_tree.identify_row(event.y)
        column = self.queue_tree.identify_column(event.x)
        
        if not item or column != "#3":  # Colonne "output"
            return
        
        idx = self.queue_tree.index(item)
        if idx >= len(self.queue_items):
            return
        
        current_out = self.queue_items[idx][1]
        new_path = filedialog.asksaveasfilename(
            title="Modifier le chemin de sortie",
            initialfile=Path(current_out).name,
            initialdir=Path(current_out).parent,
            defaultextension=".ome.zarr",
            filetypes=[("OME-Zarr", "*.ome.zarr"), ("Zarr", "*.zarr")]
        )
        
        if new_path:
            self.queue_items[idx] = (self.queue_items[idx][0], new_path)
            self.queue_tree.set(item, "output", Path(new_path).name)
    
    def _update_queue_count(self):
        """Met à jour le compteur de fichiers"""
        count = len(self.queue_items)
        self.queue_count_label.configure(text=f"{count} fichier(s)")
    
    def _update_item_status(self, index: int, status: str):
        """Met à jour le statut d'un élément dans la Treeview"""
        children = self.queue_tree.get_children()
        if index < len(children):
            self.queue_tree.set(children[index], "status", status)
            # Scroller vers l'élément courant
            self.queue_tree.see(children[index])
    
    def _inspect_selected_file(self):
        """Inspecte le fichier sélectionné pour voir ses niveaux natifs"""
        selected = self.queue_tree.selection()
        
        if not selected:
            # Si aucune sélection, proposer d'ouvrir un fichier
            path = filedialog.askopenfilename(
                title="Sélectionner un fichier MRXS à inspecter",
                filetypes=[("MRXS files", "*.mrxs"), ("All files", "*.*")]
            )
            if not path:
                return
        else:
            # Utiliser le fichier sélectionné
            idx = self.queue_tree.index(selected[0])
            if idx >= len(self.queue_items):
                return
            path = self.queue_items[idx][0]
        
        try:
            slide = openslide.OpenSlide(path)
            
            # Chercher le grossissement dans les propriétés
            mag = None
            for prop in ['openslide.objective-power', 'aperio.AppMag', 'hamamatsu.SourceLens']:
                if prop in slide.properties:
                    try:
                        mag = float(slide.properties[prop])
                        break
                    except ValueError:
                        pass
            
            if mag:
                self.source_mag.set(mag)
            
            w, h = slide.dimensions
            filename = Path(path).name
            self._log(f"\n─── Inspection : {filename} ───")
            self._log(f"Dimensions niveau 0 : {w:,} × {h:,} pixels")
            if mag:
                self._log(f"Grossissement détecté : {mag}×")
            
            # Afficher les niveaux disponibles
            num_levels = slide.level_count
            level_values = []
            
            self._log(f"Niveaux disponibles ({num_levels}) :")
            for lvl in range(num_levels):
                lw, lh = slide.level_dimensions[lvl]
                ds = slide.level_downsamples[lvl]
                effective_mag = mag / ds if mag else None
                
                if effective_mag:
                    info = f"  Niveau {lvl}: {lw:,}×{lh:,} (×{ds:.1f}, ~{effective_mag:.1f}×)"
                else:
                    info = f"  Niveau {lvl}: {lw:,}×{lh:,} (×{ds:.1f})"
                
                self._log(info)
                level_values.append(lvl)
            
            # Mettre à jour le combobox des niveaux
            self.base_level_combo['values'] = level_values
            if self.base_level.get() >= num_levels:
                self.base_level.set(0)
            
            # Stocker les infos pour le label
            self._level_info = []
            for lvl in range(num_levels):
                lw, lh = slide.level_dimensions[lvl]
                ds = slide.level_downsamples[lvl]
                effective_mag = mag / ds if mag else None
                self._level_info.append((lvl, lw, lh, ds, effective_mag))
            
            self._update_level_info_label()
            
            # Binding pour mettre à jour le label quand on change de niveau
            # (éviter les doublons de trace)
            try:
                self.base_level.trace_remove("write", self._level_trace_id)
            except (AttributeError, tk.TclError):
                pass
            self._level_trace_id = self.base_level.trace_add("write", lambda *args: self._update_level_info_label())
            
            slide.close()
            
        except Exception as e:
            self._log(f"Erreur lors de l'inspection : {e}")
            messagebox.showerror("Erreur", f"Impossible de lire le fichier :\n{e}")
    
    def _update_level_info_label(self):
        """Met à jour le label d'info du niveau sélectionné."""
        if not hasattr(self, '_level_info') or not self._level_info:
            self.level_info_label.configure(text="(sélectionner un fichier)", foreground="gray")
            return
        
        lvl = self.base_level.get()
        if lvl < len(self._level_info):
            _, lw, lh, ds, effective_mag = self._level_info[lvl]
            if effective_mag:
                self.level_info_label.configure(text=f"→ {lw:,}×{lh:,} (~{effective_mag:.1f}×)", foreground="black")
            else:
                self.level_info_label.configure(text=f"→ {lw:,}×{lh:,} (×{ds:.1f})", foreground="black")

    # === Paramètres ===
    
    def _update_quality_label(self, *args):
        self.quality_label.configure(text=str(self.quality.get()))
    
    def _toggle_zip_options(self):
        """Active/désactive les options ZIP selon la checkbox"""
        state = "normal" if self.create_zip.get() else "disabled"
        self.zip_level_spin.configure(state=state)
        self.zip_delete_check.configure(state=state)
    
    def _apply_preset(self, event=None):
        """Applique les paramètres du preset sélectionné."""
        preset_name = self.preset.get()
        if preset_name not in PRESETS:
            return
        
        preset = PRESETS[preset_name]
        
        # Appliquer les paramètres
        self.tile_size.set(preset["tile_size"])
        self.downscale.set(preset["downscale"])
        self.max_levels.set(preset["max_levels"])
        self.quality.set(preset["quality"])
        self.compression.set(preset["compression"])
        self.use_native_levels.set(preset["use_native_levels"])
        self.base_level.set(preset["base_level"])
        
        # Mettre à jour la description
        self.preset_desc_label.configure(text=preset["description"])
        
        # Log
        self._log(f"Preset '{preset_name}' appliqué")

    def _calc_downscale(self):
        """Calcule le downscale à partir des grossissements source et cible."""
        source = self.source_mag.get()
        target = self.target_mag.get()
        
        if target <= 0 or source <= 0:
            messagebox.showinfo("Info", "Définissez un grossissement cible > 0")
            return
        
        if target > source:
            messagebox.showwarning("Attention", "Le grossissement cible ne peut pas être supérieur à la source.")
            return
        
        downscale = source / target
        self.downscale.set(round(downscale, 2))
        self._log(f"Downscale calculé : {source}× → {target}× = facteur {downscale:.2f}")

    # === Logging et progression ===

    def _log(self, msg: str):
        """Thread-safe logging."""
        def _insert():
            self.log.configure(state="normal")
            self.log.insert("end", msg + "\n")
            self.log.see("end")
            self.log.configure(state="disabled")
        self.after(0, _insert)

    def _update_file_progress(self, current: int, total: int):
        """Met à jour la progression du fichier courant."""
        def _update():
            pct = (current / total * 100) if total > 0 else 0
            self.file_progress["value"] = pct
            self.file_progress_label.configure(text=f"{current:,} / {total:,} tuiles ({pct:.1f}%)")
        self.after(0, _update)
    
    def _update_global_progress(self, current: int, total: int):
        """Met à jour la progression globale."""
        def _update():
            pct = (current / total * 100) if total > 0 else 0
            self.global_progress["value"] = pct
            self.global_progress_label.configure(text=f"{current} / {total} fichiers")
        self.after(0, _update)

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # === Conversion ===
    
    def _cancel_conversion(self):
        """Demande l'annulation de la conversion"""
        if self._converting:
            self._cancel_requested = True
            self._log("⚠ Annulation demandée, veuillez patienter…")
            self.cancel_btn.configure(state="disabled")

    def _run_queue(self):
        """Lance la conversion de tous les fichiers de la file"""
        if self._converting:
            return
        
        if not self.queue_items:
            messagebox.showwarning("Attention", "La file d'attente est vide.\nAjoutez des fichiers MRXS à convertir.")
            return

        self._converting = True
        self._cancel_requested = False
        self._current_item_index = 0
        
        self.convert_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.global_progress["value"] = 0
        self.file_progress["value"] = 0
        
        total_files = len(self.queue_items)
        self._log(f"═══ Démarrage de la conversion de {total_files} fichier(s) ═══")
        
        def worker():
            success_count = 0
            error_count = 0
            
            for i, (mrxs_path, out_path) in enumerate(self.queue_items):
                if self._cancel_requested:
                    self._log(f"⚠ Conversion annulée après {i} fichier(s)")
                    # Marquer les restants comme annulés
                    for j in range(i, len(self.queue_items)):
                        self.after(0, lambda idx=j: self._update_item_status(idx, "⚪ Annulé"))
                    break
                
                self._current_item_index = i
                filename = Path(mrxs_path).name
                
                self._log(f"\n─── Fichier {i+1}/{total_files} : {filename} ───")
                self.after(0, lambda idx=i: self._update_item_status(idx, "🔄 En cours…"))
                self.after(0, lambda curr=i, tot=total_files: self._update_global_progress(curr, tot))
                
                # Reset progression fichier
                self.after(0, lambda: self.file_progress.configure(value=0))
                self.after(0, lambda: self.file_progress_label.configure(text="Démarrage…"))
                
                try:
                    success = convert_mrxs_to_omezarr(
                        mrxs_path=mrxs_path,
                        out_zarr=out_path,
                        tile_size=self.tile_size.get(),
                        quality=self.quality.get(),
                        downscale=self.downscale.get(),
                        max_levels=self.max_levels.get(),
                        compression=self.compression.get(),
                        base_level=self.base_level.get(),
                        use_native_levels=self.use_native_levels.get(),
                        log_callback=self._log,
                        progress_callback=self._update_file_progress,
                    )
                    
                    if success:
                        # Compression ZIP si demandée
                        if self.create_zip.get():
                            zip_success = self._create_zip_archive(out_path)
                            if zip_success:
                                self.after(0, lambda idx=i: self._update_item_status(idx, "✅ Terminé (ZIP)"))
                            else:
                                self.after(0, lambda idx=i: self._update_item_status(idx, "⚠ ZIP échoué"))
                        else:
                            self.after(0, lambda idx=i: self._update_item_status(idx, "✅ Terminé"))
                        success_count += 1
                    else:
                        self.after(0, lambda idx=i: self._update_item_status(idx, "❌ Erreur"))
                        error_count += 1
                        
                except Exception as e:
                    self._log(f"✗ Exception : {e}")
                    self.after(0, lambda idx=i: self._update_item_status(idx, "❌ Erreur"))
                    error_count += 1
            
            # Finalisation
            def finish():
                self._converting = False
                self._cancel_requested = False
                self.convert_btn.configure(state="normal")
                self.cancel_btn.configure(state="disabled")
                self._update_global_progress(len(self.queue_items), len(self.queue_items))
                
                if error_count == 0 and not self._cancel_requested:
                    self._log(f"\n═══ Conversion terminée : {success_count} fichier(s) réussi(s) ═══")
                    messagebox.showinfo("Terminé", f"Conversion réussie !\n{success_count} fichier(s) converti(s)")
                else:
                    self._log(f"\n═══ Conversion terminée : {success_count} réussi(s), {error_count} erreur(s) ═══")
                    if error_count > 0:
                        messagebox.showwarning("Terminé avec erreurs", 
                            f"Conversion terminée.\n{success_count} réussi(s), {error_count} erreur(s)\nVoir le journal pour les détails.")
            
            self.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()
    
    def _create_zip_archive(self, zarr_path_str: str) -> bool:
        """Crée une archive ZIP du dossier zarr"""
        import zipfile
        import shutil
        
        zarr_path = Path(zarr_path_str)
        zip_path = zarr_path.parent / f"{zarr_path.name}.zip"
        
        self._log(f"📦 Création du ZIP: {zip_path.name}")
        
        try:
            # Compter les fichiers
            all_files = list(zarr_path.rglob("*"))
            files = [f for f in all_files if f.is_file()]
            total = len(files)
            
            self._log(f"   {total} fichiers à compresser...")
            
            # Choisir le mode de compression
            level = self.zip_level.get()
            if level > 0:
                compression = zipfile.ZIP_DEFLATED
                self._log(f"   Compression niveau {level}")
            else:
                compression = zipfile.ZIP_STORED
                self._log(f"   Mode stockage (niveau 0)")
            
            # Mettre à jour le label de progression
            self.after(0, lambda: self.file_progress_label.configure(text="Compression ZIP…"))
            
            with zipfile.ZipFile(zip_path, 'w', compression=compression, 
                                compresslevel=level if level > 0 else None) as zf:
                for i, file in enumerate(files):
                    # Chemin relatif dans le ZIP (inclut le nom du dossier zarr)
                    arcname = file.relative_to(zarr_path.parent)
                    zf.write(file, arcname)
                    
                    if i % 50 == 0:
                        pct = (i + 1) / total * 100
                        self.after(0, lambda p=pct: self.file_progress.configure(value=p))
            
            # Taille finale
            src_size = sum(f.stat().st_size for f in files)
            dst_size = zip_path.stat().st_size
            ratio = dst_size / src_size * 100 if src_size > 0 else 100
            
            self._log(f"   ✓ ZIP créé: {src_size/1e6:.1f}MB → {dst_size/1e6:.1f}MB ({ratio:.1f}%)")
            
            # Supprimer le dossier source si demandé
            if self.delete_folder_after_zip.get():
                self._log(f"   Suppression du dossier {zarr_path.name}...")
                shutil.rmtree(zarr_path)
                self._log(f"   ✓ Dossier supprimé")
            
            return True
            
        except Exception as e:
            self._log(f"   ✗ Erreur ZIP: {e}")
            return False


if __name__ == "__main__":
    app = ConverterGUI()
    app.mainloop()