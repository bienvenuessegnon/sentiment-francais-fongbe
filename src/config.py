"""Chemins, constantes et paramètres du projet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXTERNAL_DIR = DATA / "external"  # sources volumineuses, non versionnées
RAW_DIR = DATA / "raw"
PROCESSED_DIR = DATA / "processed"
REPORTS = ROOT / "reports"
FIGURES_DIR = REPORTS / "figures"
MODELS_DIR = ROOT / "models"

SEED = 42

# --- Sources ---
# Allociné (Blard) : critiques de films, polarité binaire, licence MIT
ALLOCINE_HF = "https://huggingface.co/datasets/tblard/allocine/resolve/main/allocine"
ALLOCINE_FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}

# Corpus parallèle Fon–Français consolidé (CC BY 4.0), d'après :
# Dossou et al. 2021, Zenodo 4432712 ; Degila et al. 2020, Zenodo 4266935
FFR_HF = "https://huggingface.co/datasets/Flex1-tech/fon-french-parallel-corpus/resolve/main/data/fon_fr.csv"

# Corpus « Fon Sentiment » (Owusu, 2025) : étiquettes automatiques, licence MIT.
# Conservé uniquement pour un audit de qualité, pas pour l'entraînement.
FON_SENT_HF = (
    "https://huggingface.co/datasets/michsethowusu/fon-sentiments-corpus/resolve/main/"
    "data/train-00000-of-00001-878ca0dd744ad48f.parquet"
)

# --- Échantillonnage français (Allociné, découpage officiel conservé) ---
FR_MIN_CHARS = 40
FR_MAX_CHARS = 700
FR_TRAIN_N = 10000
FR_VAL_N = 2000
FR_TEST_N = 2000

# --- Projection d'étiquettes vers le fongbé ---
PROJ_MIN_WORDS_FR = 4
PROJ_MIN_WORDS_FON = 3
PROJ_MIN_CHARS_FR = 20
PROJ_MAX_CHARS_FR = 220
PROJ_PROBA_POS = 0.92
PROJ_PROBA_NEG = 0.08

# Jeu vérifié : seuil plus strict, puis relecture du français
VERIF_PROBA_POS = 0.97
VERIF_PROBA_NEG = 0.03
VERIF_TARGET_N = 120

# --- TF-IDF ---
WORD_MAX_FEATURES = 20000
CHAR_MAX_FEATURES = 20000

# --- Transformeur (sous-ensemble, CPU) ---
TRANSFORMER_NAME = "microsoft/Multilingual-MiniLM-L12-H384"
TRANSFORMER_TRAIN_N = 4000
TRANSFORMER_MAX_LEN = 128
TRANSFORMER_EPOCHS = 1
TRANSFORMER_BATCH = 8

LABEL_NAMES = {0: "négatif", 1: "positif"}
LANG_NAMES = {"fr": "français", "fon": "fongbé"}

# Couleurs des figures
COLOR_NEG = "#B91C1C"
COLOR_POS = "#0F766E"
COLOR_FR = "#4338CA"
COLOR_FON = "#C2410C"
COLOR_ACCENT = "#4338CA"
