"""
Configuration file for the Waste Classification Deep Learning Project
הגדרות כלליות לכל הפרויקט - מרכזי אחד לכל הערכים החשובים
"""

import torch
from pathlib import Path

# ========================
# נתיבי תיקיות (Paths)
# ========================
# נתיב לתיקיית הפרויקט הראשית
PROJECT_ROOT = Path(__file__).parent.parent  # final_project/
DATA_PATH = PROJECT_ROOT.parent / "recycling_dataset"  # התיקייה של ה-dataset

# תיקיות לשמירת תוצאות
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = RESULTS_DIR / "logs"

# ========================
# פרמטרים של ה-Dataset
# ========================
NUM_CLASSES = 4  # 4 קטגוריות: cardboard, aluminum, glass, plastic
CLASS_NAMES = [
    'cardboard box waste',
    'crushed aluminum can',
    'glass bottle waste',
    'plastic bottle waste'
]

# גודל התמונות (כל התמונות יעברו resize לגודל הזה)
IMAGE_SIZE = 224  # 224x224 - גודל סטנדרטי לרשתות עמוקות

# חלוקת Dataset: Train / Validation / Test
TRAIN_SPLIT = 0.7  # 70% לאימון
VAL_SPLIT = 0.15   # 15% לוולידציה
TEST_SPLIT = 0.15  # 15% לטסט

# ========================
# Hyperparameters - פרמטרים לאימון
# ========================
BATCH_SIZE = 32  # כמה תמונות בכל batch
LEARNING_RATE = 0.001  # קצב למידה התחלתי
NUM_EPOCHS = 25  # הורדנו ל-25 כדי לחסוך זמן (היה 50)

# Weight Decay (L2 Regularization)
WEIGHT_DECAY = 1e-4  # 0.0001

# Early Stopping
PATIENCE = 5  # כמה epochs לחכות ללא שיפור לפני עצירה

# Dropout rate (לשכבות Dropout)
DROPOUT_RATE = 0.5

# ========================
# הגדרות GPU / CPU
# ========================
# בדיקה אוטומטית אם יש GPU זמין
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================
# הגדרות Data Augmentation
# ========================
# פרמטרים לשינויים רנדומליים על התמונות (כדי להגדיל את ה-dataset "מלאכותית")
AUGMENTATION_PARAMS = {
    'rotation_degrees': 20,        # סיבוב אקראי עד 20 מעלות
    'brightness': 0.2,             # שינוי בהירות
    'contrast': 0.2,               # שינוי ניגודיות
    'horizontal_flip_prob': 0.5,   # סיכוי להיפוך אופקי
}

# ========================
# הגדרות אופטימיזציה
# ========================
# רשימת optimizers שנרצה להשוות
OPTIMIZERS = ['adam', 'sgd']

# פרמטרים ל-SGD
SGD_MOMENTUM = 0.9

# Learning Rate Scheduler
USE_SCHEDULER = True
SCHEDULER_STEP_SIZE = 10  # כל כמה epochs להוריד את ה-learning rate
SCHEDULER_GAMMA = 0.5     # בכמה להכפיל את ה-learning rate (0.5 = חצי)

# ========================
# הגדרות Early Stopping
# ========================
# עוצרים את האימון אם ה-validation loss לא משתפר
EARLY_STOPPING_PATIENCE = 10  # כמה epochs לחכות בלי שיפור

# ========================
# Random Seed - לשחזוריות
# ========================
# קובעים seed כדי שהתוצאות יהיו זהות בכל הרצה
RANDOM_SEED = 42

# ========================
# הגדרות לוגים והדפסות
# ========================
# כל כמה batches להדפיס את הפרוגרס
PRINT_EVERY = 10

# ========================
# פונקציית עזר להדפסת כל ההגדרות
# ========================
def print_config():
    """
    מדפיסה את כל ההגדרות הנוכחיות - שימושי לתיעוד ניסויים
    """
    print("=" * 60)
    print("🔧 Project Configuration")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data path: {DATA_PATH}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 60)

if __name__ == "__main__":
    # אם מריצים את הקובץ הזה ישירות, הוא ידפיס את כל ההגדרות
    print_config()
