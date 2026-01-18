# 🗑️ Waste Classification with Deep Learning - Final Project
## CNN מ-Scratch, Transfer Learning, ו-ResNet50 ב-PyTorch

---

## 📋 תיאור הפרויקט

פרויקט סופי בקורס **Introduction to Deep Learning** - בניית מערכת לסיווג פסולת למחזור באמצעות רשתות נוירונים קונבולוציוניות (CNN).

### 🎯 מטרות הפרויקט:
1. **בניית CNN מאפס** - עיצוב וניסוי עם ארכיטקטורות שונות
2. **השוואת אופטימיזרים** - Adam vs SGD with momentum
3. **Batch Normalization ו-Regularization** - שיפור ביצועים ומניעת overfitting
4. **Transfer Learning** - שימוש במודל שאומן על CIFAR-10
5. **ResNet50 Pretrained** - Fine-tuning של מודל מאומן על ImageNet

---

## 📊 Dataset

**Waste Recycling Dataset** - 4 קטגוריות של פסולת למחזור:
- 🟤 Cardboard box waste (קרטון)
- ⚪ Crushed aluminum can (פחיות אלומיניום)
- 🟢 Glass bottle waste (בקבוקי זכוכית)
- 🔵 Plastic bottle waste (בקבוקי פלסטיק)

**מיקום הנתונים:** `../recycling_dataset/`

**חלוקת הנתונים:**
- Train: 70%
- Validation: 15%
- Test: 15%

---

## 🗂️ מבנה הפרויקט

```
final_project/
├── notebooks/                                  # מחברות Jupyter
│   ├── 01_data_exploration_and_dataloader.ipynb   # ✅ סקירת נתונים + DataLoader
│   ├── 02_cnn_from_scratch.ipynb                  # ⚠️ CNN מאפס (גרסה ישנה)
│   ├── 02_cnn_from_scratch_v2.ipynb              # ✅ CNN מאפס - גרסה מיוטבת!
│   ├── 03_batch_norm_regularization.ipynb         # 🚧 Batch Norm + Dropout
│   ├── 04_transfer_learning_cifar10.ipynb         # 🚧 Transfer Learning מ-CIFAR-10
│   ├── 05_resnet50_pretrained.ipynb               # 🚧 ResNet50 + Fine-tuning
│   └── 06_final_comparison.ipynb                  # 🚧 השוואה ומסקנות
│
├── src/                                       # קוד Python לשימוש חוזר
│   ├── config.py                              # ✅ הגדרות כלליות (עודכן!)
│   ├── models.py                              # 🚧 הגדרות ארכיטקטורות CNN
│   └── utils.py                               # פונקציות עזר (train, evaluate, plot)
│
├── results/                                   # תוצאות ניסויים
│   ├── models/                                # משקולות מודלים (.pth)
│   ├── plots/                                 # גרפים ותרשימים
│   └── logs/                                  # לוגים של ניסויים (.json)
│
├── requirements.txt                           # רשימת ספריות נדרשות
└── README.md                                  # המסמך הזה
```

---

## 🚀 התחלת עבודה

### 1. התקנת סביבת העבודה

```bash
# שכפול הפרויקט (אם לא עשית כבר)
cd waste-classifier-fastai/final_project

# התקנת הספריות הנדרשות
pip install -r requirements.txt
```

### 2. בדיקת GPU (אופציונלי אבל מומלץ מאוד)

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

### 3. הרצת המחברות לפי הסדר

התחל עם `01_data_exploration_and_dataloader.ipynb` והמשך לפי הסדר המספרי.

---

## 📚 מה תלמד בכל מחברת?

### 📘 Notebook 1: Data Exploration & DataLoader
**מה נלמד:**
- איך PyTorch טוען תמונות מתיקיות
- מהם `Dataset` ו-`DataLoader`
- Data Augmentation והשפעתו
- Stratified Split (חלוקה מאוזנת)
- Normalization וחשיבותו

**תוצאות:**
- ויזואליזציה של הנתונים
- DataLoaders מוכנים לאימון
- dataset_splits.json

---

### 📗 Notebook 2: CNN from Scratch ⚡ **OPTIMIZED!**
**מה נלמד:**
- בניית ארכיטקטורות CNN בסיסיות
- Forward pass, Loss function, Backpropagation
- Early Stopping לחיסכון בזמן
- Dropout חזק למניעת overfitting

**📌 גרסה 2 (מיוטבת):**
- 🚀 **12M parameters** (במקום 102M!)
- ⏱️ **20-30 דקות אימון** (במקום 2+ שעות)
- 📉 **Dropout 0.6/0.5** למניעת overfitting
- 🛑 **Early Stopping** (patience=5)
- 📊 **3 Conv + 3 FC layers** (16→32→64→256→128→4)

**ניסויים:**
1. ~~SimpleCNN (גרסה 1 - ישנה)~~ ⚠️
2. **SimpleCNN_v2 (גרסה מיוטבת)** ✅

**תוצאות:**
- Training curves
- Validation accuracy
- מודלים שמורים ב-`results/models/`
- JSON logs ב-`results/logs/`

---

### 📙 Notebook 3: Batch Normalization & Regularization 🚧
**מה נלמד:**
- מהו Batch Normalization ואיך הוא עובד
- Dropout - מהו ואיך להשתמש בו
- Weight Decay (L2 Regularization)
- ניתוח Overfitting vs Generalization

**ניסויים:**
1. הוספת BatchNorm לארכיטקטורה הטובה ביותר
2. השוואה: עם/בלי BatchNorm
3. ניסוי עם Dropout rates שונים
4. שילוב של כל הטכניקות

**תוצאות:**
- השוואת stability
- מהירות convergence
- Overfitting analysis

---

### 📕 Notebook 4: Transfer Learning from CIFAR-10
**מה נלמד:**
- מהו Transfer Learning ומתי להשתמש בו
- Pre-training על dataset חיצוני
- Fine-tuning טכניקות
- Feature extraction vs Fine-tuning

**תהליך:**
1. אימון מודל על CIFAR-10
2. שמירת משקולות
3. טעינת המודל והחלפת ה-classifier
4. Fine-tuning על recycling dataset
5. השוואה עם אימון מאפס

**תוצאות:**
- השוואת זמני convergence
- Accuracy comparison
- ניתוח התועלת של pre-training

---

### 📔 Notebook 5: ResNet50 Pretrained
**מה נלמד:**
- ארכיטקטורת ResNet והמהפכה שלה
- טעינת מודלים מאומנים מראש
- Freezing vs Unfreezing layers
- Learning rate strategies לfine-tuning

**ניסויים:**
1. טעינת ResNet50 מ-ImageNet
2. החלפת fully connected layer
3. Fine-tuning עם layers מוקפאים
4. Unfreezing שכבות עליונות
5. ניסוי עם learning rates שונים

**תוצאות:**
- ביצועים לעומת CNN מאפס
- ניתוח זמני אימון
- Stability analysis

---

### 📓 Notebook 6: Final Comparison & Conclusions
**מה נעשה:**
- השוואה בין כל הגישות
- ניתוח כמותי (accuracy, loss, training time)
- ניתוח איכותי (stability, generalization)
- Confusion matrices
- הצגת דוגמאות טובות/רעות

**מסקנות:**
- איזו גישה עבדה הכי טוב?
- מתי כדאי להשתמש בכל גישה?
- מה היינו עושים אחרת עם יותר זמן/נתונים?

---

## 🔧 הגדרות ופרמטרים (config.py)

```python
# הגדרות בסיסיות
IMAGE_SIZE = 224          # גודל תמונות
NUM_CLASSES = 4           # מספר קטגוריות
BATCH_SIZE = 32           # גודל batch
LEARNING_RATE = 0.001     # קצב למידה
NUM_EPOCHS = 50           # מספר epochs

# Regularization
DROPOUT_RATE = 0.5        # אחוז dropout
WEIGHT_DECAY = 1e-4       # L2 regularization

# Data Augmentation
rotation_degrees = 20     # סיבוב אקראי
horizontal_flip_prob = 0.5  # הסתברות להיפוך
```

---

## 📈 תוצאות צפויות

### 🎯 Baseline (CNN מאפס):
- **Accuracy:** ~70-80% (תלוי בגודל dataset)
- **Training time:** מהיר יחסית
- **Pros:** הבנה מלאה של הארכיטקטורה
- **Cons:** דורש dataset גדול, פחות יציב

### 🔄 Transfer Learning (CIFAR-10):
- **Accuracy:** ~75-85%
- **Training time:** קצר יותר (convergence מהיר)
- **Pros:** למידה של features כלליים
- **Cons:** CIFAR-10 שונה מהמשימה שלנו

### 🏆 ResNet50 (Pretrained):
- **Accuracy:** ~85-95%
- **Training time:** קצר (רק fine-tuning)
- **Pros:** ביצועים מצוינים, יציב, מהיר
- **Cons:** "קופסה שחורה", דורש הרבה זיכרון

---

## 💡 טיפים להצלחה

### ✅ Do's:
- **תעד הכל** - כל ניסוי, כל החלטה, כל תוצאה
- **שמור משקולות** - אל תריץ אימון ארוך פעמיים
- **השתמש ב-GPU** - חוסך זמן רב (Google Colab חינמי!)
- **התחל פשוט** - CNN קטן → תוצאות מהירות → הבנה
- **השוה הוגן** - אותם היפרפרמטרים לכל הגישות

### ❌ Don'ts:
- **אל תגע ב-test set** - רק בסוף לבדיקה סופית!
- **אל תזלזל בvalidation** - חשוב לזיהוי overfitting
- **אל תשכח seed** - לשחזוריות תוצאות
- **אל תעתיק קוד בלי הבנה** - חייבים להבין כל שורה

---

## 📊 מה להגיש?

### 1. **קוד (GitHub)**
- ✅ כל המחברות מריצות ועובדות
- ✅ קוד מתועד עם הערות
- ✅ README מעודכן (זה!)
- ✅ requirements.txt

### 2. **מצגת (PPT)**
- **Slide 1:** כותרת + שם
- **Slide 2-3:** הבעיה וה-dataset
- **Slide 4-8:** כל ניסוי (מטרה, שיטה, תוצאות)
- **Slide 9-10:** השוואות וגרפים
- **Slide 11:** מסקנות
- **Slide 12:** רפלקציה (מה למדת? מה היית עושה אחרת?)

### 3. **הערות חשובות:**
- ההסבר חשוב יותר מהדיוק!
- הראה שהבנת למה דברים עבדו/לא עבדו
- גרפים ברורים וקריאים
- מסקנות ממוקדות

---

## 🎓 קריטריונים להערכה

| קריטריון | משקל | מה בודקים? |
|----------|------|------------|
| **ניסויים** | 30% | מגוון, שיטתיות, יצירתיות |
| **ניתוח** | 25% | עומק ההבנה, איכות ההסברים |
| **תיעוד** | 20% | קוד נקי, README, הערות |
| **תוצאות** | 15% | גרפים, השוואות, ויזואליזציה |
| **מסקנות** | 10% | רפלקציה, תובנות, למידה |

**שימו לב:** דיוק המודל הוא רק חלק קטן מהציון!

---

## 📚 משאבים נוספים

### PyTorch Documentation:
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html)
- [torchvision.models](https://pytorch.org/vision/stable/models.html)

### מאמרים חשובים:
- [ResNet Paper](https://arxiv.org/abs/1512.03385) - He et al., 2015
- [Batch Normalization](https://arxiv.org/abs/1502.03167) - Ioffe & Szegedy, 2015
- [Dropout](https://jmlr.org/papers/v15/srivastava14a.html) - Srivastava et al., 2014

### קורסים:
- [CS231n - Stanford](http://cs231n.stanford.edu/)
- [Deep Learning Specialization - Coursera](https://www.coursera.org/specializations/deep-learning)

---

## 🤝 עזרה ותמיכה

### נתקעת? הנה כמה טיפים:

1. **בעיות טכניות:**
   - בדוק שכל הספריות מותקנות (`pip list`)
   - וודא שהנתיב ל-dataset נכון
   - נסה להריץ עם batch size קטן יותר

2. **תוצאות לא טובות:**
   - זה תקין בהתחלה!
   - התחל עם מודל פשוט
   - בדוק overfitting (train vs val loss)
   - נסה data augmentation חזק יותר

3. **אימון איטי:**
   - השתמש ב-GPU (Colab/Kaggle)
   - הקטן batch size
   - הקטן את גודל המודל

---

## ✅ Checklist לפני הגשה

- [ ] כל המחברות רצות ללא שגיאות
- [ ] יש תיעוד והסברים בקוד
- [ ] כל הגרפים נשמרו ב-`results/plots/`
- [ ] README מעודכן ומלא
- [ ] requirements.txt רלוונטי
- [ ] יש מסקנות ורפלקציה
- [ ] המצגת מוכנה
- [ ] הקוד ב-GitHub מעודכן

---

## 📞 יצירת קשר

**שם:** דן  
**פרויקט:** Waste Classification with Deep Learning  
**קורס:** Introduction to Deep Learning  
**תאריך:** ינואר 2026

---

**בהצלחה! 🚀**

זכור: המטרה היא **ללמוד ולהבין**, לא רק לקבל accuracy גבוה.
תהנה מהתהליך! 😊
