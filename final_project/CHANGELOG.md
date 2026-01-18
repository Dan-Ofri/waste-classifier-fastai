# Changelog - Final Project Updates

## [v2.0] - 2026-01-18

### 🎯 Major Optimizations

#### Architecture Changes
- **Reduced model size:** 102M → 12M parameters (8.5x smaller!)
- **Conv layers:** Changed from 2 to 3 layers with reduced channels (16→32→64)
- **FC layers:** Added intermediate layer (256→128→4) for better gradient flow
- **Dropout:** Increased to 0.6/0.5 for better regularization

#### Training Improvements
- **Epochs:** Reduced from 50 to 25 (faster training)
- **Early Stopping:** Added with patience=5 (stops when no improvement)
- **Training time:** ~20-30 min (was 2+ hours)

#### Files Added
- `02_cnn_from_scratch_v2.ipynb` - Optimized training notebook
- `config.py` - Updated with PATIENCE and reduced NUM_EPOCHS
- `.gitignore` - Proper git ignore rules
- `CHANGELOG.md` - This file

### 📊 Expected Improvements
- ⏱️ Training time: 4-6x faster
- 📉 Overfitting: Better regularization
- 🎯 Accuracy: Maintained or improved

### 🐛 Bug Fixes
- Fixed notebook cell structure in v2
- Corrected forward pass in SimpleCNN
- Updated config imports

---

## [v1.0] - 2026-01-17

### Initial Implementation
- Created project structure
- Implemented data exploration (notebook 01)
- Built first SimpleCNN model (notebook 02)
- Identified overfitting issues (8.4% gap)
- Identified slow training (126 minutes)
