# Waste Classifier using FastAI

A deep learning image classifier for recycling waste categorization, built with FastAI and PyTorch.

## 🎯 Project Overview

This project uses transfer learning with ResNet architectures to classify waste into four categories:
- Crushed Aluminum Cans
- Plastic Bottle Waste
- Cardboard Box Waste
- Glass Bottle Waste

## 🚀 Features

- **Transfer Learning**: Utilizes pre-trained ResNet models (ResNet18, ResNet34, ResNet50)
- **Data Augmentation**: Enhanced training with image transformations
- **High Accuracy**: Achieves low error rates through fine-tuning
- **Easy to Use**: Simple Jupyter notebook workflow

## 📋 Requirements

See [requirements.txt](requirements.txt) for full dependencies.

Main libraries:
- fastai
- torch
- icrawler
- duckduckgo-search

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 📊 Dataset

The dataset is automatically collected using Bing Image Crawler:
- 100 images per category
- 80/20 train/validation split
- Images resized to 192x192 pixels

## 🏃 Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook Image_Classifier.ipynb
```

2. Run the cells sequentially to:
   - Install dependencies
   - Collect images
   - Train the model
   - Evaluate results

## 📈 Model Performance

The final model uses ResNet50 with:
- Mixed precision training (FP16)
- 8 epochs of fine-tuning
- Data augmentation (flips, rotations, etc.)

## 📁 Project Structure

```
Deep_Learning/
├── Image_Classifier.ipynb    # Main training notebook
├── recycling_dataset/         # Image dataset
│   ├── crushed aluminum can/
│   ├── plastic bottle waste/
│   ├── cardboard box waste/
│   └── glass bottle waste/
├── models/                    # Saved model weights
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🤝 Contributing

Feel free to open issues or submit pull requests!

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

Dan Ofri

## 🙏 Acknowledgments

- Built with [FastAI](https://www.fast.ai/)
- Dataset collected via Bing Image Search
