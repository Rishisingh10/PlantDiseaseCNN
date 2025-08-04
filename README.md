# Plant Disease Detection using CNN

This project uses a Convolutional Neural Network (CNN) built from scratch to detect 38 types of plant diseases using image data. The model processes leaf images through multiple convolutional layers to extract patterns, shapes, and color features.

## 📁 Dataset
- Source: [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- 38 disease classes
- Train/Validation split used

## 🧠 Model Highlights
- Custom CNN built manually using TensorFlow
- Five convolutional blocks with increasing filters (32 → 512)
- Dropout layers to prevent overfitting
- Final dense layer with softmax activation for multi-class classification

## ⚙️ Training
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Batch Size: 32
- Epochs: 10
- Checkpoint callback to save best model

## 📈 Results
- Accuracy and loss tracked per epoch
- Training and validation accuracy plotted
- Model saved as `best_model.h5`

## 🔍 Why Custom CNN?
While pretrained models like VGG, ResNet, or MobileNet are faster and more powerful due to transfer learning, building a model manually helps understand deep learning fundamentals, layer tuning, and architecture design.

## 🚀 Future Work
- Compare performance with pretrained models (e.g., ResNet50, MobileNetV2)
- Deploy model with a simple frontend interface
- Optimize model for mobile or edge devices

## 📦 Requirements
- TensorFlow
- NumPy, Matplotlib, Seaborn
- (Optional) scikit-learn for evaluation metrics



