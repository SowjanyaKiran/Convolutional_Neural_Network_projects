# 🚦 German Traffic Sign Recognition using CNN

This project implements a Convolutional Neural Network (CNN) to classify German Traffic Signs using the GTSRB dataset.
The dataset is loaded from KaggleHub and trained with TensorFlow/Keras, with experiments on different hidden layer configurations.

# 📂 Dataset

Name: GTSRB - German Traffic Sign Recognition Benchmark

Classes: 43 traffic sign categories

Training: 31,368

Validation: 7,841 (split from training data)

Test: Provided separately

# ⚙️ Requirements

Install required libraries before running:

pip install kagglehub tensorflow matplotlib seaborn pandas

# 🚀 Project Workflow

1. Data Loading

Dataset downloaded via kagglehub

Train/Test paths defined

import kagglehub
path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")

2. Data Preprocessing

Images resized to 64x64

ImageDataGenerator used for:

Rescaling

Training/Validation split

Augmentations (rotation, shift, flip, zoom)

3. Data Augmentation Examples

The project visualizes different augmentations:

Original

Rotation

Shifting

Flipping

Zooming

4. CNN Model Architecture
Conv2D(32, (3,3), activation='relu')
MaxPooling2D(2,2)

Conv2D(64, (3,3), activation='relu')
MaxPooling2D(2,2)

Flatten()

(Dense + Dropout layers depending on config)

Dense(43, activation='softmax')


Optimizer: Adam

Loss: Categorical Crossentropy

Metrics: Accuracy

5. Model Configurations Tested
Hidden Layers	Neurons	Patience	Validation Accuracy
1	64	2	88.17%
2	128	2	92.48%
3	256	2	92.09%

# 📊 Best result achieved: 2 hidden layers, 128 neurons → 92.48% accuracy

6. Training Results Visualization

Validation accuracy plotted across different model configurations:

plt.plot(results_df['val_acc'], marker='o')

# 📈 Key Insights

Adding more hidden layers initially improves accuracy, but too many layers may lead to overfitting.

Best trade-off: 2 hidden layers with 128 neurons.

Data augmentation significantly boosts generalization.

# 🔑 How to Run

Open in Google Colab or Jupyter Notebook

Install requirements

Run cells step by step:

Download dataset

Data preprocessing

Train CNN models

Compare results

# 🎯 Future Improvements

Use transfer learning (ResNet, MobileNet, etc.)

Hyperparameter tuning (batch size, learning rate, dropout rates)

Evaluate on the test set and generate classification reports
