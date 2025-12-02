# Multiclass-Fish-Image-Classification
Multiclass Fish Image Classification using deep learning. This project trains a CNN and fine-tunes multiple pre-trained models to classify fish into various categories.
It saves the best model and includes a Streamlit app for real-time fish image prediction.

1.Problem Statement:
This project focuses on classifying fish images into multiple categories using deep learning models.
The task involves training a CNN from scratch and leveraging transfer learning with pre-trained models to enhance performance.
The project also includes saving models for later use and deploying a Streamlit application to predict fish categories from user-uploaded images.

2.Business Use Cases:
Enhanced Accuracy: Determine the best model architecture for fish image classification.
Deployment Ready: Create a user-friendly web application for real-time predictions.
Model Comparison: Evaluate and compare metrics across models to select the most suitable approach for the task.

3.Approach:
Data Preprocessing and Augmentation

Rescale images to [0, 1] range.
Apply data augmentation techniques like rotation,
zoom, 
and flipping to enhance model robustness.

Model Training
Train a CNN model from scratch.
Experiment with five pre-trained models
(e.g., VGG16,
ResNet50,
MobileNet,
InceptionV3, 
EfficientNetB0).
Fine-tune the pre-trained models on the fish dataset.
Save the trained model (max accuracy model ) in .h5 or .pkl format for future use.

Model Evaluation
Compare metrics such as
accuracy,
precision, 
recall,
F1-score,
and confusion matrix across all models.
Visualize training history (accuracy and loss) for each model.

Deployment
Build a Streamlit application to:
Allow users to upload fish images.
Predict and display the fish category.
Provide model confidence scores.

4.Dataset
The dataset consists of images of fish, categorized into folders by species.
The dataset is loaded using TensorFlow's ImageDataGenerator for efficient processing.

5. Model Architecture
CNN from Scratch
•	Input
 → Conv2D (32 filters) → MaxPooling → Conv2D (64 filters) → MaxPooling →
 Conv2D (128 filters) → MaxPooling → Flatten → Dense(128) → Dropout
 → Dense(num_classes, softmax)Transfer Learning

•	Base pre-trained model (VGG16, ResNet50, etc.)
 → GlobalAveragePooling → Dense(256) → Dropout → Dense(num_classes, softmax)

7. Evaluation Metrics
Metric	Description
Accuracy	Correct predictions / Total predictions
Precision	TP / (TP + FP)
Recall	TP / (TP + FN)
F1-score	Harmonic mean of Precision & Recall
Confusion Matrix	True vs predicted class counts

9. Results
•	The best CNN model achieved high accuracy on test set
•	Transfer learning models outperformed the CNN from scratch
•	Confusion matrices visualize misclassifications
•	Fine-tuning improved the performance of pre-trained models
(Add actual numbers, plots, and screenshots here from your experiments)

18. Folder Structure
fish_classification_project/

├─ data/

│  ├─ train/

│  ├─ val/

│  └─ test/

├─ app.py

│  ├─ best_fish_model_MobileNetV2.h5

│  ├─ best_model_info.pkl

├─ fish classification.ipynb

├─ README.md

└─ image classification.doc

9. Deliverables
•	Python scripts:
o	train_and_eval.ipynb — train and evaluate models
o	app.py — Streamlit web application
•	Trained models:
o h5 and pkl files for each model
•	Training history plots and confusion matrices

11. Tools & Libraries
•	Python 3.x
•	TensorFlow / Keras
•	NumPy, Matplotlib, Seaborn, Scikit-learn
•	Streamlit

12. Future Work
•	Add more fish species for higher diversity
•	Hyperparameter tuning using libraries like Keras Tuner
•	Integrate with a mobile app or web API for broader access
•	Experiment with ensemble models for even better accuracy


