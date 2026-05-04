# 🖼️ CIFAR-10 Object Recognition using ResNet50

This project uses **Deep Learning** and **Transfer Learning** with **ResNet50** to classify images into one of the 10 CIFAR-10 object categories.

---

## 📌 Objective

The objective of this project is to build an image classification model that can recognize objects from the CIFAR-10 dataset using a pretrained ResNet50 model.

The model classifies images into the following classes:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

---

## 📊 Dataset

The project uses the **CIFAR-10** dataset, which contains 60,000 color images across 10 classes.

* **Training images:** 50,000
* **Test images:** 10,000
* **Image size:** 32 × 32 pixels
* **Number of classes:** 10

---

## ⚙️ Project Workflow

1. Dataset loading
2. Image preprocessing
3. Data normalization
4. Train-test split
5. Baseline neural network training
6. Transfer learning using ResNet50
7. Model training
8. Model evaluation
9. Classification report
10. Confusion matrix
11. Single image prediction
12. Model saving
13. Streamlit app development

---

## 🧠 Model Used

### ResNet50 Transfer Learning

* Base model: **ResNet50**
* Pretrained weights: **ImageNet**
* Custom classification layers added for CIFAR-10
* Output layer: **10 classes with softmax activation**

---

## 📈 Evaluation

The model was evaluated using:

* Test accuracy
* Classification report
* Confusion matrix
* Single image prediction

---

## 🖥️ Streamlit App

A Streamlit web app is included for image classification.

### Features

* Upload an image
* Display uploaded image
* Predict CIFAR-10 class
* Show confidence score
* Display class probability chart

Run the app:

```bash
streamlit run app.py
```
---

## ⚠️ Important Note

This model is trained on CIFAR-10 style images. It works best with simple images that resemble CIFAR-10 categories.

For real-world images, predictions may be less accurate because CIFAR-10 images are small 32×32 images.

---


