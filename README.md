# CIFAR-10 Image Classifier (Streamlit Web App)

A web-based image classification application built with **Streamlit** and **TensorFlow**, using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset.

---

## Purpose / Goal

Demonstrate how to deploy a simple image classification model interactively and responsibly using Streamlit.

---

## Project Overview

This project is a simple machine learning web application that allows users to upload an image and receive a predicted class label.

The model classifies images into one of the following 10 categories:

* **Airplane**
* **Automobile**
* **Bird**
* **Cat**
* **Deer**
* **Dog**
* **Frog**
* **Horse**
* **Ship**
* **Truck**

The application demonstrates how to integrate a trained deep learning model into an interactive web interface.

---

## About the Model

The model was trained on the **CIFAR-10 dataset**, which contains:

* 60,000 images
* 32×32 pixel resolution
* 10 object classes

### Important Note

CIFAR-10 images are small (32×32) and contain clean, single-object scenes.

This model performs best on images similar to the dataset.
Real-world photos (e.g., multiple objects, complex backgrounds, text, or human portraits) may be misclassified.

To improve reliability, a confidence threshold is implemented. If the model is not confident enough, the app displays a warning instead of returning an unreliable prediction.

---

## Features

* Image upload functionality using Streamlit
* Automatic resizing to 32×32 pixels
* Image normalization before prediction
* Confidence-based filtering to prevent unreliable outputs
* Clean and user-friendly interface

---

## Installation & Setup

Clone the repository:

```bash
git clone <your-repository-link>
cd imageclassification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app/streamlitapp.py
```

---

## Project Structure

```
imageclassification/
│
├── app/
│   └── streamlitapp.py
│
├── model/
│   └── cifar10model.h5
│
├── requirements.txt
└── README.md
```

---

## Learning Objectives

This project demonstrates:

* Image preprocessing (resizing and normalization)
* Loading and using trained deep learning models
* Implementing prediction confidence checks
* Building interactive machine learning applications with Streamlit
* Understanding model limitations and responsible AI presentation

---

## Technologies Used

* Python
* Streamlit
* TensorFlow / Keras
* NumPy
* Pillow