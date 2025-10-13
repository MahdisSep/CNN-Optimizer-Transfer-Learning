# 🚀 Deep Learning for Image Classification: CNN Architecture and Optimization Study

## 📝 Overview
This project presents a comprehensive study on deep learning techniques for image classification, completed as the main assignment for the Computer Vision course. The work focuses on designing and training Convolutional Neural Networks (CNNs) from scratch, evaluating the impact of different optimization algorithms, and utilizing **Transfer Learning** with powerful pre-trained models.

The primary dataset used is a **3-class image dataset** (e.g., classifying different types of footwear).

## 🛠️ Technologies Used
* **Framework:** TensorFlow / Keras
* **Language:** Python
* **Libraries:** `MobileNetV2`, `ResNet50V2`, `ImageDataGenerator`, `Adam`, `SGD`, `RMSprop`, `Matplotlib`.
* **Environment:** Jupyter Notebooks (Colab) / Python scripts.

## 🎯 Project Objectives

1.  **Custom CNN Design:** Develop and train a sequential CNN model with multiple Conv2D and MaxPooling blocks.
2.  **Optimizer Comparison:** Conduct an empirical study on the effect of different optimization algorithms (`Adam`, `SGD`, `RMSprop`) on training speed, stability, and final accuracy.
3.  **Data Augmentation:** Implement real-time data augmentation (rotation, shifting, zooming, brightness adjustment) using `ImageDataGenerator` to improve model generalization and prevent overfitting.
4.  **Transfer Learning (Bonus):** Fine-tune and evaluate a state-of-the-art pre-trained architecture (MobileNetV2 or ResNet50V2) to achieve superior performance with minimal training time.

## 🧠 Model Architecture and Components

### 1. Custom CNN Model (Implemented in `Q1.ipynb` / `q1.py`)
The custom model is a shallow, yet effective, sequential CNN designed for feature extraction:
* **Architecture:** Four blocks, each consisting of a `Conv2D` layer followed by a `MaxPooling2D` layer.
* **Regularization:** Utilized `BatchNormalization` and `Dropout` layers to stabilize training and combat overfitting.
* **Output:** A final `Dense` layer with `softmax` activation for 3-class classification.

### 2. Transfer Learning Model (Implemented in `Q1_bonus.ipynb`)
For enhanced performance, a Transfer Learning approach was used:
* **Base Model:** **MobileNetV2** (or **ResNet50V2**) was loaded with pre-trained ImageNet weights.
* **Fine-Tuning:** The weights of the base model were frozen initially, and new classification layers were added and trained. In a subsequent step, a few top convolutional layers of the base model were unfrozen for fine-tuning.

## 📈 Key Results and Findings

| Experiment | Best Optimizer | Best Validation Accuracy | Key Conclusion |
| :--- | :--- | :--- | :--- |
| **Custom CNN** | **Adam** (typically) | Approx. 80-85% | Demonstrates proficiency in building and training models from scratch. |
| **Transfer Learning (MobileNetV2/ResNet50V2)** | Adam | **Approx. 95%+** | Confirms the superior performance and efficiency of using pre-trained weights for feature extraction in image tasks. |

> **Conclusion on Optimizers:** The study showed that the **Adam** optimizer provided the fastest convergence and generally achieved better final validation accuracy compared to `SGD` and `RMSprop` in this specific setup, due to its adaptive learning rate capabilities.

## 📂 Repository Structure

```

├── CNN-Optimizer-Transfer-Learning/
│   ├── README.md                 \<-- You are here
│   ├── Q1.ipynb                  \<-- Custom CNN implementation and Optimizer comparison
│   ├── q1.py                     \<-- Python script version of the custom CNN model
│   ├── Q1\_bonus.ipynb            \<-- Transfer Learning implementation (MobileNetV2/ResNet50V2)
│   └── dataset/                  \<-- Placeholder for dataset (e.g., 'train', 'test' folders)

````

## ⚙️ How to Run

1.  **Clone the Repository:**
    ```bash
    git clone [Your Repository URL]
    cd CNN-Optimizer-Transfer-Learning
    ```
2.  **Setup Data:** The dataset (e.g., `shoes.zip`) must be placed in the root directory and extracted into the expected `train` and `test` folder structure, as handled by the `ImageDataGenerator` setup in the notebooks.
3.  **Install Dependencies:**
    ```bash
    pip install tensorflow keras numpy matplotlib
    ```
4.  **Execute:** Run the Jupyter Notebooks (`Q1.ipynb` and `Q1_bonus.ipynb`) sequentially to replicate the experiments and view the visualizations of optimizer performance and model training history.

