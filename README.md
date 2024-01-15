**Repository Description:**

Welcome to the RBM Feature Extraction for MNIST project! This repository showcases the application of a Restricted Boltzmann Machine (RBM) for feature extraction from the MNIST dataset. MNIST is a collection of 28x28 pixel grayscale images of handwritten digits (0 through 9). The RBM is employed to transform raw pixel values into a compact set of features, enhancing the dataset for subsequent machine learning tasks.

**How it Works:**

1. **MNIST Dataset Loading:**
   - The MNIST dataset is fetched using scikit-learn's `fetch_openml` function, providing pixel values (`data`) and corresponding digit labels (`target`).

2. **Data Scaling:**
   - Pixel values are normalized between 0 and 1 using `MinMaxScaler`. This preprocessing step is crucial for neural network models.

3. **Train-Test Split:**
   - The dataset is split into training and testing sets (80% for training, 20% for testing) using `train_test_split`.

4. **RBM Model Training:**
   - An RBM model is instantiated and trained on the training data. The model aims to extract meaningful features from the raw pixel values.

5. **Feature Transformation:**
   - The trained RBM is utilized to transform both training and testing data into a reduced set of features (`train_features` and `test_features`).

**Prerequisites:**

Ensure you have the required libraries installed:

- Python 3.x
- scikit-learn
- NumPy

