# Salient Object Segmentation with U-Net

This project implements and trains a U-Net model for salient object segmentation. The model is built from scratch using TensorFlow/Keras and is trained on the DUTS dataset to identify and create binary masks for the most prominent object in an image.

## Project Overview

The notebook covers the complete workflow for a deep learning segmentation task:
1.  **Data Loading:** Loading and preprocessing image and mask data from the DUTS dataset.
2.  **Model Architecture:** Defining a U-Net model from scratch.
3.  **Training:** Compiling and training the model with callbacks for learning rate reduction, checkpointing, and early stopping.
4.  **Evaluation:** Evaluating the model on the test set using Precision and Recall metrics.
5.  **Visualization:** Plotting the original images, ground truth masks, and predicted masks to visually assess performance.

## Dataset

The model is trained and evaluated on the **DUTS (Duties-Testing and DUTS-Training)** dataset.
* **Training Data:** 10,553 images and masks from `DUTS-TR`.
* **Test Data:** 5,019 images and masks from `DUTS-TE`.

All images are resized to 256x256. Input images are RGB (256, 256, 3), and masks are grayscale (256, 256, 1) and normalized to a [0, 1] range.

## Model Architecture

The model is a **U-Net**, constructed in TensorFlow/Keras with an input shape of (256, 256, 3).

* **Convolutional Block (`convBlock`):** The basic building block consists of two sequential 3x3 Conv2D layers, each followed by Batch Normalization and a ReLU activation.
* **Encoder (`encoderBlock`):** Consists of a `convBlock` followed by a 2x2 Max Pooling layer. The output from the `convBlock` is passed as a skip connection to the decoder. The model uses 4 encoder blocks, with filters increasing (64, 128, 256, 512).
* **Bottleneck:** A standard `convBlock` with 1024 filters at the base of the U-Net.
* **Decoder (`decoderBlock`):** Consists of a 2x2 UpSampling2D layer (using bilinear interpolation), which is then concatenated with the corresponding skip connection from the encoder. This is followed by a `convBlock`. The filters decrease at each block (1024 -> 512 -> 256 -> 128 -> 64).
* **Output Layer:** A final 1x1 Conv2D layer with a **sigmoid** activation produces the 256x256x1 probability mask.

## Training & Evaluation

The model was compiled and trained with the following configuration:

* **Optimizer:** Adam (learning rate = 1e-4)
* **Loss Function:** Binary Cross-Entropy (`binary_crossentropy`)
* **Metrics:** Custom Precision and Recall
* **Batch Size:** 8
* **Epochs:** 15 (with early stopping)
* **Callbacks:**
    * `ReduceLROnPlateau`: Monitors `val_loss` (patience=3, factor=0.1).
    * `ModelCheckpoint`: Saves the best model weights (`best.weights.h5`) based on `val_loss`.
    * `EarlyStopping`: Monitors `val_loss` (patience=5) and restores the best weights.

### Results

The training stopped after 14 epochs, restoring the weights from **Epoch 9** as the best.

**Best Validation Metrics (Epoch 9):**
* **Validation Loss:** 0.2145
* **Validation Precision:** 0.6778
* **Validation Recall:** 0.7560

**Final Performance on Test Set:**
* **Test Precision:** 0.6830
* **Test Recall:** 0.7493

## Key Dependencies

* `tensorflow` (keras)
* `numpy`
* `pandas`
* `matplotlib`
* `Pillow (PIL)`
* `scikit-learn`
* `tqdm`
* `visualkeras`