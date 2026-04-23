# Embryo Morphokinetic Parameter Prediction

This project implements a sequence-aware deep learning pipeline to predict the morphokinetic developmental stages of human embryos from time-lapse imaging videos. Due to the temporal and sequential nature of embryo development, the architecture relies on a feature extractor (MobileNetV2) and a sequence learning model (LSTM).

## Dataset Description

The dataset used in this project is sourced from a collection of time-lapse embryo videos and annotations (available via Zenodo). 

- **Images:** Contains raw frames of embryo development.
- **Annotations:** Contains start and end frame bounds for each developmental stage per video.

The sequence comprises **16 ordinal stages** of embryo development:
`pPB2`, `pPNa`, `pPNf`, `p2`, `p3`, `p4`, `p5`, `p6`, `p7`, `p8`, `p9+`, `pM`, `pSB`, `pB`, `pEB`, `pHB`

## Full Process and Pipeline

The project takes raw video frames and translates them into sequence predictions. The process involves multiple steps:

### 1. Feature Extraction (CNN)
Images are preprocessed and resized to `224x224`. Instead of training an image classifier from scratch, we use an ImageNet-pretrained **MobileNetV2** base accompanied by a `GlobalAveragePooling2D` layer to extract abstract, high-level features. 
- The MobileNet weights are frozen.
- Each image is converted into a compressed **1280-dimensional** feature vector.
- Generating features can take time, so extracted features are saved to a `.npz` disk cache to speed up subsequent runs.

### 2. Sequence Building and Data Splitting
Since developmental phases rely on time, predicting frames individually loses context.
- We group the continuous video features into smaller sequential chunks.
- We define a fixed sequence length (`TIMESTEPS = 6`) and a sliding step (`SLIDING_WINDOW = 15`).
- The dataset is split into **Train (70%), Validation (15%), and Test (15%)** sets. We use `GroupShuffleSplit` configured around student/patient IDs to guarantee that no video/patient leaks across the splits.

### 3. LSTM Model Training
The core model processes the constructed sequences.
- It consists of an **LSTM layer (64 units)** with `return_sequences=True`, ensuring we capture information frame-by-frame.
- A **Dropout layer (35%)** is applied for regularization.
- The output goes through a **TimeDistributed Dense layer** producing a softmax probability distribution over the 16 target stages for each frame in the sequence.
- During training, `EarlyStopping` (to prevent overfitting) and `ReduceLROnPlateau` (to help converge on the optimal weights) are applied.

### 4. Evaluation 
The final model generates predictions on the hold-out test set. We unfold the sequences and gather a frame-level classification report, a confusion matrix, and analyze the distance of our prediction mistakes.

## Custom Custom Loss: The `sequence_ordinal_loss`

A critical component of this project's success relies on understanding the nature of the data. Embryo developmental stages are strictly sequential and **ordinal**. 

In standard sparse categorical cross-entropy (CCE), predicting stage *1* as stage *2* gets penalized the same amount as predicting stage *1* as stage *16*. In reality, being slightly off is much better than being completely wrong. To teach this ordinality to the network, we introduce the `sequence_ordinal_loss`.

### How it works:
1. **Base Error:** It calculates the standard CCE loss.
2. **Expected Value Calculation:** It determines the 'average predicted stage index' by taking the dot product of softmax probabilities and the sequence of tensor indices.
3. **Distance Penalty:** It subtracts the actual target stage index from this expected prediction, and squares the difference.
4. **Weighted Combination:** The base CCE and the distance penalty are summed together (regulated by a `penalty_weight` parameter).

By doing this, the network tries to be highly accurate across classes while *heavily repelling* predictions that are temporally far from the truth.
