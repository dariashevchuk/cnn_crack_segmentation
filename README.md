## CNN Crack Segmentation

**Done by Dariia Shevchuk and Maryia Pisaryk**.

**Goal**: The primary objective is to train a fully convolutional neural network to perform binary semantic segmentation. The model takes images of surfaces (concrete or walls) as input and outputs a binary mask identifying specific pixels where cracks are present.

**Tech Stack**: Python, PyTorch (core training loop, Dataset/DataLoader, model architecture), Albumentations (Used for heavy data augmentation pipelines), OpenCV (Used for image reading and color conversion), pandas (data handling), scikit-learn (for train/validation splitting), matplotlib (visualization).

## Approaches and Algorithms
1. **Model Architecture**: U-NetThe project implements a U-Net architecture from scratch (rather than importing a pre-trained one).
- Structure: It follows the standard Encoder-Decoder pattern with skip connections.
- Contracting Path (Encoder): Uses "DoubleConv" blocks (Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU) followed by MaxPooling to extract features.
- Expanding Path (Decoder): Uses Transpose Convolutions for upsampling and concatenates features from the encoder (skip connections) to preserve spatial resolution.
- Configuration: The model uses a base channel size of 16 (lightweight) and an input image size of 256x256.

2. **Training Strategy**
- Loss Function: A hybrid loss function designed to handle class imbalance (cracks occupy very few pixels compared to the background):$$Loss = 0.5 \times \text{BCEWithLogitsLoss} + 0.5 \times \text{SoftDiceLoss}$$
- Optimizer: AdamW with a learning rate of 3e-4.
- Scheduler: ReduceLROnPlateau (reduces learning rate if validation IoU stops improving).
- Mixed Precision: Utilizes torch.cuda.amp for faster training and lower memory usage on GPUs.

3. **Data Pipeline**
- Augmentation: To prevent overfitting, the training pipeline applies: 
- Geometric transformations: Flip (Horizontal/Vertical), Shift, Scale, Rotate.
- Pixel-level transformations: Random Brightness/Contrast, Gaussian Blur.
- Normalization: Images are normalized using standard ImageNet mean and standard deviation.

## Results
- Metric: The model is evaluated using IoU (Intersection over Union) and Dice Coefficient.
- Kaggle Score: The notebook explicitly states a submission score of 0.600424 on the test set.
- Submission Format: The inference results are converted into Run-Length Encoding (RLE) format for the final CSV submission.