

**![Customized Residual U-Network](https://github.com/user-attachments/assets/f06ea3db-6ddf-42df-80bc-c297bdf24c4b)


The Residual U-Net (ResUNet) architecture in the provided code is a sophisticated deep learning model tailored for image segmentation, enhancing the traditional U-Net by incorporating residual learning principles to improve training stability and performance. The architecture processes 256x256x1 grayscale input images and is structured into three main components: an encoder, a bridge, and a decoder, with residual connections integrated throughout to address vanishing gradients and facilitate learning identity functions.
Encoder
The encoder begins with an input layer accepting 256x256x1 images, followed by a stem block that applies a 3x3 convolution with 16 filters, batch normalization (BN), and ReLU activation, complemented by a 1x1 convolutional shortcut (without activation) to form a residual connection via an element-wise addition. This is followed by four residual blocks, each downsampling the feature maps through strided convolutions (stride=2), progressively increasing the filter sizes to [32, 64, 128, 256]. Each residual block comprises two convolutional sub-blocks: the first applies BN, ReLU, and a 3x3 convolution, while the second repeats this sequence without downsampling (stride=1). A 1x1 convolutional shortcut matches the filter count and spatial dimensions, followed by BN (no activation), and is added to the main path output, preserving information flow and enhancing gradient propagation.
Bridge
The bridge, connecting the encoder and decoder, processes the deepest feature maps (256 filters) from the encoder’s final residual block. It consists of two convolutional blocks, each applying BN, ReLU, and a 3x3 convolution with 256 filters, maintaining the spatial dimensions (strides=1). This bridge refines high-level features before upsampling, ensuring robust feature representation for the decoder.
Decoder
The decoder mirrors the encoder, upsampling feature maps to recover spatial resolution while integrating skip connections from corresponding encoder layers to preserve fine-grained details. It comprises four upsample-concatenate-residual blocks. Each block starts with a 2x2 upsampling layer, followed by concatenation with the encoder’s feature maps at the same resolution (e.g., e4 for the first decoder block). The concatenated features are processed by a residual block, identical in structure to the encoder’s (two 3x3 convolutions with BN and ReLU, plus a 1x1 shortcut). Filter sizes decrease symmetrically ([256, 128, 64, 32]) to align with the encoder’s hierarchy. The final decoder output is passed through a 1x1 convolution with a sigmoid activation to produce a single-channel segmentation mask (256x256x1), representing pixel-wise probabilities.
Key Features and Mechanisms

Residual Connections: Each residual block includes a shortcut path (1x1 convolution) added to the main path, enabling the model to learn residual functions (F(x) + x), which mitigates degradation problems and enhances training of deep networks.
Batch Normalization and ReLU: Applied after each convolution to stabilize and accelerate training, with ReLU introducing non-linearity.
Skip Connections: Concatenation in the decoder integrates encoder features, preserving spatial information lost during downsampling.
Loss and Metrics: The model is compiled with a combined binary cross-entropy and Dice loss (bce_dice_loss) to balance pixel-wise accuracy and region overlap, evaluated using Dice and IoU coefficients for segmentation quality.

Training Configuration
The model is trained on preprocessed 256x256x1 images and masks (normalized to [0,1]), using the Adam optimizer, a batch size of 32, and up to 60 epochs. Callbacks like EarlyStopping, ModelCheckpoint, and a custom ShowProgress (with GradCAM visualization) monitor performance, while a learning rate scheduler reduces the learning rate by 0.1 every 30 epochs to refine convergence.
This architecture’s integration of residual learning with U-Net’s encoder-decoder framework makes it particularly effective for segmentation tasks requiring precise boundary detection, leveraging both low-level and high-level features while maintaining training stability.
