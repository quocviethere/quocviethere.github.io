# Masked Image Training for Generalizable Deep Image Denoising

Tags: Computer Vision, Masked Image Modelling

Authors: Chen et al.

Code: <https://github.com/haoyuc/MaskedDenoising>

Conferences/Journal: CVPR2023

Last edited time: September 5, 2023 8:16 PM

# Motivation

Existing denoising models train and evaluate on images corrupted with Gaussian noise → limit the model performance to only one distribution → performance drops when being applied to remove noise from other distribution.

Most existing methods perform denoising by overfitting the training noise.

# Main contribution

Making deep models that denoise image more generalizable by using masked training method (instead of conventional models that are trained on Gaussian noise only).

-   Mask random pixels of the input image
-   Reconstruct the missing information during training
-   Masked out the features in the self-attention layers to avoid training-testing inconsistency

→ learn image content reconstruction instead of just overfitting to the noise.

**Masking mechanisms:**

1.  Input mask
2.  Attention mask

**Shifted window mechanism**

![Proposed Pipeline](Masked%20Image%20Training%20for%20Generalizable%20Deep%20Image%206cae52c83db54cb88074bd477e800bb2/Screen_Shot_2023-09-05_at_19.58.08.png)

**Training procedure:**

-   Transformers divide the input signal into tokens and process spatial information using self-attention layers
-   Project the image into $$C$$-dimensional feature tokens using a convolutional layer with kernel size of 1.

![Effect of Input Mask and Attention Mask](Masked%20Image%20Training%20for%20Generalizable%20Deep%20Image%206cae52c83db54cb88074bd477e800bb2/Screen_Shot_2023-09-05_at_20.02.03.png)

The input mask randomly masks out the feature tokens embedded by the first convolution layer and encourages the network to complete the masked information during training. Concretely, given the feature token tensor $$\mathbf{f} \in \mathbb{R}^{H \times W \times C}$$, the features tokens are randomly replaced with the `[mask token]` with probability $$p_{IM}$$ (mask ratio).

# Experiments

Data: for clean images: DIV2K, Flickr2K, BSD500, WED

![Results](Masked%20Image%20Training%20for%20Generalizable%20Deep%20Image%206cae52c83db54cb88074bd477e800bb2/Screen_Shot_2023-09-05_at_20.09.09.png)

# Further Reading

[SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/pdf/2108.10257.pdf)

[Restormer: Efficient Transformer for High-Resolution Image Restoration](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf)
