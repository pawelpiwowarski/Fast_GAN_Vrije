# Fast GAN Architecture with Additional Decoders in Discriminator

This is an implementation of the Fast GAN architecture with additional decoders in the discriminator part, as described in the paper [Fast GAN: High Fidelity Synthesis with Lightweight Architecture](https://arxiv.org/abs/2101.04775). This project was developed as part of the Machine Learning course at Vrije Universiteit Amsterdam, by group 18.

The Fast GAN architecture is a lightweight GAN model that can achieve high-quality image synthesis with fewer parameters and computation time compared to traditional GAN models. The additional decoders in the discriminator part further improve the model's performance by helping it to better distinguish between real and fake images.

## Requirements

- Python
- TensorFlow > 2.9.0
- TensorFlow Addons 0.19.0
- typeguard == 2.13.3

## Other Software Used

- LPIPS metric: https://github.com/moono/lpips-tf2.x (based on https://arxiv.org/abs/1801.03924)
- Differential Augmentation: https://github.com/mit-han-lab/data-efficient-gans


## Dataset 

An example dataset with 752 oil paintings by Vincent Van Gogh is provided. 
## Training

It takes around 7 hours to train on Google Colab.
