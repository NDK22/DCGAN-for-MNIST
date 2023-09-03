# DCGAN-for-MNIST
This repository contains the implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) for generating handwritten digit images using the MNIST dataset.

Description
DCGANs consist of two neural networks, a generator, and a discriminator, that are trained simultaneously through adversarial learning. The generator learns to create realistic images, while the discriminator learns to distinguish between real and fake images. This project demonstrates the training and generation of digit images using DCGAN.

Prerequisites
Make sure you have the following prerequisites installed:

Python 3
TensorFlow
Keras
NumPy
Matplotlib
Usage

The script will train the DCGAN on the MNIST dataset for 50 epochs. Generated images will be saved in the "generated_images" directory, and the generator model will be saved as "generator.h5" in the project folder.

You can visualize the generated images and real MNIST images using the saved images and the "real_images.png" and "generated_images.png" files.

Acknowledgments
This project is based on the DCGAN architecture introduced by Radford et al. in their paper "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks."
Feel free to explore the code and experiment with different settings to generate your own digit images!
