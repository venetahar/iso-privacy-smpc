# README #

### What is this repository for? ###

* Quick summary

This repository contains code related to the Independent Study Option Module on Private Inference.

The PySyft code is based on the following tutorial:
https://github.com/OpenMined/PySyft/blob/master/examples/tutorials/Part%2011%20-%20Secure%20Deep%20Learning%20Classification.ipynb

The CrypTen code is based on  the following tutorial:
https://github.com/facebookresearch/CrypTen/blob/master/tutorials/Tutorial_4_Classification_with_Encrypted_Neural_Networks.ipynb

### How do I get set up? ###

Install the necessary dependencies in the requirements file. 

The repository contains experiments on the MNIST dataset, the Malaria dataset and 
the Diabetics dataset (these are not working correctly at the moment).

Both the MNIST and Malaria dataset private inference can be run both using CrypTen and PySyft. 
1. To run MNIST using PySyft launch the following file: mnist/pysyft/pysyft_mnist.py
2. To run MNIST using CrypTen launch the following file: mnist/cryp_ten/crypten_mnist.py
3. To run Malaria using PySyft launch the following file: malaria/pysyft/pysyft_malaria.py
4. To run Malaria using CrypTen launch the following file: malaria/crypten/crypten_malaria.py
