# README #

### What is this repository for? ###

This repository contains code related to the Independent Study Option Module on Private Inference.

The PySyft code is based on the following tutorial:
https://github.com/OpenMined/PySyft/blob/master/examples/tutorials/Part%2011%20-%20Secure%20Deep%20Learning%20Classification.ipynb

The CrypTen code is based on  the following tutorial:
https://github.com/facebookresearch/CrypTen/blob/master/tutorials/Tutorial_4_Classification_with_Encrypted_Neural_Networks.ipynb

### How do I get set up? ###

Install the necessary dependencies in the requirements file. 

The repository contains experiments on the MNIST dataset and the Malaria dataset.

The main.py file contains the code required to run the mnist and malaria experiments using both PySyft and CrypTen. 
By default it runs all mnist and malaria experiments without retraining the model. If you wish to retrain the model 
invoke the method using should_retrain_model=True.

If running from the command line ```python3 main.py``` from the root of the project. 