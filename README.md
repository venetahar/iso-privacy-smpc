# README #

### What is this repository for? ###

This repository contains code related to the Independent Study Option Module on Private Inference.

The PySyft code is based on the following tutorial:
https://github.com/OpenMined/PySyft/blob/master/examples/tutorials/Part%2011%20-%20Secure%20Deep%20Learning%20Classification.ipynb

The CrypTen code is based on the following tutorial:
https://github.com/facebookresearch/CrypTen/blob/master/tutorials/Tutorial_4_Classification_with_Encrypted_Neural_Networks.ipynb

### How do I get set up? ###

Install the necessary dependencies in the requirements file. 

The repository contains experiments on the MNIST dataset and the Malaria dataset.

The main.py file contains the code required to run the mnist and malaria experiments using both PySyft and CrypTen. 
By default it runs all mnist and malaria experiments without retraining the model. If you wish to retrain the model 
pass --retrain flag.

I.e. ```python3 main.py --experiment_name=mnist_conv --framework=pysyft --retrain``` from the root of the project. 
Different experiments can be selected to by passing the relevant experiment_name.

NOTE: In order to benchmark Pysyft, you need to edit the following library files in syft as 
per the advice of the developers in order to measure the communication cost:
File virtual.py, 
```def _recv_msg(self, message: bin) -> bin:
    if not hasattr(self, "count"):
        self.count = False
    if self.count:
        if not hasattr(self, "received_load"):
            self.received_load = 0
        message_size = len(message)
        self.received_load += message_size
    ... and of the function```:

