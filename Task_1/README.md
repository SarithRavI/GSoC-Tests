## Common Task 1. Electron/photon classification

**_SPECIAL NOTE:_** 
**_Due to a thermal issue in my personal machine it tends to turn off unexpectedly while training. After each epoch we save the model weights as a checkpoint So we can continue the training from that last checkpoint if the machine turned off amid training. Please note that's why there are notebook outputs with epochs starting from where it left off (you can see the logic in cell `In [15] & [16]`)._** 

**_If you want to view the training outcomes, please refer to following cells:_**
- **_training Xception model: `In [17]`_**

**_Click [here](https://tensorboard.dev/experiment/TjfMWJJNQkmSXQoc4n92wQ/) to view the Tensorboard._**

### 1. Introduction
For this task we trained a Convolutional Neural Network (CNN), Xception model sepcified [here](https://arxiv.org/pdf/1610.02357.pdf).

Our model achieves the required ROC-AUC score of `0.8` on test dataset.

Link for the [PyTorch implementation](https://github.com/SarithRavI/GSoC-Tests/blob/master/Project_CMS/Task_1/Task-1-pytorch-Xception.ipynb) 
Link for the [tensorflow/keras implementation](https://github.com/SarithRavI/GSoC-Tests/blob/master/Project_CMS/Task_1/Task-1-keras-Xception.ipynb)

It's worth noting the keras implementation is same as the pytorch implementation of Xception model, except for 2 alterations: 
- In case of keras model we output only one scalar probability. In case of pytorch implementation we output two scalar probabilities (softmax). 
This small difference is due to unavailability of multi class AUC metric in tf/keras ([link to issue](https://github.com/tensorflow/addons/issues/265)).
- Given the size of the dataset, we found that it is easier to preprocess images while in the training loop rather than preprocessing beforehand training.
Hence there are two additional layers in keras model to `resize` and `standardize` the image batch.

### 2. Training 

- we split the dataset into train: validation: test with ratio of `70%: 20%: 10%`.
- In both models, `optimizer : Adam`, `learning rate : 1e-3`, `batch size : 64`.

**_NOTE: We didn't run the keras model fully, since it is the same of pytorch model. For the purpose of demonstrating that keras model works, we run it for few epochs._**

### 3. Performance

- In addition to Xception model, we trained `ResNet-18` & `ResNet-34`. Xception model outperformed others and the training time was also much less.

**_Test data set was evaluated on the model weights produced at the epoch that gives heighest ROC-AUC score for validation data._**

- In epoch `17` we obtain the heighest ROC-AUC score on validation data.

Xception model obtained AUC (of ROC) of :
- `0.82` in `train set`
- `0.799` ~ `0.8` in `test set`
   

