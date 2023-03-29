## Common Task 1. Electron/photon classification

For this task we trained a Convolutional Neural Network (CNN), Xception model sepcified [here](https://arxiv.org/pdf/1610.02357.pdf).

Link for the [PyTorch implementation](https://github.com/SarithRavI/ML4SCI-GSoC-Tests/blob/test/Task_1/Task-1-pytorch-Xception.ipynb) 
Link for the [tensorflow/keras implementation](https://github.com/SarithRavI/ML4SCI-GSoC-Tests/blob/test/Task_1/Task-1-keras-Xception.ipynb)

It's worth noting the keras implementation is same as the pytorch implementation of Xception model, except for 2 alterations: 
- In case of keras model we output only one scalar probability. In case of pytorch implementation we output two scalar probabilities (softmax). 
This small difference is due to unavailability of multi class AUC metric in tf/keras ([link to issue](https://github.com/tensorflow/addons/issues/265)).
- Given the size of the dataset, we found that it is easier to preprocess images while in the training loop rather than preprocessing beforehand training.
Hence there are two additional layers in keras model to `resize` and `standardize` the image batch.

### 1. Training 

- we split the dataset into train: validation: test with ratio of `70%: 20%: 10`.
- In `pytorch model` we observered that the model overfits heavily after the `epoch 21`. Given that we stopped training at `epoch 23`.
- In both models, `optimizer : Adam`, `learning rate : 1e-3`, `batch size : 64`.

**_NOTE: We didn't run the keras model fully, since it is the same of pytorch model. For the purpose of demonstrating that keras model works, we run the 1st epoch._**

### 2. Performance

- In addition to Xception model, we trained `ResNet-18` & `ResNet-34`. Xception model outperformed others and the training time was also much less.

Xception model obtained AUC (of ROC) of :
- `0.833` in `train set`
- `0.789` in `test set`
- `0.82` in `entire dataset`
