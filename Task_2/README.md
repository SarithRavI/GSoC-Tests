## Common Task 2.  Deep Learning based Quark-Gluon Classification

For this task we trained a Convolutional Neural Network (CNN), Xception model sepcified [here](https://arxiv.org/pdf/1610.02357.pdf).

### 1. Training 

**_NOTE: We created image dataset only with raw file identified by `QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272` and all 36272 images of this have been used in training._**

- we split the dataset into train: validation: test with ratio of `70%: 20%: 10%`.
- Earlystopping patience is `3` for monotonic increase in validation loss.
- In `pytorch model` we observered that the model overfits heavily after the `epoch 6`. Given that we stopped training at `epoch 9`.
- In both models, `optimizer : Adam`, `learning rate : 1e-3`, `batch size : 64`.

### 2. Performance

Xception model obtained AUC (of ROC) of :
- `0.806` in `train set`
- `0.769` in `test set`
- `0.8` in `entire dataset`
