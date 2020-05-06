Stress Detection using  SVM and Fully Neural Network 

## Quick start

I. Guideline
1) Clone  this repository
2) Using pip3 install file requiment
```bash
pip3 install  -r requiments.txt
```
3) File svm_model.py is file source code  model and  processing data for traing model SVM
4) File venture_model.py is file source code  model and  processing data for traing model NN (Neural Network)

```
model.pkl file model for svm  : 0.91 %
model.h5  file model for nn   : 0.89 %
```
```
258502/258502 [==============================] - 3s 10us/step - loss: 8.4464 - acc: 0.4874 - val_loss: 8.8813 - val_acc: 0.4478
Epoch 2/30
258502/258502 [==============================] - 3s 11us/step - loss: 4.9564 - acc: 0.6920 - val_loss: 3.5136 - val_acc: 0.5237
Epoch 3/30
258502/258502 [==============================] - 3s 10us/step - loss: 0.5024 - acc: 0.7728 - val_loss: 0.4633 - val_acc: 0.7938
Epoch 4/30
258502/258502 [==============================] - 3s 10us/step - loss: 0.4553 - acc: 0.7971 - val_loss: 0.4372 - val_acc: 0.8069
Epoch 5/30
258502/258502 [==============================] - 2s 8us/step - loss: 0.4270 - acc: 0.8144 - val_loss: 0.4098 - val_acc: 0.8253
Epoch 6/30
258502/258502 [==============================] - 2s 8us/step - loss: 0.4031 - acc: 0.8300 - val_loss: 0.3888 - val_acc: 0.8385
Epoch 7/30
258502/258502 [==============================] - 3s 10us/step - loss: 0.3862 - acc: 0.8429 - val_loss: 0.3726 - val_acc: 0.8483
Epoch 8/30
258502/258502 [==============================] - 3s 11us/step - loss: 0.3760 - acc: 0.8507 - val_loss: 0.3651 - val_acc: 0.8546
Epoch 9/30
258502/258502 [==============================] - 3s 10us/step - loss: 0.3678 - acc: 0.8551 - val_loss: 0.3652 - val_acc: 0.8541
Epoch 10/30
258502/258502 [==============================] - 2s 8us/step - loss: 0.3641 - acc: 0.8562 - val_loss: 0.3782 - val_acc: 0.8503
Epoch 11/30
258502/258502 [==============================] - 2s 8us/step - loss: 0.3572 - acc: 0.8593 - val_loss: 0.3706 - val_acc: 0.8505
Epoch 12/30
258502/258502 [==============================] - 2s 8us/step - loss: 0.3507 - acc: 0.8624 - val_loss: 0.3425 - val_acc: 0.8642
Epoch 13/30
258502/258502 [==============================] - 3s 10us/step - loss: 0.3474 - acc: 0.8640 - val_loss: 0.3347 - val_acc: 0.8675
Epoch 14/30
258502/258502 [==============================] - 3s 10us/step - loss: 0.3451 - acc: 0.8649 - val_loss: 0.3394 - val_acc: 0.8698
Epoch 15/30
258502/258502 [==============================] - 2s 9us/step - loss: 0.3455 - acc: 0.8646 - val_loss: 0.3460 - val_acc: 0.8578
Epoch 16/30
258502/258502 [==============================] - 3s 10us/step - loss: 0.3409 - acc: 0.8671 - val_loss: 0.3346 - val_acc: 0.8689
Epoch 17/30
258502/258502 [==============================] - 2s 9us/step - loss: 0.3388 - acc: 0.8685 - val_loss: 0.3248 - val_acc: 0.8759
Epoch 18/30
258502/258502 [==============================] - 3s 11us/step - loss: 0.3372 - acc: 0.8696 - val_loss: 0.3379 - val_acc: 0.8688
Epoch 19/30
258502/258502 [==============================] - 2s 10us/step - loss: 0.3345 - acc: 0.8706 - val_loss: 0.3461 - val_acc: 0.8631
Epoch 20/30
258502/258502 [==============================] - 3s 10us/step - loss: 0.3346 - acc: 0.8707 - val_loss: 0.3733 - val_acc: 0.8524
Epoch 21/30
258502/258502 [==============================] - 3s 10us/step - loss: 0.3358 - acc: 0.8695 - val_loss: 0.3148 - val_acc: 0.8836
Epoch 22/30
258502/258502 [==============================] - 3s 10us/step - loss: 0.3339 - acc: 0.8705 - val_loss: 0.3232 - val_acc: 0.8776
Epoch 23/30
258502/258502 [==============================] - 3s 10us/step - loss: 0.3322 - acc: 0.8713 - val_loss: 0.3735 - val_acc: 0.8463
Epoch 24/30
258502/258502 [==============================] - 3s 11us/step - loss: 0.3321 - acc: 0.8712 - val_loss: 0.3249 - val_acc: 0.8739
Epoch 25/30
258502/258502 [==============================] - 3s 10us/step - loss: 0.3294 - acc: 0.8730 - val_loss: 0.3179 - val_acc: 0.8781
Epoch 26/30
258502/258502 [==============================] - 2s 9us/step - loss: 0.3286 - acc: 0.8737 - val_loss: 0.3486 - val_acc: 0.8628
Epoch 27/30
258502/258502 [==============================] - 3s 11us/step - loss: 0.3273 - acc: 0.8743 - val_loss: 0.3090 - val_acc: 0.8839
Epoch 28/30
258502/258502 [==============================] - 3s 10us/step - loss: 0.3247 - acc: 0.8763 - val_loss: 0.3247 - val_acc: 0.8753
Epoch 29/30
258502/258502 [==============================] - 3s 11us/step - loss: 0.3237 - acc: 0.8767 - val_loss: 0.3121 - val_acc: 0.8854
Epoch 30/30
258502/258502 [==============================] - 3s 11us/step - loss: 0.3216 - acc: 0.8780 - val_loss: 0.3185 - val_acc: 0.8790
acc: 87.90%
Saved model to disk
```
