# object_recognition_using_ResNet50

This project uses a ResNet50 backbone pre-trained on ImageNet’s millions of images and adapts it to the CIFAR-10 dataset.

The notebook includes following steps:
1)Fetch & prep CIFAR-10 via the Kaggle API, unzip the 32×32 PNGs, map labels from trainLabels.csv, split into train/test, and scale pixels to [0,1].

2)Train a tiny Dense network (Flatten→Dense(64)→Softmax) to ~40% accuracy.

3)Upsample each image to 256×256, feed through ResNet50 then add a Flatten→BatchNorm→Dense(128)→Dropout→Dense(64)→Dropout→Softmax.

4)Fine-tune for 10 epochs with RMSProp and sparse categorical crossentropy, boosting test accuracy to 94%.

![image](https://github.com/user-attachments/assets/959d03cc-70dd-4926-b373-09c5a725ac9b)

![image](https://github.com/user-attachments/assets/46a94943-d791-40a1-a7b1-e8ad8ff7210a)


