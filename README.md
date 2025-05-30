# Object_Recognition_using_ResNet50

This project demonstrates how to build and fine-tune a ResNet50-based classifier on the CIFAR-10 dataset.

## Why Transfer Learning?

- **Data efficiency**  
  You don’t need tens of thousands of labeled images to train a deep network from scratch.  
- **Speed**  
  Leveraging pre-trained weights gets you to high accuracy in just a few epochs.  
- **State-of-the-art backbone**  
  ResNet50’s residual blocks help the model learn rich, hierarchical features that generalize well.

---

## Pipeline

1. **Data preparation**  
   - Downloaded via Kaggle API  
   - Extracted images and labels  
   - Train/test split (80 %/20 %)  
   - Pixel values rescaled to [0, 1]

2. **Baseline MLP**  
   - Flatten → Dense(64, ReLU) → Dense(10, Softmax)  
   - Serves as a quick sanity check

3. **ResNet50 + custom head**  
   - Upsample 32×32 images to 256×256  
   - Load `ResNet50(weights='imagenet', include_top=False)`  
   - Add Flatten → BatchNorm → Dense → Dropout layers → final softmax  

4. **Training**  
   - Adam (baseline) and RMSProp (ResNet head) optimizers  
   - 10 epochs, 10 % validation split  
   - Early results:  
     - Baseline MLP: ~ 40 % accuracy  
     - ResNet50 transfer: ~ 94 % accuracy

5. **Evaluation & Plots**  
   - Visualize train/val loss and accuracy curves  
   - Report final test accuracy

![image](https://github.com/user-attachments/assets/959d03cc-70dd-4926-b373-09c5a725ac9b)

![image](https://github.com/user-attachments/assets/46a94943-d791-40a1-a7b1-e8ad8ff7210a)


