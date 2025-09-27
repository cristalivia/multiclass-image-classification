# Multiclass Image Classification
This project applies deep learning to classify disaster images into four categories: Urban Fire, Water Disaster, Land Slide, and Earthquake. Two models were tested: a custom CNN and a transfer learning model (MobileNetV2).

## Approach
- Model 1: Custom CNN (Conv2D, MaxPooling, Dense, Dropout, BatchNorm).
- Model 2: MobileNetV2 pretrained on ImageNet, fine-tuned for this dataset.
- Training: Adam optimizer, categorical crossentropy loss, EarlyStopping.
- Tuning: learning rate search (best result at 0.00001).

## Results
Model 1 (Custom CNN): Accuracy ≈ 55%, biased toward majority class.
<img width="704" height="728" alt="image" src="https://github.com/user-attachments/assets/019b4f81-d536-44dd-8137-62fc399cbbaf" />

Model 2 (MobileNetV2): Accuracy ≈ 78%, more balanced across classes.
<img width="708" height="735" alt="image" src="https://github.com/user-attachments/assets/d2c579d9-e465-4ee5-a366-865f1f7ab6f7" />

After LR tuning (0.00001): highest accuracy with improved balance.
<img width="718" height="736" alt="image" src="https://github.com/user-attachments/assets/b9da1a0b-05a6-4dd2-aed7-14272d429290" />

five samples are used to demonstrate the best performance results of Model 2:
<img width="1316" height="719" alt="image" src="https://github.com/user-attachments/assets/c2dba57b-f608-48b0-ac89-04fa34803866" />

## Conclusion

The improved MobileNetV2 model gave much better and more balanced results than the custom CNN. MobileNetV2 performed well because it was pretrained on a large image dataset (ImageNet), which helped it learn useful features for multiclass classification. With learning rate tuning, the model reached the best accuracy, showing that transfer learning and proper optimization are key for reliable performance on imbalanced datasets.

## Tools & Technologies
- Programming & Data Handling: Python, Pandas, NumPy
- Visualization: Matplotli, Seaborn
- Deep Learning: TensorFlow/Keras (CNN, MobileNetV2, Dropout, BatchNorm, Adam, EarlyStopping)
- Evaluation: Scikit-learn (classification report, confusion matrix)
