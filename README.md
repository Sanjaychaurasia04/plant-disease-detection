# 🌿 Plant Disease Detection System

A deep learning model that identifies **38 plant diseases** from leaf images with over **90% accuracy** using TensorFlow and Keras.

![Prediction Demo](https://github.com/yourusername/plant-disease-detection/blob/main/demo.gif?raw=true)  
*Example: Apple Scab detected with 96.2% confidence*

---

## 🔧 Features
- Detects **38 disease classes** from 14 crop species
- **Custom CNN** with 4 convolutional layers + dropout
- Uses **data augmentation** for robust training
- Shows **confidence scores** and predicted labels

---

## 📁 Dataset  
**PlantVillage (Augmented)**:  
[🔗 Kaggle Link](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)  
Includes 63K+ labeled images with healthy and infected leaves.

---

## 🧠 How to Use

### 🔹 Training
```python
model.fit(train_gen, epochs=30, callbacks=[EarlyStopping()])


pred_class, confidence = predict_disease("samples/tomato_leaf.jpg")
# Output: Tomato_Early_Blight (94.5%) ✅
