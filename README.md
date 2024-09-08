# Plant Seedlings Classification

## 1. Technical Description
### **Step 1.** Install requirements.txt
```
pip install -r requirements.txt
```
### **Step 2.** run main.py
```
python main.py
```

#### Issue : Multiclass classification (Images)
#### Using model : ResNet50
#### Validation method : Leave-one-out cross-validation
#### Some configs : training_epoch = 25, The ratio between training dataset and validation dataset -> 4:1

## 2. Experimental results
### Loss Curve
![image](https://github.com/zan8161/Plant_Seedlings_Classification/blob/main/result/loss.png)
### Confusion Matrix
![image](https://github.com/zan8161/Plant_Seedlings_Classification/blob/main/result/confusion_matrix.png)
### Kaggle Submission Score
![image](https://github.com/zan8161/Plant_Seedlings_Classification/blob/main/result/kaggle_score.png)
