# Plant Seedlings Classification

## 1. Technical Description
### 1.1 Program Execution
#### **Step 1.** Install requirements.txt
```
pip install -r requirements.txt
```
#### **Step 2.** run main.py
```
python main.py
```
### 1.2 Project Enviroment
#### Python version : 3.10.14
#### Cuda version : 11.6
#### Issue : Multiclass classification (Images)
#### Using model : ResNet50
#### Validation method : Leave-one-out cross-validation
#### Some configs : training_epoch = 25, The ratio between training dataset and validation dataset -> 4:1

## 2. Experimental results
### 2.1 Loss Curve
![image](https://github.com/zan8161/Plant_Seedlings_Classification/blob/main/result/loss.png)
### 2.2 Confusion Matrix
![image](https://github.com/zan8161/Plant_Seedlings_Classification/blob/main/result/confusion_matrix.png)
### 2.3 Kaggle Score
![image](https://github.com/zan8161/Plant_Seedlings_Classification/blob/main/result/kaggle_score.png)
