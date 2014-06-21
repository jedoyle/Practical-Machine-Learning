Practical Machine Learning Class - Project Write-up
========================================================
In this project write-up, I have used the data from [Human Activity Recognition](http://groupware.les.inf.puc-rio.br/har) (HAR). The aim was to train a model based on the data of various sensor values, which could later be used to predict the **Classe** variable, that is the manner in which the participants of HAR did the exercise.

After having examined the data briefly using the [Rattle GUI](http://rattle.togaware.com/), I have realized that some columns have a lot of missing (NA) values. Instead of trying to model them, I have decided to remove them from the data set. So the first step, after having loaded the required caret library (I've skipped the demonstration of Rattle GUI, since, after all, it was an interactive session with GUI part), was to detect and eliminate columns with a lot of missing values:


```r
library(caret)

# Load the training data set
trainingAll <- read.csv("pml-training.csv",na.strings=c("NA",""))

# Discard columns with NAs
NAs <- apply(trainingAll, 2, function(x) { sum(is.na(x)) })
trainingValid <- trainingAll[, which(NAs == 0)]
```

This resulted in 60 columns (variables), instead of 160.

After having removed the columns with missing values, I have proceeded to create a subset of the training data set because I have seen that the whole set contained **19622** rows (observations) from the HAR study. I thought this was a lot of data because I wanted to use the Random Forests algorithm from the caret package, and my experience with it (when I used it for quizzes) indicated that it was a relatively expensive algorithm, my relatively old laptop computer spent a lot of CPU time and got heated for data sets that were much smaller than this HAR data set. Therefore I have decided to take 20% of the whole HAR data set as a representative sample. 

Moreover, after creating this subset, I also removed the columns related to timestamps, the X column, user_name, and new_window because they were not sensor values, so I thought they would not help much (or at all) for prediction:



```r
# Create a subset of trainingValid data set
trainIndex <- createDataPartition(y = trainingValid$classe, p=0.2,list=FALSE)
trainData <- trainingValid[trainIndex,]

# Remove useless predictors
removeIndex <- grep("timestamp|X|user_name|new_window", names(trainData))
trainData <- trainData[, -removeIndex]
```

As a result, I had a subset of HAR data set that had only 3927 rows of of 54 variables.

Then, based on the suggestion of the instructor ("... how you used cross validation"), I've decided to use cross validation, and instead of the usual 10-fold cross validation, I've used 4-fold cross validation (again, due to the limited resources of my machine, and my impatience, too). After setting the trainControl, I have finally used the Random Forests (rf) algorithm in the following manner:


```r
# Configure the train control for cross-validation
tc = trainControl(method = "cv", number = 4)

# Fit the model using Random Forests algorithm
modFit <- train(trainData$classe ~., 
                data = trainData, 
                method="rf", 
                trControl = tc, 
                prox = TRUE,
                allowParallel = TRUE)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

Having some experience with "rf", I expected relatively good model performance, and a relatively low out of sample error rate:


```r
print(modFit)
```

```
## Random Forest 
## 
## 3927 samples
##   53 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 2946, 2944, 2945, 2946 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.004        0.005   
##   30    1         1      0.004        0.005   
##   50    1         1      0.005        0.007   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```


```r
print(modFit$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE,      allowParallel = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 1.55%
## Confusion matrix:
##      A   B   C   D   E class.error
## A 1116   0   0   0   0     0.00000
## B   10 739  11   0   0     0.02763
## C    0  13 669   2   1     0.02336
## D    1   1   9 631   2     0.02019
## E    0   5   0   6 711     0.01524
```

After having fit the model with training data, I have used it for predictions on test data. I've applied the same removal of columns to the test data as I have done for the training data set:


```r
# Load test data
testingAll = read.csv("pml-testing.csv",na.strings=c("NA",""))

# Only take the columns of testingAll that are also in trainData
testing <- testingAll[ , which(names(testingAll) %in% names(trainData))]

# Run the prediction
pred <- predict(modFit, newdata = testing)

# Utility function provided by the instructor
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(pred)
```

The model performed predictions very accurately, it correctly predicted 20 cases out of 20. This leads to some questions such as "could I have similar accuracy with even less data?", "are some sensor values correlated and therefore can they be left out of training?", and "would another training method such as SVM perform faster and be more accurate?".
