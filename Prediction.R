library(caret)

# Load the training data set
trainingAll <- read.csv("pml-training.csv",na.strings=c("NA",""))

# Discard columns with NAs
NAs <- apply(trainingAll, 2, function(x) { sum(is.na(x)) })
trainingValid <- trainingAll[, which(NAs == 0)]

# Create a subset of trainingValid data set
trainIndex <- createDataPartition(y = trainingValid$classe, p=0.2,list=FALSE)
trainData <- trainingValid[trainIndex,]

# Remove useless predictors
removeIndex <- grep("timestamp|X|user_name|new_window", names(trainData))
trainData <- trainData[, -removeIndex]


# Configure the train control for cross-validation
tc = trainControl(method = "cv", number = 4)

# Fit the model using Random Forests algorithm
modFit <- train(trainData$classe ~., 
                data = trainData, 
                method="rf", 
                trControl = tc, 
                prox = TRUE,
                allowParallel = TRUE)

print(modFit)
print(modFit$finalModel)

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