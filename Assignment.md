---
title: "Predicting physical activity type using sensor data"
output: html_document
---
# Predicting physical activity type using sensor data

The aim of this project is to train a machine learning algorithm which can be used to predict in what way someone is performing an activity by using sensor data.
The participants do an activity in five different ways, labelled `A`, `B`, `C`, `D` and `E`, and corresponding sensor outputs are recorded.
This document will go through the process of partitioning the data into a training and test set, ending with using the model to predict `20` techniques with unknown labels.
The training data set can be found [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) and the `20` final testing problems can be found [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv).

First load up the required `caret` library and turn the seed up to `11`.

```r
library(caret)
set.seed(11)
```


## Data reading and preprocessing.
Load in the training data provided with the assignment, treating the `#DIV/0!` entries as `NA`.


```r
data <- read.csv('Data/pml-training.csv', na.strings=c("NA","#DIV/0!"))
```

Doing some quick data analysis, we check how many columns have firstly any `NA` entries, then more than `95%` `NA` entries.


```r
# Columns which have any NAs (100).
sum(apply(is.na(data), 2, sum) / dim(data)[1] > 0)
```

```
## [1] 100
```

```r
# Columns which have more than 95% NAs.
sum(apply(is.na(data), 2, sum) / dim(data)[1] > .95)
```

```
## [1] 100
```

```r
# Both are apparently 100.
```

Notice here that there are two main types of columns, ones that have a complete set of datapoints, and ones that have less than `5%` rows filled in.
Seeing as imputing values with more than `95%` missing values would probably be a silly idea, we simply drop those `100` columns.
The names to remove are listed below.


```r
names(data)[(apply(is.na(data), 2, sum) / dim(data)[1] > .95)]
```

```
##   [1] "kurtosis_roll_belt"       "kurtosis_picth_belt"     
##   [3] "kurtosis_yaw_belt"        "skewness_roll_belt"      
##   [5] "skewness_roll_belt.1"     "skewness_yaw_belt"       
##   [7] "max_roll_belt"            "max_picth_belt"          
##   [9] "max_yaw_belt"             "min_roll_belt"           
##  [11] "min_pitch_belt"           "min_yaw_belt"            
##  [13] "amplitude_roll_belt"      "amplitude_pitch_belt"    
##  [15] "amplitude_yaw_belt"       "var_total_accel_belt"    
##  [17] "avg_roll_belt"            "stddev_roll_belt"        
##  [19] "var_roll_belt"            "avg_pitch_belt"          
##  [21] "stddev_pitch_belt"        "var_pitch_belt"          
##  [23] "avg_yaw_belt"             "stddev_yaw_belt"         
##  [25] "var_yaw_belt"             "var_accel_arm"           
##  [27] "avg_roll_arm"             "stddev_roll_arm"         
##  [29] "var_roll_arm"             "avg_pitch_arm"           
##  [31] "stddev_pitch_arm"         "var_pitch_arm"           
##  [33] "avg_yaw_arm"              "stddev_yaw_arm"          
##  [35] "var_yaw_arm"              "kurtosis_roll_arm"       
##  [37] "kurtosis_picth_arm"       "kurtosis_yaw_arm"        
##  [39] "skewness_roll_arm"        "skewness_pitch_arm"      
##  [41] "skewness_yaw_arm"         "max_roll_arm"            
##  [43] "max_picth_arm"            "max_yaw_arm"             
##  [45] "min_roll_arm"             "min_pitch_arm"           
##  [47] "min_yaw_arm"              "amplitude_roll_arm"      
##  [49] "amplitude_pitch_arm"      "amplitude_yaw_arm"       
##  [51] "kurtosis_roll_dumbbell"   "kurtosis_picth_dumbbell" 
##  [53] "kurtosis_yaw_dumbbell"    "skewness_roll_dumbbell"  
##  [55] "skewness_pitch_dumbbell"  "skewness_yaw_dumbbell"   
##  [57] "max_roll_dumbbell"        "max_picth_dumbbell"      
##  [59] "max_yaw_dumbbell"         "min_roll_dumbbell"       
##  [61] "min_pitch_dumbbell"       "min_yaw_dumbbell"        
##  [63] "amplitude_roll_dumbbell"  "amplitude_pitch_dumbbell"
##  [65] "amplitude_yaw_dumbbell"   "var_accel_dumbbell"      
##  [67] "avg_roll_dumbbell"        "stddev_roll_dumbbell"    
##  [69] "var_roll_dumbbell"        "avg_pitch_dumbbell"      
##  [71] "stddev_pitch_dumbbell"    "var_pitch_dumbbell"      
##  [73] "avg_yaw_dumbbell"         "stddev_yaw_dumbbell"     
##  [75] "var_yaw_dumbbell"         "kurtosis_roll_forearm"   
##  [77] "kurtosis_picth_forearm"   "kurtosis_yaw_forearm"    
##  [79] "skewness_roll_forearm"    "skewness_pitch_forearm"  
##  [81] "skewness_yaw_forearm"     "max_roll_forearm"        
##  [83] "max_picth_forearm"        "max_yaw_forearm"         
##  [85] "min_roll_forearm"         "min_pitch_forearm"       
##  [87] "min_yaw_forearm"          "amplitude_roll_forearm"  
##  [89] "amplitude_pitch_forearm"  "amplitude_yaw_forearm"   
##  [91] "var_accel_forearm"        "avg_roll_forearm"        
##  [93] "stddev_roll_forearm"      "var_roll_forearm"        
##  [95] "avg_pitch_forearm"        "stddev_pitch_forearm"    
##  [97] "var_pitch_forearm"        "avg_yaw_forearm"         
##  [99] "stddev_yaw_forearm"       "var_yaw_forearm"
```

The following commands do two things.
Firstly, it drops the columns which are mostly `NA` values (those listed above), then it drops the first `7` columns of the dataset.
The reason for dropping the first `7` columns is because they are not columns we can predict on, `user_name`s and timestamp variables.


```r
columnsKept <- (apply(is.na(data), 2, sum) / dim(data)[1] <= .95)
# Get rid of the following columns.
columnsKept[1:7]
```

```
##                    X            user_name raw_timestamp_part_1 
##                 TRUE                 TRUE                 TRUE 
## raw_timestamp_part_2       cvtd_timestamp           new_window 
##                 TRUE                 TRUE                 TRUE 
##           num_window 
##                 TRUE
```

```r
columnsKept[1:7] <- FALSE
data <- data[,columnsKept]
```

## Training the Model.
Now that the data has been cleaned, it's time to get to the machine learning part of the project.
The first step is to partition the data into a training and testing set.


```r
inTrain <- createDataPartition(y=data$classe, p=0.75, list=FALSE)
training <- data[inTrain,]
testing <- data[-inTrain,]
```

In my original attempts to train a random forest on the full `training` set, my `R` session took up large amounts of RAM and repeatedly crashed after only a minute of calculation.
To get around this problem, I ran `PCA` before applying random forest, and, seeing as the following random forest training took my computer `70` minutes, maybe it was a good thing I didn't continue with the pull `52` variables.
With `PCA`, I used the top `25` components by varience.


```r
# Try PCA before random forest.
# Using the whole 52 variables made my R session like to crash.
# Took 70minutes.
preProc <- preProcess(training[,-53], method="pca", pcaComp=25)
trainPC <- predict(preProc, training[,-53])
modFit <- train(training$classe ~ ., method="rf", data=trainPC)
confusionMatrix(training$classe, predict(modFit, trainPC))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

Looking at the confusion matrix above, one can see that the trained model has an accuracy rate of `100%` on the training set, using only `25` predictors.
This is suspiciously good and could be a strong sign of overfitting, however, we will look at what happens on the test set before jumping to conclusions.

## Testing the Model.
Using this trained model, we move on to checking our model on the test set and look at the `confusionMatrix`.


```r
testPC <- predict(preProc, testing[,-53])
confusionMatrix(testing$classe, predict(modFit, testPC))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1390    2    1    1    1
##          B   17  922    9    0    1
##          C    1   13  835    5    1
##          D    2    0   33  767    2
##          E    0    2    4    2  893
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9802          
##                  95% CI : (0.9759, 0.9839)
##     No Information Rate : 0.2875          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.975           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9858   0.9819   0.9467   0.9897   0.9944
## Specificity            0.9986   0.9932   0.9950   0.9910   0.9980
## Pos Pred Value         0.9964   0.9715   0.9766   0.9540   0.9911
## Neg Pred Value         0.9943   0.9957   0.9884   0.9980   0.9988
## Prevalence             0.2875   0.1915   0.1799   0.1580   0.1831
## Detection Rate         0.2834   0.1880   0.1703   0.1564   0.1821
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9922   0.9875   0.9709   0.9904   0.9962
```

The concerns before about overfitting, while not totally unfounded, shouldn't be too much of a concern in this case.
The accuracy rate here is not overly bad and should be adequate for the purposes of this project (though a bit of tweaking needs to be done to get one of the final predictions).

### Out of sample error rate.
The confusion matrix shows an accuracy of `0.9806`.
This is the accuracy we should expect our model to perform at in real-world situations with new data.
This value of accuracy says that the model trained here should have an out-of-sample error rate of about `2%`.

## Run the predictions on the project's test set.
Using this trained model, we turn our attention to predicting the `20` techniques in the provided `testing` set.
This simply applies the same transformations that were done to the training set to this new `projectProblems` set.


```r
projectProblems <- read.csv('Data/pml-testing.csv', na.strings=c("NA","#DIV/0!"))

projectProblems <- projectProblems[,columnsKept]

# 53 is the 'problem_id'.
projectProblemsPC <- predict(preProc, projectProblems[,-53])
predictions <- predict(modFit, projectProblemsPC)
predictions
```

```
##  [1] B A B A A E D B A A A C B A E E A B B B
## Levels: A B C D E
```

```r
# 19/20 correct.
```

## Getting the last prediction correct.
After submitting the contents of `predictions` to the assignment submission page, only `19` out of the `20` predictions were correct (the `11`th problem was incorrect).
Unfortunately this is a bit annoying, mainly because the first time I tried this I didn't bother to set the seed and got all `20` in the first shot, so now I have to do more work to find the final answer (the cheat way would be to just mess with the seed until I get all `20` again, but that would be bad form).

There are two ways which we can go about trying to find the correct prediction from here, both use the fact that the incorrect prediction was `A`.
The first path is to look at the confusion matrix of our model above, looking at the first row, `A` was predicted `1390` times when the answer was actually `A`, twice when the answer was `B`, and once for each `C`, `D` and `E`.
This could be used to *guess* that this `A` was actually a misclassified `B`, but with such small numbers I'm not too confident with that analysis.
The other option is to retrain a new model where all of the `A` training cases are removed, then use it to predict the `20` project problems and see what the `11`th problem is predicted as.
This approach is shown below.

This code will remove all of the `A` cases from both the training and the testing set.


```r
training <- training[training$classe != "A",]
training$classe <- droplevels(training$classe)
testing <- testing[testing$classe != "A",]
testing$classe <- droplevels(testing$classe)
```

Then the same process is done that was done before, doing `PCA`, training and testing with a `confusionMatrix`.


```r
preProc <- preProcess(training[,-53], method="pca", pcaComp=25)
trainPC <- predict(preProc, training[,-53])
modFit <- train(training$classe ~ ., method="rf", data=trainPC)
confusionMatrix(training$classe, predict(modFit, trainPC))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    B    C    D    E
##          B 2848    0    0    0
##          C    0 2567    0    0
##          D    0    0 2412    0
##          E    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9996, 1)
##     No Information Rate : 0.2704     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000    1.000   1.0000
## Specificity            1.0000   1.0000    1.000   1.0000
## Pos Pred Value         1.0000   1.0000    1.000   1.0000
## Neg Pred Value         1.0000   1.0000    1.000   1.0000
## Prevalence             0.2704   0.2437    0.229   0.2569
## Detection Rate         0.2704   0.2437    0.229   0.2569
## Detection Prevalence   0.2704   0.2437    0.229   0.2569
## Balanced Accuracy      1.0000   1.0000    1.000   1.0000
```

```r
testPC <- predict(preProc, testing[,-53])
confusionMatrix(testing$classe, predict(modFit, testPC))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   B   C   D   E
##          B 934  12   2   1
##          C  12 836   5   2
##          D   2  33 764   5
##          E   3   7   4 887
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9749          
##                  95% CI : (0.9692, 0.9798)
##     No Information Rate : 0.271           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9665          
##  Mcnemar's Test P-Value : 0.0004188       
## 
## Statistics by Class:
## 
##                      Class: B Class: C Class: D Class: E
## Sensitivity            0.9821   0.9414   0.9858   0.9911
## Specificity            0.9941   0.9928   0.9854   0.9946
## Pos Pred Value         0.9842   0.9778   0.9502   0.9845
## Neg Pred Value         0.9934   0.9804   0.9959   0.9969
## Prevalence             0.2710   0.2531   0.2209   0.2551
## Detection Rate         0.2662   0.2382   0.2177   0.2528
## Detection Prevalence   0.2704   0.2437   0.2291   0.2568
## Balanced Accuracy      0.9881   0.9671   0.9856   0.9929
```

```r
projectProblemsPC <- predict(preProc, projectProblems[,-53])
predictions <- predict(modFit, projectProblemsPC)
predictions
```

```
##  [1] B C B C C B D B E C B C B B E E E B B B
## Levels: B C D E
```

```r
# The misclassified one from before was the 11th one.
predictions[11]
```

```
## [1] B
## Levels: B C D E
```

```r
# This new prediction is correct (B).
# Final answers: B A B A A E D B A A B C B A E E A B B B
```

Indeed the new classifier where `A` was no-longer an option has sucessfully predicted `B` to be the activity undertaken (the same as we *guessed* above).
Giving all `20` out of `20` correct.
