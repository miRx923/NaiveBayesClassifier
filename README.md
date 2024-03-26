# NaiveBayesClassifier


School assignment about implementing a popular machine learning algorithm Naive Bayes Classifier in C# programming language.

<h1>Documentation of the Naive Bayes Classifier in C#</h1>

## 1. Introduction
&emsp;&emsp;This documentation provides a description and code usage guide for the Naive Bayes Classifier (hereafter referred to as "NBC" or "classifier") in C#. NBC is one of the most common machine learning classification algorithms and is based on Bayes Theorem. The code in this document implements NBC and allows training a classifier on a set of input examples and then classifying the new examples into one of the categories based on the probability of occurrence of each attribute for each category. This document contains a description of the NaiveBayesClassifier class, which implements the NBC algorithm consisting of a method for training and a method for classifying, along with examples for using the class. In addition, the document focuses on explaining the working principles of NBC and its applications in machine learning.

## 2. Training sets
&emsp;&emsp;The code is ready to be trained on two training sets namely the PlayTennis dataset[1] and the truncated Iris-Species dataset[2]. It has been modified to contain only 2 classes out of 3 so that we can perform dichotomous classification. Also, in addition to truncating the dataset, it was split into an 80% training set in the file "Iris-TrainingData.csv” and a 20% testing part in the file "Iris-TestingData.csv". The Play-Tennis dataset was also split into a training set "play_tennis-TrainingData.csv” and a testing set. 

PlayTennis data is represented by four comma-separated attributes, and the last attribute is the class.

<img src="_ignore_Documentation images/play-tennis_data.png" alt="play-tennis_data" width="300">

Image 1: Dataset Play-Tennis

Iris Species data works the same way, four attributes separated by commas and a class. This dataset is truncated by a third, it is modified to include only two of the three original classes, namely Iris-setosa and Iris-virginica.

<img src="_ignore_Documentation images/iris-species_data.png" alt="iris-species_data" width="300">

Image 2: Sample of a part of the Iris-Species dataset

# 3. Code components
## 3.1 Class NaiveBayesClassifier
&emsp;&emsp;The first C# file NaiveBayesClassifier.cs contains the main <b>NaiveBayesClassifier</b> class, which declares three variables: <b>classCounts, featureCounts, numExamples.</b>

<img src="_ignore_Documentation images/1.png" alt="class variables" width="1000">

Image 3: Class variable declaration

• <b>classCounts</b> is a dictionary that stores the number of occurrences of each class in
the training set.

• <b>featureCounts</b> is a dictionary of vocabularies that stores the number of
occurrences of each attribute for each class in the training set.

• <b>numExamples</b> stores the total number of examples in the training set.

### 3.1.1 Constructor class
&emsp;&emsp;In the class constructor, the classifier class variables are initialized with initial values.

<img src="_ignore_Documentation images/2.png" alt="class constructor" width="1000">

Image 4: Class constructor

### 3.1.2. Train method
&emsp;&emsp;The <b>Train (string[] features, string label)</b> method is used to train the classifier, that is, to learn a model based on the training set. The first input to the method is the attributes (features) of the input example, and the second input to the method is the class (label) to which the input example belongs. First, the class is checked to see if it already exists in the classCounts dictionary. If not, it is added with a starting value of O and incremented later. Then, for each attribute of the input example, it is determined whether it already exists in the featureCounts dictionary for the class. If not, it is added with a starting value of 0 and incremented later. The number of occurrences of the attribute for the class is then incremented. The total number of instances of numExamples is also incremented at the end.

<img src="_ignore_Documentation images/3.png" alt="train method" width="1000">

Image 5: Train method

### 3.1.3. Predict method

&emsp;&emsp;The <b>Predict (string[] features)</b> method is used to classify new examples based on the trained data. The input to the method is an array of attributes (features) of the new input example to be classified. First, the variable bestLabe1 is initialized as an empty string and the variable bestScore is initialized as the smallest value of the double data type. The method traverses all the classes in classCounts and for each class, it computes the probability of occurrence of that class in the training set and then computes the probability that the input example belongs to that class. The calculation of this probability is done according to the following formula:

<b>P(class) + Σ P(feature<sub>i</sub> | class)</b>


<b>Where:</b>

• <b>P(class)</b> represents the probability of occurrence of a given class in the training set,

• <b>Σ P(feature<sub>i</sub> | class)</b> represents the probability of occurrence of attribute <i>i</i> in the given class.

This probability is compared to the current highest probability. If the calculated probability is greater than the current best probability, the variable bestLabel is updated to the class name and the variable bestScore is updated to the new highest probability value. Finally, the class name with the highest probability is returned.

<img src="_ignore_Documentation images/4.png" alt="predict method" width="1000">

Image 6: Predict method

## 3.2 Main program
&emsp;&emsp;The second C# file, Program.cs, contains the main program that allows you to train the classifier on two training sets.

### 3.2.1. Training with Play-Tennis dataset 
&emsp;&emsp;This chapter focuses on training a classifier using Play-Tennis data. With this training data, the classifier is trained on 80% of the data. This part of the main program involves creating an instance of the classifier object. A standard object is created to read the .csv file containing the data. The first line of the file is skipped because it does not contain the data, but the header of the file. Then the file is read line by line and the comma separated data is separated into the value list fields and the expected class is stored in the label variable. The new classifier is then trained using the Train method on the Play-Tennis data.

<img src="_ignore_Documentation images/5.png" alt="training with play tennis" width="1000">

Image 7: Main program for training with Play-Tennis data

### 3.2.2. Training with Iris-Species dataset
&emsp;&emsp;This chapter focuses on training the classifier using Iris-Species data. For these training data, the classifier is trained on 80% of the data. Initially, an instance of the classifier object is created. A standard object is created to read the .csv file with the training set "Iris-TrainingData.csv". The first line is skipped because it does not contain the data, but the header of the file. Then the file is read line by line and the comma-separated data is separated into the fields of the values list and the expected class is stored in the label variable. The new classifier is then trained using the Train method on the Iris-Species data.

<img src="_ignore_Documentation images/6.png" alt="training with iris species" width="1000">

Image 8: Main program for training with Iris-Species data

### 3.2.3. Testing the classifier on Play-Tennis data
&emsp;&emsp;This chapter focuses on testing the classifier using Play-Tennis data. It is tested on 20% of the data. The file is read line by line as in training. This time with the test data. Each row is split into attribute values and per class. Then the Predict method is used to predict the class and compare it to the actual example class to determine if it was a correct prediction.

<img src="_ignore_Documentation images/7.png" alt="testing with play tennis" width="1000">

Image 9: Testing a trained classifier with Play-Tennis data

### 3.2.4. Testing the classifier on Iris-Species data
&emsp;&emsp;This chapter focuses on testing the classifier using Iris-Species data. It is tested on 20% of the data. The file is read line by line. This time with test data from the file "Iris-TestingData.csv". Each row is split into attribute values and per class. Then the Predict method is used to predict the class and compare it to the actual example class to determine if it was a correct prediction.

<img src="_ignore_Documentation images/8.png" alt="testing with iris species" width="1000">

Image 10: Testing a trained classifier with Iris-Species data

### 3.2.5. Measuring accuracy metrics
&emsp;&emsp;The main standard efficiency measures Precision, Recall, F1, Accuracy, as well as other less used measures were used to measure the effectiveness of the classifier.

<img src="_ignore_Documentation images/9.png" alt="efficiency measures" width="1000">

Image 11: Calculation of model efficiency measures

## 3.3. Classifier efficiency test results
&emsp;&emsp;The final testing of the classifier produced very favourable results, which can be seen in Tables 1 and 2.

## For Play-Tennis dataset:
| Efficiency metrics             |    Value    |
| ------------------             | ----------- | 
| <b>Precision<b>                | <b>1</b>    | 
| <b>Recall</b>                  | <b>1</b>    | 
| <b>F1</b>                      | <b>1</b>    | 
| <b>Accuracy</b>                | <b>1</b>    | 
| ErrorRate                      | 0           | 
| TruePositiveRate (TPR)         | 1           | 
| TrueNegativeRate (TNR)         | 1           | 
| PositivePredictiveValue (PPV)  | 1           | 
| NegativePredictiveValue (NPV)  | 1           | 
| TruePositiveRate (FNR)         | 0           | 
| FalsePositiveRate (FPR)        | 0           | 

Table 1: Model quality measurement results for Play-Tennis data


## For Iris-Species dataset:
| Efficiency metrics             |    Value    |
| ------------------             | ----------- | 
| <b>Precision<b>                | <b>0,90909</b> | 
| <b>Recall</b>                  | <b>1</b>       | 
| <b>F1</b>                      | <b>0,95238</b> | 
| <b>Accuracy</b>                | <b>0,95000</b> | 
| ErrorRate                      | 0,05000        | 
| TruePositiveRate (TPR)         | 1              | 
| TrueNegativeRate (TNR)         | 0,90000        | 
| PositivePredictiveValue (PPV)  | 0,90909        | 
| NegativePredictiveValue (NPV)  | 1              | 
| TruePositiveRate (FNR)         | 0              | 
| FalsePositiveRate (FPR)        | 0,10000        |

Table 2: Model quality measurement results for Iris-Species data

# 4. Conclusion
&emsp;&emsp;This code contains a simple and efficient implementation of the Naive Bayes Classifier in the C# programming language. The classifier achieved very favorable efficiency results. This algorithm can be used for various tasks such as spam filters, text classification and so on. The implementation of the algorithm includes methods to train and predict the class for the input data. For the implementation, I have used C# programming language and Microsoft Visual Studio 2022 Community Edition environment.

# References
[1] F. Breno, <i>"Play-Tennis dataset"</i>, [online].
<a>https://www.kaggle.com/datasets/fredericobreno/play-tennis</a>, 10. Dec. 2018.

[2] UCI Machine Learning, <i>"Iris-Species dataset"</i>, [online].
<a>https://www.kaggle.com/datasets/uciml/iris</a>, 27. Sep. 2016.
  
# License
Licensed under <b>GPL-3.0.</b>
