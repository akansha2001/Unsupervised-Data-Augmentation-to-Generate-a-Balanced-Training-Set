# Unsupervised Data Augmentation to Generate a Balanced Training Set

This includes the report for my Summer Research Internship (2021) at [VIPLAB, IIT Kharagpur](https://cse.iitkgp.ac.in/?researchlabs.html) under the guidance of [Prof. Jayanta Mukhopadhyay](https://dblp.org/pid/21/3522.html).

## Problem Statement
Our aim is to form an importance vector (a trainable parameter of a feed forward artificial neural network, with importance values corresponding to each of the training examples) to generate pseudo labels for a dataset. These pseudo labels will help segregate the examples into majority and minority classes. We can then proceed to obtain a balanced distribution of training examples to train the model.

## Objective
To obtain a balanced distribution of training examples for better training of the model, without using the real labels.

## Motivation
Skewed data sets are those datasets that have an inadequate number of training examples for one of the classes, known as the minority class. In datasets with a small number of training examples, the representation of minority examples is very poor, i.e. the model learns almost nothing about the minority class from these few examples. While this is representative of the real world distribution of data, the machine learning models cannot learn the input to output mappings for the minority class as well as it learns the mapping for the majority class. This causes problems in generalizing to real world examples in the test set. Further, the fact that these examples are a minority means they are rare to find, and we desire a model that will assign adequate weightage to such rare examples, as there seems to be more to “learn” from these examples, in order to classify them correctly. The overwhelming number of majority examples undermines the contribution of the minority examples in determining the total loss (or a model evaluation metric like accuracy)- which results in poor classification of minority examples. Our aim is to determine a classification for the training examples (by means of an importance vector and subsequently using a suitable threshold to generate pseudo labels) of a skewed dataset, and use this classification to obtain a balanced distribution of the two classes, which will help the model learn a better input to output mapping for the given dataset.
(Here, we have considered the case of binary datasets, with a skewed distribution of training examples. Highly skewed distributions are harder to train, so we stick to a 70-30 distribution of majority-minority data.)

## Methodology 
### Intuition
We have proposed a fix for the above problem in our paper, which assigns an importance value to each of the training examples in the dataset. 
The "importance" of an example is a scalar value associated with the example (which is fed into the network as a trainable parameter of the model), that determines how “useful” the example is, for training the model. A higher value of importance suggests that the machine stands to “learn” more from the corresponding training example, i.e. it would help if there were more examples of the sort. 
Here we have used the idea that similar examples produce similar gradients during backpropagation [[1]](https://arxiv.org/pdf/2003.07422v2.pdf) and thus, their associated importances are expected to be similar as well (updating a parameter in backpropagation is directly proportional to the gradient of the parameter).
We use an unsupervised learning loss function (EntropyLoss: specifically useful for numerical or image data; it does not work on categorical input data) to train the model with the trainable importance vector (during backpropagation, the importance vector is regularly updated along with the usual trainable parameters of the network). 
The real labels are not used here, hence we do not require the real label information to find the importance vector for the training set.
### Assumptions
Having obtained the importance vector, we assume the following: it is expected that the model will assign higher importance to the examples that belong to the minority class and do not contribute significantly to the overall loss of the model. That is to say the model wants more such examples, so it can learn a better input to output mapping to generalize better to real world examples. 
On the other hand, majority examples, by virtue of their percentage in the training dataset, contribute substantially to the overall loss, and the model is expected to lower the importance values of such examples (during backpropagation) as these examples are numerous. Other explanations for the importance values of different examples could be: lower importance of an example because it is an “exception” i.e. it tends to “confuse” the model by contributing significantly to the overall loss, etc. 
Here, we adhere to our assumption that higher importance is assigned to the minority class, and relatively lower importance is assigned to the majority class.
### Using the importance values for oversampling
We use the importance values in the vector to find a threshold. We can take a simple arithmetic mean of the minimum and maximum values in the importance vector, or some fraction of the sum of imp.min() and imp.max(), or use K Means classification on the importance vector to classify them into 2 classes, etc. 
Using the threshold obtained, we generate “pseudo labels” for the different training examples, to classify them into majority and minority classes. Determination of the threshold has nothing to do with the actual labels of the examples. However, we do use the actual labels to compare with the pseudo labels, to generate an F-score or to determine the NMI by comparing the real and pseudo labels. A high F-score suggests that our assumption for the significance of the importance values of different examples stands true. The real labels are used for verification purposes only.
Proceeding with the assumption that the pseudo labels are indicative of the real labels, we have now obtained an estimate of which examples belong to the majority class, and which belong to the minority class. Using this information only, we can use synthetic sampling techniques like SMOTE to oversample the minority class [[2]](https://papers.nips.cc/paper/2004/file/d7619beb6eb189509885fbc192d2874b-Paper.pdf). We do not undersample the majority class as there is loss of valuable information. For the time being, we overlook the drawbacks of oversampling that disturb the real world distribution of the binary data, and focus on ensuring the model can learn more from the minority examples.

## Dataset Used
We have used the Breast Cancer Wisconsin (Original) Dataset from the UCI Machine Learning Repository. It has the following properties:
1.	10 integer valued attributes for each example
2.	Two output labels (2 for benign, 4 for malignant)
3.	699 training examples: 241 malignant, 458 benign

## Model Description 
We use an MLP model for binary classification. A multilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN) that uses a learning technique called backpropagation for training. Its multiple layers and non-linear activation distinguish MLP from a linear perceptron.
In our code, we have used a specially defined EntropyLoss function for unsupervised learning.

## Importance Vector
The importance vector is a trainable parameter of the model that will help us determine the relative “importance” of different training examples, to the model. 
The idea is to generate an importance vector for the examples in the dataset, without using the real labels (unsupervised learning). The motivation was derived from our knowledge of the backpropagation algorithm for optimizing a given loss function. Similar examples generate similar gradients, and are thus expected to have similar “importances” associated with them (the idea of importance is explored below). 
Our aim is to determine which examples have “higher” importance. The backpropagation algorithm is used to obtain a backward gradient for all trainable parameters of the model (using a computation graph and applying chain rule. We have used PyTorch as these functionalities are easily available as built-in modules, so we can focus on implementing our idea)

## The idea of importance and the associated mathematics
The importance vector is initially set to a vector of ones, the length of which is the same as the number of training examples in our dataset. That is, each example has its associated importance or weight which will eventually determine how “effective” or “desirable” a particular example is, to training an MLP model. Initially, all examples have the same “importance”.
The importance vector is multiplied to the training batch being fed into the network (using python broadcasting techniques). We have prepared a custom dataset class that will pass three items to the DataLoader: 
1. The training example “x”  (the EntropyLoss function used in our code works for purely numerical data or image data only); 
2. the real output label “y” (which will remain unused as we use unsupervised loss function. This real label, however, helps us test our model by generating a normalized mutual information score for each epoch while training) ; 
3. The importance value (scalar) corresponding to a particular example.

## The mechanism of the algorithm
Each training example is multiplied with its corresponding importance value before being fed into the network for forward propagation through the MLP. Initially, all the “importance” values are 1. At this point, all examples are given equal weightage while training the model. In this first step, the forward propagation is the same as that of usual deep learning.

The backpropagation step computes the gradients of the parameters using the computation map and chain rule. While gradients may be computed for all parameters (for mathematical reasons), only the trainable parameters are updated during backpropagation. As we know, the optimizer tries to find the optimal model parameters that minimize the loss. The importance of an example being a trainable parameter of the model, is updated during backpropagation. There are the following cases:
* The training example contributes a considerable fraction of the total loss, i.e. it increases the loss. In simple terms, this example is detrimental to the model: it may be confusing the model, or the model may be unable to “learn” much from this example. In this case, it is not very “important” to the model: this is mathematically implemented by reducing the importance value of this example in the backpropagation step.
* The training example contributes to a small fraction of the total loss. It does not increase the loss significantly. This example is thus of more “importance” to the model as the model stands to “learn” more from this example: this is mathematically implemented by increasing the importance value of this example in the backpropagation step.

At the end of training, the importance vector is expected to reflect the relative importances of the different training examples. At this point, we assume that the examples belonging to the “minority” class in the skewed binary classification datasets have been assigned higher importance values: the minority class has fewer examples, so they are expected to have a lower contribution to the total loss, as compared to the majority class. (Here we use the idea that “similar” examples generate similar gradients. We thus expect the similar examples of the minority class to have comparable importance values assigned to them.) 
The model could use more examples from the minority class to obtain enhanced model weights and biases that will generalize better to real life examples, i.e. the model stands to “learn” more from the rare minority class. It is expected that this “desire to learn more” or “want for minority examples” will be reflected in the importance values assigned to the examples. By choosing an appropriate “threshold” for the importance vector (by studying the importance values in the vector and using the statistical tools at our disposal to find a threshold), we can thus obtain pseudo labels to determine whether an example belongs to the majority class or the minority class. This procedure has not made use of the real labels; using only the pseudo labels so generated, we can proceed to oversample the minority class using available sampling techniques and thereby obtain a more balanced distribution: one the model can “learn” from in a better way.

## Segregation of the 2 classes using pseudo labels and oversampling
As discussed earlier, we use the importance vector and importance values to generate pseudo labels for the examples. This helps us segregate the majority and minority classes. We then oversample the “minority” class so found using synthetic sampling (SMOTE) on the pseudo labels to obtain a balanced distribution of classes. Undersampling the majority class results in loss of training data, hence oversampling is preferable.
Now we have a relatively balanced dataset to train the model on. The NMI metric is expected to be enhanced while training and testing this new model.
Results and discussion:
(It is preferable to have purely numerical input data as it is a tedious process to encode the categorical data to obtain a suitable mathematical representation)
Breast Cancer Dataset:
The Breast Cancer Wisconsin (Original) Dataset has the following properties:
Data Set Characteristics: Multivariate
Number of Instances: 699
Number of Attributes: 10
Attribute Characteristics: Integer

###	Model

Sequential(
  (0): Linear(in_features=10, out_features=96, bias=True)
  (1): ReLU(inplace=True)
  (2): Linear(in_features=96, out_features=32, bias=True)
  (3): ReLU(inplace=True)
  (4): Linear(in_features=32, out_features=2, bias=True)
)


####	We have used Xavier initialization and Adam optimizer with:

num_epochs, learning_rate, batch_size = 100, 0.001, 32

 
####	Training NMI:

NMI starts off at 0.6356888745027739 for the first epoch, 
peaks at 0.8237724836895612 for the 5th epoch 
and eventually stabilizes to 0.7941075368566127.

####	We determine the threshold as (a stricter threshold has been used)

thresh = 1*(imp.max()+imp.min())/3

####	And the corresponding pseudo labels are generated as a tensor using properties of boolean logic

new_imp = 1* torch.tensor(imp>thresh)

####	Comparing new_imp with the real labels, we have

Confusion matrix:

[[388  70]
[ 89 152]]

F score: 0.6566

NMI: 0.18564019463441173

####	We check to see the percentage of minority examples is 31.76% now, as opposed to 34.48% using the real labels.

####	Training after oversampling: (importance vector is not used)

The same model is retained.

num_epochs, learning_rate, batch_size = 20, 0.001, 32

NMI starts off at 0.5999974901884495 for the first epoch, 
then it rises 
and eventually stabilizes to 0.7843688130628772.


####	Testing NMI:

0.8074102006870343


•	Total number of examples before oversampling: 699
•	Total number of examples after oversampling: 954

## Conclusion
Using our knowledge of backpropagation and the similarity of gradients for similar examples, we have used an importance vector to generate pseudo labels that classify training examples of a skewed binary dataset into majority and minority classes, using unsupervised learning techniques. This helped us determine which examples to assign more importance to, and thereby generate a balanced distribution to train the model. A balanced distribution of data further helped the model learn a more generalized input to output mapping, as seen in the NMI determined by testing.

## Proposal to address the shortcomings
We observed that the model failed to produce useful results for non-numerical data. It did not help to eliminate the non-numerical data altogether, as useful information was lost. Further, it was all the more difficult to obtain a numerical representation for such features of data. To address this drawback, we propose looking for techniques to effectively encode non-numerical data numerically, using some form of encoding mechanism such as neural nets or other simpler approaches. 
Further, the model was seen to perform well only for imbalanced binary datasets, and could not be scaled to multiclass classification problems with imbalanced datasets. This can be attributed to the fact that only one class is deemed to be a "minority" class at a time, hence other "minority" classes are not taken into account as we expect.

