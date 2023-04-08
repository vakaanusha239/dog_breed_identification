# Dog_breed_identification

There are many types of dogs overall the world. Some known breeds are Pug, Bullmastiff,
German Shepard, etc., Not only these breeds various types of dogs can be identified. The way
to identify the breed is by seeing them live or in photo format. The main objective is to
classify the dogs&#39; images to predict the breed. The implementation of a machine learning
model which can identify the breed by observing a dog image. The existing approaches to
classify using SVM Regressor, Histogram of Oriented Gradient (HOG), and CNN
(Convolutional Neural Network). The advantage of using CNN is, it takes the image as input.
Disadvantages or the drawbacks of these methods are, they take more time to build and train
&amp; their accuracies are not up to the mark to classify all the dogs&#39; images. Now the proposed
thing here is the CNN transfer learning technique which can give better accuracy than the
existing methods. The pre-trained model used for the classification is ResNet50V2 (version
2) from the Keras module. The overall view is to identify the image and classify them
accordingly. The accuracy gained by this approach is 86.24 % approximately. For further
iterations or modifications, the accuracy may be changed.

VARIOUS ARGUMENTS FOR NEURAL NETWORKS:
● include_top: It states whether the model is fully connected or not.
● weights: The pre-trained ‘imagenet’ weights are assigned.
● input_tensor: It is an optional Keras tensor.
● input_shape: The fixed shape of the image is (224, 224, 3)
● pooling: This pooling argument is activated when ‘inclide_top is false’.

⮚ None: This refers output layer as a 4D tensor.
⮚ Average: It applies global average pooling which results in a 2D
tensor.
⮚ Max: Global max pooling needed to be applied.
● classes: optional number of classes to be classified the images.
● classifier_activation: The activation function returns to the top layers when
‘include_top is true.

ResNet50v2 Architecture:

![image](https://user-images.githubusercontent.com/105503752/230727281-6e9395f3-8ff5-4d4d-984f-8184d275e369.png)

Activation functions:

RELU (Rectified Linear Activation Function): It is a tokenwise linear function that outputs directly if it is positive or generates zero. RELU is mostly used for input and hidden layers.
SOFTMAX: It is an exponential function that converts n real numbers into a probability distribution. It is a multiple-dimensional logistic function. Widely used for the output layer.

Dataset:
This project's data set, the Stanford Dogs Dataset, was downloaded from the Kaggle website. About 120 distinct dog breeds are represented in the collection, and each type has at least 150 photos. For this experiment, the author got access to a dataset that had 20,000 dog photos in total. 

Pre-Processing:
Once the dataset was accessible, pre-processing it to transform it into a machine-readable format was the most vital or significant step before using any kind of algorithm. This was because the dataset was an image dataset. It was now necessary to save the photographs in a way that would allow the researcher to use them to train a model. The converted 28*28 grid into a single vector allowed for the storage of the greyscale photos, which were now in a two-dimensional matrix format. This resulted in a flat file with 784 characteristics per image. The researcher stores this information in a CSV file once the photos have been successfully transformed into vectors with characteristics so that it may subsequently be directly accessed by the system and in a machine-readable manner.
It was now necessary to save the photographs in a way that would allow the researcher to use them to train a model. The converted 28*28 grid into a single vector allowed for the storage of the greyscale photos, which were now in a two-dimensional matrix format. This resulted in a flat file with 784 characteristics per image. The researcher stores this information in a CSV file once the photos have been successfully transformed into vectors with characteristics so that it may subsequently be directly accessed by the system and in a machine-readable manner.

Image Augmentation:
Through various processing techniques or combinations of multiple processing, such as random rotation, shifts, shear, flips, etc., image augmentation artificially generates training pictures. The ImageDataGenerator API in Keras makes it simple to develop an enhanced picture generator. A real-time data augmentation tool called ImageDataGenerator creates batches of picture data.

![image](https://user-images.githubusercontent.com/105503752/230727351-db996469-a873-49b8-80bc-59a7bfbf09f5.png)

Algorithm:

![image](https://user-images.githubusercontent.com/105503752/230727368-366e6943-c100-4ac6-8de0-ba498c8a85e5.png)

Model Building:
In this project the ResNet50V2 pre-trained model is used. This model is based on transfer learning mechanism. The process of transfer learning is shown in the flow chart below. Learning model 1 transmits the information it learned (weights, biases), which learning model 2 can use.

RESULTS AND DISCUSSION:  
There are 25,732,668 different model parameters in all. Although the output layer is tailored for our project, the pre-trained input layers are retained as-is. The output layer is initially constructed using the activation function "RELU," followed by the removal of certain unnecessary columns and modification using the activation function "Softmax." The model is finished by coupling the customized output layer to the pre-trained input layers

![image](https://user-images.githubusercontent.com/105503752/230727433-786deb8e-d163-47c2-9deb-595af1cd3de2.png)
![image](https://user-images.githubusercontent.com/105503752/230727445-fb62fbb1-7e8e-4925-9cc7-5db87c567b3b.png)

CONCLUSION:
When simply training the fully linked layers, it is advantageous to remove bottleneck features to save time. This paper presents the improvement of dog breed classification from the convolutional neural network. The experiment showed that using CNN with transfer learning achieves a significant accuracy of 86.24%.Therefore, it is proven that CNN could be used on the dog breed classification.

The supplementary information for our research is as follows: 
- Increase the amount of data used to train the dog breed and identification model.
- Attempt to fine-tune the model using several transfer learning techniques.
- Try to increase the architecture's layer count.










