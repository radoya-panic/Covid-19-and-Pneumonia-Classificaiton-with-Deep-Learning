# Project Requirements
1. [ ] If you take a look at Covid19 dataset, you’ll see that there are two different folders inside. Take some time to look around these directories and familiarize yourself with the images inside. As you peruse through them, think about the following:

> What types of images are we working with?
How are they organized? How will this affect preprocessing?
Will this be binary classification or multi-class classification?
After you do this, you will be ready to get started on preprocessing! Click the hint below if you want to see some insights on the image data. 

<details>
<summary>Step 1 Hint</summary>
<br>
One of the first things you may notice is that the folders are split into train and test folders. This is immensely helpful for us during preprocessing as images have already been allocated and prepared for a learning model.

When you click in and view the images, you will notice that all the X-ray scans are in grayscale. This is something to keep in mind when you dive into preprocessing the data.

Finally, when you click into either the train or test folder, you will see that there are three different folders within them:

Covid
Normal
Viral Pneumonia
These are each a different class that our learning model can output, indicating a multi-class classification problem rather than binary classification.
</details>

2. [ ] Load in your image data and get it ready for the journey through a neural network. One possible way to do this is to use an ImageGenerator object; however, feel free to try other methods you may have experienced before this project. When creating the object, remember that neural networks struggle with large integer values. Think about how you might want to get your image data ready for your neural network and get the best results.


<details>
<summary>Step 2 Hint</summary>
<br>
If you decide to create an ImageGenerator object, check out TensorFlow’s preprocessing.image module.

You will also want to rescale your images with pixel normalization.

If you look at the documentation, ImageDataGenerator() contains numerous parameters, and you can play with them as you work on your neural network. If you flip or crop your images during data augmentation, note that you would have to create separate ImageDataGenerator() objects for both for your training and validation data to avoid hurting the performance on your test set.
</details>

3. [ ] Now that you have set up your ImageDataGenerator object, it is time to actually load in your image data. You will want to create two different iterable objects from this ImageDataGenerator: a train set and a test/validation set. When you are creating these sets of images consider the following:

The directory the images come from
The types of images you are working with
The target size of the images
Click the hint below if you need any other guidance.
<details>
<summary>Step 3 Hint</summary>
<br>
o split the data into a training set and a test/validation set, create two different iterable objects that can fit right into the network model: a training_iterator and a validation_iterator

Use the flow_from_directory() method in the tensorflow.keras.preprocessing.image module.

You can set your class_mode to sparse and use sparse categorical loss instead for this model as well.

You can play with the batch_size parameter as you fine-tune your model.
</details>

4. [ ] Now that your image data is loaded and ready for analysis, create a classification neural network model to perform on the medical data.With image data, there are a ton of directions to go in. To get you grounded, we recommend you start by creating your input layer and output layer and compile it before diving into creating a more complex model. When starting your neural network, consider the following: The shape of your input The shape of your output, Using any activation functions for your output, Your gradient descent optimizer, Your learning rate, Your loss functions and metrics, Flattening the image data before the output layer

<details>
<summary>Step 4 Hint</summary>
<br>
You should create a sequential model and be mindful of the shape of the image data that is being loaded into your neural network.

Next, you need to flatten your image data and create your output layer. In this case, you have three different classes. Because of this, you will need a dense layer with three different potential outputs.

Finally, when compiling your model, you have a lot of choices to make, including optimizer, learning rate, loss function, and metric evaluations.
</details>

5. [ ] It’s time to test out the model you created! Fit your model with your training set and test it out with your validation/test set.Since you have not added many layers yet or played around with hyperparameters, it may not be very accurate yet. Do not fret! Your next adventure will be to play with your model and mold it until you see more ideal results.

<details>
<summary>Step 5 Hint</summary>
<br>
To fit your model use the .fit() function from the tensorflow.keras.Model module. Be sure to fit it on your training data and test it with your validation data. You can also specify how many steps are taken during each epoch as well as how many epochs your model performs.

Each of these functions has many different required parameters (as well as optional ones). If you need a refresher on these, look at the sklearn documentation.
</details>

6. [ ] You have created a model and tested it out. Now it is time for the real fun! Start playing around with some hidden layers and hyperparameters. When adding hidden layers, consider the type of hidden layers you add (remember this is image data). As you add in layers, you should also adjust your model’s hyperparameters. You have a lot of choices to make. You can choose: the number of epochs, The size of your batch_size, to add more hidden layersyour type of optimizer and/or activation functions, the size of your learning rate, Have fun in the hyperparameter playground. Test things out and see what works and what does not work. See what makes your model optimized between speed and accuracy. You have complete creative power!
<details>
<summary>Step 6 Hint</summary>
<br>
Note: We suggest being thoughtful about the number of layers you add. Given the small size of the dataset, you should not train it on anything more than 10K parameters or roughly five layers. This will help you avoid overfitting the data as well as avoid crashing the learning environment. If you attempt to use Dense layers rather than Conv2D layers in the learning environment, your program will likely not successfully run to completion on the platform.

Some suggestions:

Use convolutional layers as well as max pooling.
Use EarlyStopping() to make your model efficient.
Play around with batch_size to optimize model efficiency with accuracy.
</details>

7. [ ] Great work! Visit our forums to share your project with other learners. We recommend hosting your own solution on GitHub so you can share it more easily. Your solution might look different from ours, and that’s okay. There are multiple ways to solve these projects, and you’ll learn more by seeing others’ code.

## Extensions

8. [ ] Plot the cross-entropy loss for both the train and validation data over each epoch using the Matplotlib Library. You can also plot the AUC metric for both your train and validation data as well. This will give you an insight into how the model performs better over time and can also help you figure out better ways to tune your hyperparameters. Because of the way Matplotlib plots are displayed in the learning environment, please use fig.savefig('static/images/my_plots.png') at the end of your graphing code to render the plot in the browser. If you wish to display multiple plots, you can use .subplot() or .add_subplot() methods in the Matplotlib library to depict multiple plots in one figure.

Use the hint below if you have any struggles with displaying these graphs.

<details>
<summary>Step 8 Hint</summary>
<br>
# plotting categorical and validation accuracy over epochs

fig = plt.figure()


ax1 = fig.add_subplot(2, 1, 1)

ax1.plot(history.history['categorical_accuracy'])

ax1.plot(history.history['val_categorical_accuracy'])

ax1.set_title('model accuracy')

ax1.set_xlabel('epoch')

ax1.set_ylabel('accuracy')

ax1.legend(['train', 'validation'], loc='upper left')

 
# plotting auc and validation auc over epochs

ax2 = fig.add_subplot(2, 1, 2)


ax2.plot(history.history['auc'])

ax2.plot(history.history['val_auc'])

ax2.set_title('model auc')

ax2.set_xlabel('epoch')

ax2.set_ylabel('auc')

ax2.legend(['train', 'validation'], loc='upper left')

 
# used to keep plots from overlapping

fig.tight_layout()
 
fig.savefig('static/images/my_plots.png')

</details>

9. [ ] Another potential extension to this project would be implementing a classification report and a confusion matrix. These are not tools we have introduced you to; however, if you would like further resources to improve your neural network, we recommend looking into these concepts. As a brief introduction, these concepts evaluate the nature of false positives and false negatives in your neural network taking steps beyond simple evaluation metrics like accuracy. In the hint below, you will see a possible solution to calculate a classification_report and a confusion_matrix, but you will need to do some personal googling/exploring to acquaint yourself with these metrics and understand the outputs.


<details>
<summary>Step 9 Hint</summary>
<br>
test_steps_per_epoch = numpy.math.ceil(validation_iterator.samples / validation_iterator.batch_size)

predictions = model.predict(validation_iterator, steps=test_steps_per_epoch)

test_steps_per_epoch = numpy.math.ceil(validation_iterator.samples / validation_iterator.batch_size)

predicted_classes = numpy.argmax(predictions, axis=1)

true_classes = validation_iterator.classes

class_labels = list(validation_iterator.class_indices.keys())

report = classification_report(true_classes, predicted_classes, target_names=class_labels)

print(report)   
 
cm=confusion_matrix(true_classes,predicted_classes)

print(cm)
</details>