---
layout: page
title: Approach
---

<!-- <p class="message">
  Sample Message
</p> -->
<!-- $$a^2 + b^2 = c^2$$ -->

We will be approaching this problem of Image Classification by using ideas around Deep Learning Architectures explored in the recent years in context of the [ImageNet Large Scale Visual Recognition Challenge](http://www.image-net.org/challenges/LSVRC/)(*ILSVRC*),
where the participants attempt to classify a large number of images from the [ImageNet dataset](http://image-net.org/) among 1000 classes.

  We will in particular be focussing on [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-w), the winning solution in ILSVRC 2012.

  ![AlexNet Visualization](http://image.slidesharecdn.com/pydatatalk-150729202131-lva1-app6892/95/deep-learning-with-python-pydata-seattle-2015-35-638.jpg?cb=1438315555)

  This tutorial is more meant to be a hands on guide, so we will not get into all the details and basic concepts that fuel Deep Learning Architectures like that of AlexNet, but if you are curious and want to develop a proper intuition behind many of these ideas, we would strongly recommend [Michael Nielsen](http://michaelnielsen.org/)'s free online book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/).    

As a quick summary, you would notice in the Image above, most deep learning architectures have multiple layers of neurons stacked one after another. The input layers are formed by the raw pixel values obtained from the image, and the final layer gives a probability distribution across all the classes. The intermediate layers use a "processed version" of the output of the previous layer as their input, and over the whole training period they learn to activate against more and more complex features depending on how deep they are in the overall architecture. For example, the earlier layers might learn to activate against basic edges and textures in the image; the layers after that build up on that information to activate against more complex features like say the presence of facial features like nose, eyes, etc; and so on, until the final layer uses these high level features to obtain a probability distribution across all the classes.

The training of architectures like these are obviously computationally very intensive, and usually take multiple weeks when training on huge datasets like that of ImageNet. But the features learnt by the earlier layers turn out to be pretty generic in nature for multiple Image Classification problems, and hence in principle can be easily reused when attempting an Image Classification problem on a different dataset of images spread over completely different classes. This approach is more commonly referred to as [Fine Tuning or Transfer Learning](https://en.wikipedia.org/wiki/Inductive_transfer), where we take an already trained model, and use the learnt weights from its earlier layers as the starting point for training a similar model after slightly modifying the architecture based on the requirements of the new problem.   

A closer look at the image above helps us notice that the last layer in the AlexNet architecture is of size 1000. The output of the last layer is used to determine the final class of an image, and hence it makes sense that it is exactly equal to the number of classes in ImageNet for which the AlexNet architecture was designed. In the PlantVillage Classification Challenge, we have a total of *38 classes*, so our adapted version of AlexNet of course needs to have a size of 38 instead of 1000.

As a part of this tutorial, what we will do is that we will start with a model that was trained on the ImageNet dataset using the AlexNet architecture. We will resize the last layer from 1000 to 38 to correspond with the number of classes we have in our dataset; and then as the last layer predicts based on the output of the penultimate layer, we will simply reset the weights of the penultimate layer in the trained model, and *resume* training the model again in this new configuration by using the PlantVillage dataset.   
