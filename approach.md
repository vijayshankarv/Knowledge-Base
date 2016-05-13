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

  This tutorial is more meant to be a hands on guide, so we will not get into all the details and basic concepts that fuel Deep Learning Architectures like that of AlexNet, but if you are curious and want to develop a proper intuition behind many of these ideas, we would strongly recommend [Michael Nielsen](http://michaelnielsen.org/)'s [free online book on Deep Learning](http://neuralnetworksanddeeplearning.com/).   

As a quick summary, as you would notice in the Image above, most deep learning architectures have multiple layers of neurons stacked one after another. The input layers are formed by the raw pixel values obtained from the image, and the final layer gives a probability distribution across all the classes. The intermediate layers use a "processed version" of the output of the previous layer as their input, and over the whole training period they learn to activate against more and more complex features depending on how deep they are in the overall architecture. For example, the earlier layers might learn to activate against basic edges and textures in the image; the layers after that build up on that information to activate against more complex features like say the presence of facial features like nose, eyes, etc; and so on, until the final layer uses these high level features to obtain a probability distribution across all the classes.
