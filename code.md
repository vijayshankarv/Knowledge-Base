---
layout: page
title: Code
---


## Setting up the Model

A pretrained AlexNet model along with the corresponding prototxt files for caffe are available at : [https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)

We download all the required files and store them in a separate folder named `AlexNet`

{%highlight bash %}
cd /home/<your_user_name>/plantvillage
mkdir AlexNet
cd AlexNet
wget http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
wget https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt
wget https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/solver.prototxt
wget https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/train_val.prototxt 
{% endhighlight %}

## Training
--Start Training

## Prediction
--Predict and submit output file
