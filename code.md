---
layout: page
title: Code
---


## Setting up the Model

### Downloading pretrained model

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

### Updating LMDB data store references

Now that we have all the required files, we will start with first pointing the `train_val.prototxt` to the correct training and validation lmdb stores (and also the corresponding mean file) . The `train_val.prototxt` file is the configuration file that caffe will refer to understand the structure of the network during the training and the validation phases.

We will do this, by editing the `train_val.prototxt` file to change the following block (`line 7-19`) :

{% highlight prototxt %}
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 227
    mean_file: "data/ilsvrc12/imagenet_mean.binaryproto"
  }
  data_param {
    source: "examples/imagenet/ilsvrc12_train_lmdb"
    batch_size: 256
    backend: LMDB
  }
{% endhighlight %}

to look like :

{% highlight prototxt %}
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 227
    mean_file: "lmdb/mean.binaryproto"
  }
  data_param {
    source: "lmdb/train_lmdb"
    batch_size: 256
    backend: LMDB
  }
{% endhighlight %}

We will do the same thing for the prototxt block corresponding to the validation data, by changing the following block (`line 26-38`):

{% highlight prototxt %}
  include {
    phase: TEST
  }
  transform_param {
    mirror: false
    crop_size: 227
    mean_file: "data/ilsvrc12/imagenet_mean.binaryproto"
  }
  data_param {
    source: "examples/imagenet/ilsvrc12_val_lmdb"
    batch_size: 50
    backend: LMDB
  }
{% endhighlight %}

to look like :

{% highlight prototxt %}
  include {
    phase: TEST
  }
  transform_param {
    mirror: false
    crop_size: 227
    mean_file: "lmdb/mean.binaryproto"
  }
  data_param {
    source: "lmdb/val_lmdb"
    batch_size: 50
    backend: LMDB
  }
{% endhighlight %}

### Adapting the downloaded AlexNet model for FineTuning on our dataset

As the dataset we are working with needs to be classified across 38 classes instead of the standard 1000 classes that AlexNet was designed for; we will first change the number of outputs of the final layer from 1000 to 38. This can be done by manually editing the corresponding section for the last layer (`fc8` in this case) in both the `train_val.prototxt` and `deploy.prototxt`. The `num_output` value in that layer needs to be changed to 38 from 1000. And this can be quickly done by using :

{% highlight bash %}
cd /home/<your_user_name>/plantvillage/AlexNet

sed -i 's/num_output: 1000/num_output: 38/' train_val.prototxt

sed -i 's/num_output: 1000/num_output: 38/' deploy.prototxt
{% endhighlight %}

Then we also need to reset the weights in the last layer of the network, which can be very easily done by renaming the last layer, so that Caffe has to re-initialize the weights of the layer when it does not find any corresponding weights in the associated layer. This can be done by manually renaming all references to `fc8` (the last layer) in both `train_val.prototxt` and `deploy.prototxt` to `fc8_plantvillage`. Or, it can be quickly done by using :

{% highlight bash %}
cd /home/<your_user_name>/plantvillage/AlexNet

sed -i 's/fc8/fc8_plantvillage/' train_val.prototxt

sed -i 's/fc8/fc8_plantvillage/' deploy.prototxt
{% endhighlight %}

### Configuring the Solver Parameters

In the final step before we can start training, we need to configure the solver parameters in `solver.prototxt`. To start with, we should start with a base learning rate of `0.001`. The reason being, as we will be finetuning an already trained model, the model is in principle much ahead in the training phase in contrast to when you try to start training from scratch. Then we will start experimenting by running the training for `30 epochs`, where 1 epoch basically refers to one full pass through the training set.

## Training

If you followed all the previous steps correctly, then you should be able to start the training simply by :

{% highlight bash %}
cd /home/<your_user_name>/plantvillage/AlexNet
$CAFFE_ROOT/build/tools/caffe train \
      -solver solver.prototxt \
      -weights bvlc_reference_caffenet.caffemodel
      -gpu 0 #Only if you have a GPU, else you should ignore this flag
{% endhighlight %}

## Prediction
--TO-DO
