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
    mean_file: "../lmdb/mean.binaryproto"
  }
  data_param {
    source: "../lmdb/train_lmdb"
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
    mean_file: "../lmdb/mean.binaryproto"
  }
  data_param {
    source: "../lmdb/val_lmdb"
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

The final `solver.prototxt` should look like :

{% highlight prototxt %}
net: "train_val.prototxt"
test_iter: 3
test_interval: 59
base_lr: 0.001
lr_policy: "step"
gamma: 0.1
stepsize: 590
display: 11
max_iter: 1770
momentum: 0.9
weight_decay: 0.0005
snapshot: 59
snapshot_prefix: "../snapshots/snapshots_"
solver_mode: GPU
{% endhighlight %}

**NOTE:** The `solver_mode` parameter should be set to `CPU` if you do not have access to a GPU on the host machine. Apart from that, these parameters are mostly hyperparameters which you will need to hand tune a bit till you are confident you get the best results.

we will also need to create a folder called as `snapshots` where caffe can dump the models at certain intervals. Based on the reference we gave in our `solver.prototxt`, we will have to create it at : `/home/<your_user_name>/plantvillage/snapshots`, so we should simply do a :

{% highlight bash %}
mkdir /home/<your_user_name>/plantvillage/snapshots
{% endhighlight %}

## Training

If you followed all the previous steps correctly, then you should be able to start the training simply by :

{% highlight bash %}
cd /home/<your_user_name>/plantvillage/AlexNet
$CAFFE_ROOT/build/tools/caffe train \
      -solver solver.prototxt \
      -weights bvlc_reference_caffenet.caffemodel
      -gpu 0 #Only if you have a GPU, else you should ignore this flag
{% endhighlight %}

If you are running the training in GPU mode, and you get an `out of memory` error, then you can try reducing the training and testing `batch_size` in `train_val.prototxt` in line_number `17` and `36`.

## Prediction

The first step is to select the model that we will use to predict. In the `snapshots` folder, you will find many files of the form `snapshots__<iteration_number>.caffemodel` or `snapshots__<iteration_number>.solverstate`. The `*.solverstate` files are used to "resume" the training from a particular state, while the `*.caffemodel` files is primarily used for prediction, but it is pretty much the same as the solverstate file minus some training specific state variables. So we will select the latest model from among the snapshots, and use that in the following sub section for prediction all the test images. You can do that by :

{% highlight bash %}
cd /home/<your_user_name>/plantvillage/
cp snapshots/`ls -t snapshots/ | head -n 1` plantvillage.caffemodel
{% endhighlight %}

Please note that, if you want to use the snapshot from any other iteration that the latest iteration, you can also do something like `cp snapshots/snapshots__<iteration_number>.caffemodel plantvillage.caffemodel`

And now we move to the actual predictions of all the test images.
Before we can predict the class of the images, we will have to first make them similar to the kind of images we used for training. Before we initiated the training, we "squashed" the images to `256x256 pixels`, and now we do the same to all the test images. You can do that by creating the following script at `/home/<your_user_name>/plantvillage/resize_test_images.sh`:
{% highlight bash %}
#!/bin/bash

TEST_FOLDER_NAME="/home/<your_user_name>/plantvillage/test"
for file in `ls $TEST_FOLDER_NAME`
do
echo $file
convert $TEST_FOLDER_NAME/$file -resize 256x256! $TEST_FOLDER_NAME/$file
done
{% endhighlight %}

Then you can execute it by :
{% highlight bash %}
cd /home/<your_user_name>/plantvillage
chmod +x resize_test_images.sh
./resize_test_images.sh
{% endhighlight %}

Now that we have everything in order, we can use the following script to get started on how to make the predictions and put it in an appropriate format for the crowdAI PlantVillage Classification Challenge. This script has to exist at the location `/home/<your_user_name>/plantvillage/predict.py`.

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob

"""
# You can use the commented block of code below to
$ make sure that caffe is on the python path:
# This takes the path to caffe_root from the environment variable, so make sure
# the $CAFFE_ROOT environment variable is set
#
#
caffe_root = os.environ['CAFFE_ROOT']
import sys
sys.path.insert(0, caffe_root + 'python')
"""

import caffe

"""
Adapted from from : http://www.cc.gatech.edu/~zk15/deep_learning/classify_test.py
"""

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = 'AlexNet/deploy.prototxt'
PRETRAINED = 'plantvillage.caffemodel'
BINARY_PROTO_MEAN_FILE = "lmdb/mean.binaryproto"

"""
Replicated from https://github.com/BVLC/caffe/issues/290
"""
blob = caffe.proto.caffe_pb2.BlobProto()
data = open( BINARY_PROTO_MEAN_FILE  , 'rb' ).read()
blob.ParseFromString(data)
mean_arr = np.array( caffe.io.blobproto_to_array(blob) )[0]


##NOTE : If you do not have a GPU, you can uncomment the `set_mode_cpu()` call
#        instead of the `set_mode_gpu` call.
caffe.set_mode_gpu()
#caffe.set_mode_cpu()


net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=mean_arr.mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

f = open("output.csv", "w")
f.write("filename,c_0,c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8,c_9,c_10,c_11,c_12,c_13,c_14,c_15,c_16,c_17,c_18,c_19,c_20,c_21,c_22,c_23,c_24,c_25,c_26,c_27,c_28,c_29,c_30,c_31,c_32,c_33,c_34,c_35,c_36,c_37\n")

number_of_files_processed = 0
for _file in glob.glob("./test/*"):
	number_of_files_processed += 1
	FileName = _file.split("/")[-1]
	input_image = caffe.io.load_image(_file)
	prediction = net.predict([input_image])
	s = FileName+","
	for probability in prediction[0]:
		s+=str(probability)+","
	s = s[:-1]+"\n"
	f.write(s)
	print "Number of files : ", number_of_files_processed
	print 'predicted class:', prediction[0].argmax()
	print "**********************************************"
{% endhighlight %}

and then execute it by

{% highlight bash %}

cd /home/<your_user_name>/plantvillage
python predict.py

{% endhighlight %}

Finally after this script executes successfully (this will take some time ;) So, be patient !! ), you should have a `output.csv` in the format that the CrowdAI PlantVillage Classification Challenge expects. **But please note that you will be disqualified if you use this approach to make a submission, as FineTuning / Transfer Learning based approaches are not allowed according to the rules of the challenge.**
