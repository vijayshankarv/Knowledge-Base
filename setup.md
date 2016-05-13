---
layout: page
title: Setup
---

## Environment

This tutorial is built around [Caffe](http://caffe.berkeleyvision.org/), and the instructions to install caffe can be found at : [http://caffe.berkeleyvision.org/installation.html](http://caffe.berkeleyvision.org/installation.html)

Given that we will be using a *Transfer Learning* based approach, **the requirement of a GPU is not strict**, and all the instructions will work perfectly well on any modern and general purpose laptop or desktop. If you do not have a GPU on your laptop or desktop, make sure to build Caffe in *CPU only* mode.   

Unfortunately Windows is not officially supported by Caffe, but there are a few unofficial Windows ports of Caffe listed on the above mentioned installation page. Instructions for Windows users to replicate this tutorial will be added soon.  

## Data

The data for the PlantVillage Classification Challenge can be obtained at : [https://www.crowdai.org/challenges/1/dataset_files](https://www.crowdai.org/challenges/1/dataset_files)   

After logging in on the above page, you can download the **Training Data** and the **Test Data** and you should have the corresponding `crowdai_train.tar` and `crowdai_test.tar` files. For replicability of all the instructions across the whole tutorial, we would encourage you to create a folder at `/home/<your_user_name>/plantvillage` and save both of these files inside this newly created folder. Then you can extract both these files by :   

{% highlight bash %}

cd /home/<your_user_name>/plantvillage
tar xvf crowdai_train.tar
tar xvf crowdai_test.tar

{% endhighlight %}


## Validation Set

To be able to get an estimate of how well the model if performing across the whole training, we will take a small subset of the training set and consider it as out validation set. We can do this by using the following python script:

{% highlight python %}
#!/usr/bin/env python

# Note: this script needs to be present at
#   /home/<your_user_name>/plantvillage/create_distribution.py
# and executed from within the
#   /home/<your_user_name>/plantvillage/
# directory

import glob
import os
import random
import shutil

TRAIN_PERCENTAGE = 70

TRAIN_SET = []
VAL_SET = []

#Distribute the files into Training and Validation sets
for _image in glob.glob("crowdai/*/*"):
	className = _image.split("/")[-2]

	# Some fileNames contain spaces, which creates some incompatibility with a preprocessing script shipped with caffe
	# Hence we replace all spaces in the filename with _
	newFileName = _image.split("/")[-1]
	newFileName = newFileName.replace(" ", "_")
	newFilePath = "crowdai/"+className+"/"+newFileName
	shutil.move(_image, newFilePath)



	if random.randint(0,100) < TRAIN_PERCENTAGE:
		TRAIN_SET.append((newFilePath, className.split("_")[-1]))
	else:
		VAL_SET.append((newFilePath, className.split("_")[-1]))

#Write the distribution into a separate text files
try:
	os.mkdir("lmdb")
except:
	pass

f = open("lmdb/train.txt", "w")
for _entry in TRAIN_SET:
	f.write(_entry[0]+" "+_entry[1]+"\n")
f.close()

f = open("lmdb/val.txt", "w")
for _entry in VAL_SET:
	f.write(_entry[0]+" "+_entry[1]+"\n")
f.close()
{% endhighlight %}

This can then be executed by :
{% highlight bash %}
cd /home/<your_user_name>/plantvillage
python create_distribution.py
{% endhighlight %}

At the end of its execution, it will create a folder named `lmdb` with two text files by the name `train.txt` and `val.txt`. Each line in these two text files, correspond to the path to a single image and its corresponding class, and we randomly use `~70%` of the available labelled data as the training set and the rest as the validation set. The distribution can be changed simply by changing the `TRAIN_PERCENTAGE` variable in the above script.

## Preprocessing

As we are trying to fine-tune an AlexNet model, we will have to use input images of the exact same size  as was used to train the said model. AlexNet was trained on images of size `256x256` pixels with randomized/central crop of `227x227` pixels which was eventually fed into the network. As the images in our dataset have varied image sizes, we will "squash" them all to `256x256` pixels before we feed into the adapted AlexNet architecture that we want to fine-tune. Apart from that, instead of having to deal with all images straight from the disk, we will store them in [LMDB](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database) which is a high performance embedded transactional database. While Caffe does supports reading images directly from the disk, using LMDB as the datastore has quite significant performance gains.

Caffe ships with a utility to quickly convert images on disk into LMDB. To convert our training and validation sets, we will need to do :
{% highlight bash %}
cd /home/<your_user_name>/plantvillage

$CAFFE_ROOT/build/tools/convert_imageset \
    --resize_height 256 \
    --resize_width 256 \
    --shuffle \
    /home/mohanty/plantvillage/ \
    lmdb/train.txt \
    lmdb/train_lmdb`

$CAFFE_ROOT/build/tools/convert_imageset \
    --resize_height 256 \
    --resize_width 256 \
    --shuffle \
    /home/mohanty/plantvillage/ \
    lmdb/val.txt \
    lmdb/val_lmdb
{% endhighlight %}

which should spit out something along these lines : [http://pastebin.com/ptymwZDm](http://pastebin.com/ptymwZDm)

**NOTE:** `$CAFFE_ROOT` is the environment variable which should point to your caffe installation root. If the `bin` folder of your Caffe installation is in your system path, you can also simply try `convert_imageset` instead of `$CAFFE_ROOT/build/tools/convert_imageset`   

A quick guide to some other features of the `convert_imageset` utility can be found [here](http://stackoverflow.com/questions/31427094/guide-to-use-convert-imageset-cpp#answer-31431716  
).
