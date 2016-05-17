---
layout: page
title: Introduction
---

<p class="message">
  <strong>NOTE:</strong> This tutorial uses a "Transfer Learning" or "Fine Tuning" approach to solve
  the image classification problem. This tutorial should be used to assist in development of basic ideas when it comes to approaching this and similar problems. 
  
  Transfer Learning is against the rules of the <a href="https://www.crowdai.org/challenges/1">PlantVillage Classification Challenge</a>, and this tutorial is not intended to "generate" a submission. <strong>All submissions made by using a Transfer Learning approach (as described in the tutorial, or otherwise) will be disqualified.</strong>
</p>

CrowdAI’s educational vision is to become a great open access learning resource
for data analysis and machine learning. To make this happen, we are launching
the CrowdAI Knowledge Base, a place where everyone in the community comes together to build high quality resources to help data scientists at all levels of expertise,
from beginners to experts.

 In this tutorial we will focus on the
[PlantVillage Classification Challenge](https://www.crowdai.org/challenges/1)
hosted on [CrowdAI](https://www.crowdai.org).   

The goal of the challenge is to classify a set of images of plant leaves into 38 possible crop-disease pairs.
Here are a few examples from across all the 38 crop-disease pairs represented in the PlantVillage dataset.   
![PlantVillageDataset](https://s3.amazonaws.com/salathegroup-static/plantvillage/plantvillage-min.png)

In the following sections we will walk through the basic steps of how to get started on this problem, and similar Image Classification problems using [Caffe](http://caffe.berkeleyvision.org/), a very powerful and popular [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning) framework developed by [Berkley Vision and Learning Center](http://bvlc.eecs.berkeley.edu/).   

The [PlantVillage Classification Challenge](https://www.crowdai.org/challenges/1) requires the participants to train a model by using labelled images provided in the Training Set to predict a probability distribution across all the 38 crop-disease pairs (*classes*) for all the images in the Test Set.
