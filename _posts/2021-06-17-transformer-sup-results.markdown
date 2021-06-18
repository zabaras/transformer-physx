---
layout: post
title:  Paper Supplementary Results 
date:   2021-06-17 12:00:00 +0800
categories: results
tags: results
img: /assets/images/posts/post_2.jpg
author: nick
describe: Additional visualizations from the numerical exampled tested in the journal paper.
---

## Supplementary Results 

As a part of the the original [paper](https://arxiv.org/abs/2010.03957), animated visualizations for each of the numerical examples are provided. For additional details on each numerical example, please refer to the paper.

#### Lorenz System
Prediction of ordinary differential equations from any intial state.
<center>
<video width="95%" height="200" controls>
  <source src="{{"/assets/images/results/lorenz0.mp4" | prepend: site.baseurl}}" type="video/mp4">
  <source src="{{"/assets/images/results/lorenz0.webm" | prepend: site.baseurl}}" type="video/webm">
  Your browser does not support the video tag.
</video>
</center>

#### 2D Navier-Stokes System
Transformer with Koopman based embeddings (Transformer-KM) is able to out perform embedding methods and ConvLSTM for prediction of flow around a cylinder at different Reynolds numbers.
<center>
<figure>
<video width="95%" height="200" controls>
  <source src="{{"/assets/images/results/cylinder0.mp4" | prepend: site.baseurl}}" type="video/mp4">
  <source src="{{"/assets/images/results/cylinder0.webm" | prepend: site.baseurl}}" type="video/webm">
  Your browser does not support the video tag.
</video>
<figcaption>Reynolds number 133</figcaption>
</figure>
</center>
<center>
<figure>
<video width="95%" height="200" controls>
  <source src="{{"/assets/images/results/cylinder1.mp4" | prepend: site.baseurl}}" type="video/mp4">
  <source src="{{"/assets/images/results/cylinder1.webm" | prepend: site.baseurl}}" type="video/webm">
  Your browser does not support the video tag.
</video>
<figcaption>Reynolds number 433</figcaption>
</figure>
</center>

#### 3D Reaction Diffusion System
Prediction of 3D reaction diffusion system with random initial condition.
<center>
<video width="95%" height="400" controls>
  <source src="{{"/assets/images/results/grayscott0.mp4" | prepend: site.baseurl}}" type="video/mp4">
  <source src="{{"/assets/images/results/grayscott0.webm" | prepend: site.baseurl}}" type="video/webm">
  Your browser does not support the video tag.
</video>
</center>

<center>
<video width="95%" height="400" controls>
  <source src="{{"/assets/images/results/grayscott1.mp4" | prepend: site.baseurl}}" type="video/mp4">
  <source src="{{"/assets/images/results/grayscott1.webm" | prepend: site.baseurl}}" type="video/webm">
  Your browser does not support the video tag.
</video>
</center>