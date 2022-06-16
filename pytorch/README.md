## What's in this Fork? 

This repo is a fork of "Dive into Deep Learning" excercise. The repo contains an example of Linear regression using PyTorch and a simple way to Dockerize it. 

to use this fork please cd to pytorch director and execute main.py

Rest of the step shows how to Dockerize the script. 
Details of the implementation can be found in this blog: https://www.docker.com/blog/how-to-train-and-deploy-a-linear-regression-model-using-pytorch-part-1/

## Dockerizing Linear Regression solution using PyTorch and Docker


## Step 1. Clone the repository

```
git clone https://github.com/shanktpmm/deeplearning-docker
```

## Step 2. Build a Docker Image


```
cd deeplearning-docker/pytorch
docker build -t linear_regression .
```

## Step 3. Run the Docker container


```
docker run linear_regression
```

## Results:

```
features: tensor([ 0.8192, -0.9030])
label: tensor([8.9123])
epoch 1, loss 0.000183
epoch 2, loss 0.000098
epoch 3, loss 0.000095
epoch 4, loss 0.000095
epoch 5, loss 0.000095
error in estimating m: tensor([4.0054e-05, 9.6369e-04])
error in estimating c: tensor([-0.0003])
```

