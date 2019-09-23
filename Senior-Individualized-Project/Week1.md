# SIP Journal Entry  -- Week 1

### What I worked on this week:

- I built a neural network for analyzing the cookie recipes. 
- How I built: 
  - The neural network included **a sigmoid calculation** to update the **weights** so that the multiple layers of nodes can apply them when they are training the network. 
  - The main method of training the neural network is operated by the **forward_pass_train()** method, and the **backpropagation()** method is re-evaluating the error terms to minimize the hidden layer’s contribution to the error.
      - *forward propagation*: each layer of our network calculates the outputs for each neuron. All of the outputs from one layer become inputs to the neurons on the next layer. 
      - *backpropagation* : weights to propagate signals forward comes from the input to the output layers in a neural network. I used the weights to propagate error backwards from the output back into the network to update our weights. 
  -  The hyperparameters that I used in this neural network I built are: the learning rate, the number of hidden units, and the number of training passes.
  - This network has two layers, a **hidden layer** and an **output layer**: I used the sigmoid function for activations in the hidden layer, and the regression for the output layer since the output layer has only one node, and the output of the node is the same as the input of the node. 
      -  A function that takes the input signal and generates an output signal, but takes into account the threshold, is called an activation function. 
  
  <img width="416" alt="Screen Shot 2019-09-22 at 11 52 35 PM" src="https://user-images.githubusercontent.com/46575719/65401123-12186e00-dd94-11e9-9443-21fe45326a95.png">


- In this project, I will build a pipeline in the end, based on the neural network. This pipeline will process real-world, user-supplied images — in my SIP, the cookie images. Given an image of a cookie, my algorithm would identify an estimate of the cookie’s type.


### Milestones to celebrate, or Obstacles encountered:

- Because the neural network has multi-layered nodes, it was not easy to consider proper way of calculating the potential error terms. 
- The concept of forward_pass_train and backpropagation would be a good contribution to deal with minimizing the error terms. 


### What I gained from my peer reviewer (Name):



### Plan for next week:

- Data Gathering 
    — I will crawl all search results for a queried dish; in my SIP, it would be cookie recipes. I will determine one particular website that I will use for the recipe scheme, among several websites. 
