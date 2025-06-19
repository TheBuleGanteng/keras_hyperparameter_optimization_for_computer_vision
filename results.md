Testing process:

The otputs of my tests are shown below. Key learnings were as follows:
- Convolutional layers:
    1. Moving from 1 --> 2 convolutional layers: Massive increase in accuracy
    2. Moving from 2 --> >2 convolutional layers: Little or negative change in accuracy

- Hidden layers:
    1. Moving from 1 --> 2 hidden layers: Negative change in accuracy

- Units per hidden layer:
    1. Moving from 128 --> 512: Negative change in accuracy
    2. Moving from 128 --> 32: Negative change in accuracy

- Filter dimensions:
    1. Moving from 2x2 --> 2x3 (accidental): Negative change in accuracy (skew?)

1 convolutional layer, 32 nodes / convolutional layer, 1 hidden layer, 128 units / hidden layer, dropout = 0.5, output layer activation = "softmax"
324/324 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.0578 - loss: 3.4869


2 convolutional layers, 32 nodes / convolutional layer, 1 hidden layer, 128 units / hidden layer, dropout = 0.5, output layer activation = "softmax"
324/324 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.9714 - loss: 0.1133   


2 convolutional layers, 32 nodes / convolutional layer, 1 hidden layer, 512 units / hidden layer, dropout = 0.5, output layer activation = "softmax"
running main ... Training completed! Now evaluating...
324/324 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9560 - loss: 0.2086 

2 convolutional layers, 32 nodes / convolutional layer, 1 hidden layer, 32 units / hidden layer, dropout = 0.5, output layer activation = "softmax"
running main ... Training completed! Now evaluating...
324/324 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.0593 - loss: 3.4883 


2 convolutional layers, 32 nodes / convolutional layer, 2 hidden layers, 128/64 units / hidden layer, dropout = 0.5/0.3, output layer activation = "softmax"
running main ... Training completed! Now evaluating...
324/324 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9107 - loss: 0.3267   
