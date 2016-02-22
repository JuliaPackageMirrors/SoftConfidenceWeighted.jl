# SoftConfidenceWeighted.jl
This is an online supervised learning algorithm which utilizes the four salient properties:

* Large margin training
* Confidence weighting
* Capability to handle non-separable data
* Adaptive margin

The paper is [here](http://arxiv.org/pdf/1206.4612v1.pdf).


## Usage
SCW has 2 formulations of its algorithm which are SCW-I and SCW-II.  
You can choose the algorithm by passing a parameter to `init` like below.  

```
model = init(C, ETA, algorithm = "SCW1")
model = init(C, ETA, algorithm = "SCW2")
```

## Fitting the model
### Using data represented as a matrix
Feature vectors are given as the columns of the matrix X.  

```
model = init(C, ETA, algorithm = "SCW1")
model = fit!(model, X, y)
y_pred = predict(model, X)
```

C and ETA are hyperparameters.  
You can also initialize the model like below.  

```
model = SCW{SCW1}(C, ETA)
model = fit!(model, X, y)
y_pred = predict(model, X)
```

### Using data in a file

```
model = init(C, ETA, algorithm = "SCW1")
model = fit!(model, training_file, ndim)
y_pred = predict(model, test_file)
```

The input files must be in the svmlight format.  
`ndim` is the data dimension.  


### Note
1. This package performs only binary classification, not multiclass classification.
2. Training labels must be 1 or -1. No other labels allowed.
