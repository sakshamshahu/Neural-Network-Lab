import numpy as np

def Adaline(Input, Target, lr=0.2, stop=0.001):
    weight = np.random.random(Input.shape[1])
    bias   = np.random.random(1)

    Error=[stop +1]
    # check the stop condition for the network
    while Error[-1] > stop or Error[-1]-Error[-2] > 0.0001:
        error = []
        for i in range(Input.shape[0]):
            Y_input = sum(weight*Input[i]) + bias

            # Update the weight
            for j in range(Input.shape[1]):
                weight[j]=weight[j] + lr*(Target[i]-Y_input)*Input[i][j]

            bias=bias + lr*(Target[i]-Y_input)

            error.append((Target[i]-Y_input)**2)
        # Store sum of square errors
        Error.append(sum(error))
        print('Error :',Error[-1])
    return weight, bias

# Input dataset
x = np.array([[1.0, 1.0, 1.0],
              [1.0, -1.0, 1.0],
              [-1.0, 1.0, 1.0],
              [-1.0, -1.0, -1.0]])
# Target values
t = np.array([1, 1, 1, -1])

w,b = Adaline(x, t, lr=0.2, stop=0.001)
print('weight :',w)
print('Bias :',b)

# Predict from the evaluated weight and bias of adaline
def prediction(X,w,b):
    y=[]
    for i in range(X.shape[0]):
        x = X[i]
        y.append(sum(w*x)+b)
    return y
prediction(x,w,b)

