from numpy import random, sort
import matplotlib.pyplot as plt

# target: office prices
y = random.normal(50000, 10000, 100)
y = sort(y)

# feature: office sizes
x = random.normal(120, 25, 100)
x = sort(x)

# learned values after gradient descent
gradient = 0
y_intercept = 0

lr = 0.0000000001 # learning rate
ep = 5000 # epochs

def mean_squared_error(y, y_hat, n):

    squared_errors_sum = 0

    # calculate sum of squared errors
    for i in range(n):
        squared_error = (y[i] - y_hat[i]) ** 2
        squared_errors_sum += squared_error
    
    # calculate mean squared error
    mse = squared_errors_sum / n

    return mse

def partial_derivative_m(x, y, y_hat, n):
  
    errors_sum = 0

    # calculate sum of product of error and -x
    for i in range(n):
        error = -x[i] * (y[i] - y_hat[i])
        errors_sum += error
    
    # calculate final value of partial derivative
    pd_m = errors_sum * (2 / n)

    return pd_m

def partial_derivative_c(x, y, y_hat, n):
    
    errors_sum = 0

    # calculate sum of product of error and -1
    for i in range(n):
        error = -(y[i] - y_hat[i])
        errors_sum += error
    
    # calculate final value of partial derivative
    pd_c = errors_sum * (2 / n)

    return pd_c

def gradient_descent(x, y, learning_rate, epochs):
    
    # starting with arbitrary values of m and c
    m = 120
    c = 18000

    # number of samples
    n = len(x)

    # list of errors with each epoch
    mse_list = []

    for i in range(epochs):
        print('epoch' + str(i))
        # get predictions (y-hats)
        # list of predictions
        y_hat = []
        for j in range(n):
            pred = (m * x[j]) + c
            y_hat.append(pred)
        
        y_hat = sort(y_hat)
    
        # print('=======Y_HAT AT EPOCH ' + str(i) + '=======\n' + str(y_hat))
        # get MSE
        mse = mean_squared_error(y, y_hat, n)
        mse_list.append(mse)
        print('MSE at epoch ' + str(i) + ' = ' + str(mse_list[i]))

        # adjust m and c
        m = m - learning_rate * partial_derivative_m(x, y, y_hat, n) # new m
        print('next m: ' + str(m))
        c = c - learning_rate * partial_derivative_c(x, y ,y_hat, n) # new c
        print('next c: ' + str(c))

        global gradient
        gradient = m

        global y_intercept
        y_intercept = c
    
    # plot(x, y_hat)

def plot(x, y_hat):

    plt.scatter(x, y_hat)
    plt.plot([min(x), max(x)], [min(y_hat), max(y_hat)], color='red')
    plt.show()

def predict(office_size):
    gradient_descent(x, y, lr, ep)
    global gradient
    global y_intercept
    office_price = gradient * office_size + y_intercept
    print('Office Price: ' + str(office_price))

gradient_descent(x, y, lr, ep)
predict(100)