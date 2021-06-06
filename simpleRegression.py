from numpy import random, polyfit
import matplotlib.pyplot as plt

# target: office prices
y = random.normal(50000, 15000, 100)

# feature: office sizes
x = random.normal(120, 50, 100)

# learned values after gradient descent
gradient = 0
y_intercept = 0

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
    m = 12
    c = 500

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
            y_hat.append(m * x[j] + c)
    
        # print('=======Y_HAT AT EPOCH ' + str(i) + '=======\n' + str(y_hat))
        # get MSE
        mse = mean_squared_error(y, y_hat, n)
        mse_list.append(mse)
        print('MSE at epoch ' + str(i) + ' = ' + str(mse[i]))

        # adjust m and c
        m = m - learning_rate * partial_derivative_m(x, y, y_hat, n) # new m
        print('next m: ' + str(m))
        c = c - learning_rate * partial_derivative_c(x, y ,y_hat, n) # new c
        print('next c: ' + str(c))

        global gradient
        gradient = m

        global y_intercept
        y_intercept = c

def plot(x, y_hat):

    # m, c = polyfit(x, y, 1)
    # print(str(m) + ', ' + str(c))
    # plt.plot(x, y, 'o')
    # plt.plot(x, m * x + c)
    plt.scatter(x, y_hat)
    plt.plot([min(x), max(x)], [min(y_hat), max(y_hat)], color='red')
    plt.show()

# plot(x, y)

def predict(office_size):
    gradient_descent(x, y, 0.0001, 10)
    global gradient
    global y_intercept
    office_price = gradient * office_size + y_intercept
    print('Office Price: ' + str(office_price))

# gradient_descent(x, y, 0.003, 50)
# predict(100)