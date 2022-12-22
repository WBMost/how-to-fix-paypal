import numpy as np
import copy
import math

def score_text(lines):
    sus_words = {
    'suspicious': 1,
    'cancel': 1,
    'illegal': 1,
    'refund': 1,
    'help desk': 0.5,
    'bitcoin': 0.5,
    'authorized': 0.5,
    '24 hours': 0.25,
    'USD': 0.1,
    'BTC': 0.1
    }
    scores = []
    for index, line in enumerate(lines):
        line_total_score = 0
        for word, score in sus_words.items():
            if word.lower() in line.lower():
                line_total_score += score
        scores.append(line_total_score)
    return scores

def find_cost(lines):
    len(lines)
    values = []
    len(values)
    for index, line in enumerate(lines):
        for x, word in enumerate(line.split()):
            if '$' in word:
                word = word.replace('$','')
                if not isinstance(word[-1],int):
                    word = word[:-1]
                values.append(float(word))
                break
            elif x == len(line.split())-1:               
                values.append(0.)
        print(f'values_len: {len(values)} \t index: {index}')
        if len(values) < index:
            values.append(0.)
    return values

def load_data():
    scams = open('python/algorithms/training_data/scams.txt', 'r', encoding='utf-8').readlines()
    reals = open('python/algorithms/training_data/non_scams.txt', 'r', encoding='utf-8').readlines()
    scam_bool = np.full(len(scams),1,dtype=int)
    real_bool = np.full(len(reals),0,dtype=int)
    res = np.append(scams,reals)
    res = np.vstack((find_cost(res),score_text(res))).T
    bools = np.append(scam_bool,real_bool)
    return res,bools

def sigmoid(z):
    return 1/(1+np.exp(-z))

def compute_cost(X,y,w,b,lambda_=1):
    '''
    logistic regression cost function

    𝐽(𝐰,𝑏)=1𝑚∑𝑖=0𝑚−1[𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),𝑦(𝑖))]
    𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),𝑦(𝑖))=(−𝑦(𝑖)log(𝑓𝐰,𝑏(𝐱(𝑖)))−(1−𝑦(𝑖))log(1−𝑓𝐰,𝑏(𝐱(𝑖)))
    '''
    m, n = X.shape

    loss_sum = 0

    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb += w[j]*X[i][j]
        z_wb += b
        f_wb = sigmoid(z_wb)
        loss = -y[i]*np.log(f_wb)-(1-y[i])*np.log(1-f_wb)
        loss_sum += loss
    return (1/m)*loss_sum

def compute_gradient(X,y,w,b,lambda_=1):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb += w[j]*X[i][j]
        z_wb += b
        f_wb = sigmoid(z_wb)

        dj_db += f_wb-y[i]
        
        for j in range(n):
            dj_dw[j] += (f_wb-y[i])*X[i][j]
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_db/m, dj_dw/m

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    m = len(X)
    
    J_history = []
    w_history = []
    
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        if i<100000:
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history

def predict(X, w, b): 
    m, n = X.shape   
    p = np.zeros(m)

    for i in range(m):   
        z_wb = 0
        for j in range(n):
            z_wb += w[j]*X[i][j]
        z_wb += b
        f_wb = sigmoid(z_wb)
        p[i] = f_wb >= 0.5
    return p

X_train,y_train = load_data()

m,n = X_train.shape

np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2).reshape(-1,1) - 0.5)
initial_b = -8

iterations = 30000
alpha = 0.01

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations, 0)

predictions = predict(X_train, w,b)
print(f'predictions based off training data: {predictions}')