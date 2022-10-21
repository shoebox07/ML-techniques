# Lasso Regression

Lasso Regression is a regularized regression technique that takes advantage of both sparsity and convexity. We choose a Beta that minimizes the following:

β̂ = argmin(β) $\frac{1}{2n}  \sum \limits _{i=1} ^{n} (Y_{i} - \beta^T X_{i})^2 + \lambda \lvert\lvert \beta \rvert\rvert_{1}$

We select $\lambda$ by using leave-one-out cross-validation to measure the risk and choose the $\lambda$ that gives us the smallest risk. Afterwards, we refit the model with the non-zero coefficients using linear regression. 

Load the libraries.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import Lasso
%matplotlib inline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
```

Read in the data


```python
#X, y = pd.read_pickle('') #Uncomment and put in the file location
```

Standarize the data.


```python
ys = y - np.mean(y) #Standarize the variables
Xs = (X - np.mean(X, axis=0))/ np.std(X, axis=0)
```

Plot a graph to see how the coefficients shrink at Lambda increases. 


```python
alphas = np.linspace(0.01,1,10000) #Initalize the lassos
lasso = Lasso() 
coefs = []

for a in alphas: #Iterate through each alpha
    lasso.set_params(alpha=a) #Set the lasso with the current alpha
    lasso.fit(Xs, ys) #Fit the lasso
    coefs.append(lasso.coef_) #Add coefs to coef

plt.figure(figsize=(10,8))    
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('lambda')
plt.ylabel('Standardized Coefficients')
plt.title('Lasso coefficients vs. regularization lambda');
```

Function that will store all of the risk values for each lambda


```python
alphas = np.linspace(0.001, 1, 1000) #this is an array of lambdas to test
lasso = Lasso() #call lasso
riskvector = []#store the risk for each lambda in here

for a in alphas: #iterate through each lambda
    lasso.set_params(alpha=a) #set the lambda for the lasso
    lasso.fit(Xs,ys) #fit the model

    S_hat = [] #create a list that will store the coefficents that aren't zero
    for i in range(lasso.coef_.shape[0]): #iterate through each coefficient
        if lasso.coef_[i] != -0.: #check to see if the coefficient isn't zero
            S_hat.append(i) #append it to S_hat if it's not zero
    S = len(S_hat) #create a new variable S and it will be equal to the length of S_hat
    #S_hat has all of the non-zero coefficents so the length of it is the total number of non-zero coefs
    
    if S == 0: #if all coefficients are zero, break out of the for loop
        break #Recall that as lambda goes to infity, the coefficients approach 0
        #so we break out of the loop since from this alpha onwards the coefs will be zero
    
    selected_var = Xs[:, S_hat] #Subset of the variables whose coefficients aren't zero
    
    linear = LinearRegression() #call the linear regression function
    linear.fit(selected_var, ys) #fit the model with the non-zero cofficients

    ypre = linear.predict(selected_var) #predictions
    
    risk = ((1/n)*(np.sum((ys - ypre)**2)))/((1-(S/n))**2)
    #calculate the risk using the formula on slide 15 on september 7th lecture
    #this is basically a formula for leave-one-out cross-validation
    
    riskvector.append(risk) #add the risk to riskvector
    
```

From here you can choose the lambda that gives you the lowest risk and refit the model with the non-zero coefficients using linear regression. 


```python

```
