---
layout: post
title:  "Adadeleta Gradient Descent"
date: 2017-07-06 12:00:00
author: Rohan Kotwani
excerpt: "Least squares parameter estimation using Adadeleta Gradient Descent"
tags: 
- Regression
- Gradient Descent
- Adadelta

---


This blog post investigates adaptive gradient descent methods and compares it to the closed form solution. The adadelta algorithm is run multiple times to improve performance. A new criteria is used to check for model convergence.

#### Index

* Sickit Learn Package
* Adadelta Gradient Descent Algorithm

#### Sckitlearn

This software package can easily minimize least square error between data points by solution the the closed form solution of the gradient's derivative.


{% highlight python %}
import importlib
import Regression
importlib.reload(Regression)
{% endhighlight %}




    <module 'Regression' from '/Users/rohankotwani/Documents/MachineLearning/Regression.py'>




{% highlight python %}
import pandas as pd
import time
import seaborn
import numpy
from matplotlib import pyplot as plt
{% endhighlight %}


{% highlight python %}
train=pd.read_csv("DATA/kc_house_train_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})
test=pd.read_csv("DATA/kc_house_test_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})
{% endhighlight %}


{% highlight python %}
train.tail()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>...</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17379</th>
      <td>7936000429</td>
      <td>20150326T000000</td>
      <td>1007500.0</td>
      <td>4.0</td>
      <td>3.50</td>
      <td>3510.0</td>
      <td>7200</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>9</td>
      <td>2600</td>
      <td>910</td>
      <td>2009</td>
      <td>0</td>
      <td>98136</td>
      <td>47.5537</td>
      <td>-122.398</td>
      <td>2050.0</td>
      <td>6200.0</td>
    </tr>
    <tr>
      <th>17380</th>
      <td>2997800021</td>
      <td>20150219T000000</td>
      <td>475000.0</td>
      <td>3.0</td>
      <td>2.50</td>
      <td>1310.0</td>
      <td>1294</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>1180</td>
      <td>130</td>
      <td>2008</td>
      <td>0</td>
      <td>98116</td>
      <td>47.5773</td>
      <td>-122.409</td>
      <td>1330.0</td>
      <td>1265.0</td>
    </tr>
    <tr>
      <th>17381</th>
      <td>0263000018</td>
      <td>20140521T000000</td>
      <td>360000.0</td>
      <td>3.0</td>
      <td>2.50</td>
      <td>1530.0</td>
      <td>1131</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>1530</td>
      <td>0</td>
      <td>2009</td>
      <td>0</td>
      <td>98103</td>
      <td>47.6993</td>
      <td>-122.346</td>
      <td>1530.0</td>
      <td>1509.0</td>
    </tr>
    <tr>
      <th>17382</th>
      <td>0291310100</td>
      <td>20150116T000000</td>
      <td>400000.0</td>
      <td>3.0</td>
      <td>2.50</td>
      <td>1600.0</td>
      <td>2388</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>1600</td>
      <td>0</td>
      <td>2004</td>
      <td>0</td>
      <td>98027</td>
      <td>47.5345</td>
      <td>-122.069</td>
      <td>1410.0</td>
      <td>1287.0</td>
    </tr>
    <tr>
      <th>17383</th>
      <td>1523300157</td>
      <td>20141015T000000</td>
      <td>325000.0</td>
      <td>2.0</td>
      <td>0.75</td>
      <td>1020.0</td>
      <td>1076</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1020</td>
      <td>0</td>
      <td>2008</td>
      <td>0</td>
      <td>98144</td>
      <td>47.5941</td>
      <td>-122.299</td>
      <td>1020.0</td>
      <td>1357.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



### There are 17383 observations in the train data set!

I suspect this will slow down the gradient descent quite a bit.


{% highlight python %}
train['bedrooms_squared'] = train['bedrooms']**2
train['bed_bath_rooms'] = train['bedrooms']*train['bathrooms']
train['log_sqft_living'] = np.log(train['sqft_living'])
train['lat_plus_long'] = train['lat']+train['long']

test['bedrooms_squared'] = test['bedrooms']**2
test['bed_bath_rooms'] = test['bedrooms']*test['bathrooms']
test['log_sqft_living'] = np.log(test['sqft_living'])
test['lat_plus_long'] = test['lat']+test['long']
{% endhighlight %}

## Sckitlearn

* predictors~ sqft_living, bedrooms, bathrooms, bed_bath_rooms, bedrooms_squared
* target~ price


{% highlight python %}
from sklearn import linear_model
X = train[['sqft_living','bedrooms','bathrooms','bed_bath_rooms','bedrooms_squared']]
y = train['price']

model = linear_model.LinearRegression()
model.fit(X, y)
line_y = model.predict(X)

print("intercept: ",model.intercept_)
print("coefficients: ",model.coef_)
print("SSE: ",np.sum((y-line_y)**2))

scikit_learn_SSE = np.sum((y-line_y)**2)
scikit_learn_coeff = model.coef_

plt.plot(line_y, y, 'go')
plt.show()
{% endhighlight %}

    intercept:  321022.818566
    coefficients:  [    308.16971104 -127863.12577131 -110920.90983384   34641.50131801
       -1390.73582348]
    SSE:  1.14723203834e+15



<p><img src='/adadeleta/output_8_1.png' /></p>


#### Custom Gradient Descent Algorithm

1. Adadelta (a customization of Adagrad) [1]
    * It adapts the learning rate to the parameters
    * Reduces its decreasing learning rate
2. Recursively runs gradient descent
    * Back propogate the parameters
    * Increases learning rate if cost is decreasing
    * Decrease learning rate if cost is increasing and cost is better than average
    * Stop the descent if the decrease in cost is linear
    
[1] https://arxiv.org/pdf/1609.04747v1.pdf


#### Convergence can be slow

Sometimes the gradient doesn't converge to zero easily.
Getting pretty close can be better than waiting for the algorithm to converge. A new convergence criter to stop the algorithm when the loss function starts to decrease at a linear rate.



{% highlight python %}
def normalize_features(X):
    
    norms = np.linalg.norm(X, axis=0)
    return X/norms,norms

def multiple_gradient_descent(predictors,target,method,iters_before_descent,alpha):
    
    def cost_rate_of_change(loss_over_time,i,tolerance,iters_before_descent=None):
        point_1=i-101
        point_2=i-51
        point_3=i-1
        if iters_before_descent==None:
            iters_before_descent=(i-point_1)

        if i>iters_before_descent and i%100==0:
            slope_=(loss_over_time[point_3]-loss_over_time[point_1])/(point_3-point_1)
            intercept_=loss_over_time[point_3]-slope_*point_3
            if (slope_*point_2+intercept_-loss_over_time[point_2])/intercept_<tolerance:
                return "Linear"
        return "Non-Linear"

    tolerance=1e-6
    y=target.values
    
    mean_pred=np.sum((y-np.mean(y))**2)
    
    m,n =np.shape(predictors.values)
    x=np.ones((m,n+1))
    x[:,1:]=predictors.values
    

    w=np.zeros(x.shape[1]).reshape((x.shape[1],1))
    loss_over_time = []
    gradient_over_time = []
    alpha_over_time = []
    history_ = []
    cost_history = 0
    slope_history = 0
    converged=False
    i=1
    while(True):

        w,alphax,cost,gradient = method(x, y, 100, alpha ,w)

        cost_change=abs(cost_history-cost)
        slope=np.sum(gradient**2)
        convergence_factor=np.sqrt(np.sum(gradient**2)*2.0/(n+1))


        if cost<cost_history and alpha<10000:
            alpha=alpha*(1.05)
        elif cost>=cost_history and cost<=mean_pred:
            alpha=alpha*(0.95)    
        elif cost>=cost_history and cost>mean_pred:
            alpha=alpha 
                
        if i>9999:
            break
        
#         print("iteration: ",i)
#         print("cost: ",cost)
#         print("cost change: ",abs(cost_history-cost)/cost*100)
#         print("gradient: ",convergence_factor)
#         print("alpha: ",alpha)
#         print('overshot: ', w)
    
        loss_over_time.append(cost)
        gradient_over_time.append(convergence_factor)
        alpha_over_time.append(alpha)
        
        rate = cost_rate_of_change(loss_over_time,i,tolerance,iters_before_descent)
        if rate == "Linear":
            converged=True
            break
            
        cost_history =  cost
        slope_history = slope
        i+=1

    return w,cost,converged,loss_over_time,gradient_over_time,alpha_over_time

X = train[['sqft_living','bedrooms','bathrooms','bed_bath_rooms','bedrooms_squared']]
y = train[["price"]]

SST = np.sum((y-np.mean(y))**2)[0]
{% endhighlight %}


{% highlight python %}
import time
start_time = time.time()

input_space = X
output = y
method = Regression.adadelta_gradient_descent
iters_before_descent=3000
alpha=0.01

results =multiple_gradient_descent(
    input_space,
    output,
    method,
    iters_before_descent,
    alpha)

w,cost,converged,loss_over_time,gradient_over_time,alpha_over_time = results
print("--- %s seconds ---" % (time.time() - start_time))
{% endhighlight %}

    --- 48.816388845443726 seconds ---



{% highlight python %}
print("scikit learn coefficients: ",scikit_learn_coeff)
print("adadelta coefficients: ",w[1:].flatten())
print("SST: ",SST)
print("scikit learn SSE: ",scikit_learn_SSE)
print("adadelta SSE: ",np.sum((y-Regression.numpy_predict(X,w))**2)[0])
{% endhighlight %}

    scikit learn coefficients:  [    308.16971104 -127863.12577131 -110920.90983384   34641.50131801
       -1390.73582348]
    adadelta coefficients:  [   306.46533823 -82147.69171734 -88609.9491374   28721.78223027
      -5357.7157224 ]
    SST:  2.37576186177e+15
    scikit learn SSE:  1.14723203834e+15
    adadelta SSE:  1.14947131588e+15


#### The parameters are close to Scikit Learn's closed-form solution 

The algorithm takes 48.8 seconds to converged to a solution. The linearity of the loss rate is checked after 3000 epochs.


{% highlight python %}
fig = plt.figure(figsize=(10,12))
fig.suptitle("Adadelta Gradient Descent Performance", fontsize=16)
ax = plt.subplot(3,1,1)
ax.set_title("Loss")
ax.plot(loss_over_time[:])
ax = plt.subplot(3,1,2)
ax.set_title("Gradient")
ax.plot(gradient_over_time[:])
ax = plt.subplot(3,1,3)
ax.set_title("Alpha")
ax.plot(alpha_over_time[:])
plt.show()
{% endhighlight %}


<p><img src='/adadeleta/output_14_0.png' /></p>



{% highlight python %}
fig = plt.figure(figsize=(10,12))
fig.suptitle("Adadelta Gradient Descent Performance", fontsize=16)
ax = plt.subplot(3,1,1)
ax.set_title("Loss")
ax.plot(loss_over_time[1500:])
ax = plt.subplot(3,1,2)
ax.set_title("Gradient")
ax.plot(gradient_over_time[1500:])
ax = plt.subplot(3,1,3)
ax.set_title("Alpha")
ax.plot(alpha_over_time[1500:])
plt.show()
{% endhighlight %}


<p><img src='/adadeleta/output_15_0.png' /></p>


#### Looking at last 1600 epochs

The cost function seems to be decreasing almost linearly while the alpha and gradient functions are decreasing more jaggedly.

#### Save time at the expense of a little accuracy?

The cost function can be check for linearity. The idea is that when cost start decreasing linearly, the algorithm will take too long to converge. It could be beneficial to stop the algorithm when there is limited improvement to the cost function.




{% highlight python %}
import time
start_time = time.time()

input_space = X
output = y
method = Regression.adadelta_gradient_descent
iters_before_descent=200
alpha=0.01

results =multiple_gradient_descent(
    input_space,
    output,
    method,
    iters_before_descent,
    alpha)

w,cost,converged,loss_over_time,gradient_over_time,alpha_over_time = results
print("--- %s seconds ---" % (time.time() - start_time))
{% endhighlight %}

    --- 24.04082989692688 seconds ---



{% highlight python %}
print("scikit learn coefficients: ",scikit_learn_coeff)
print("adadelta coefficients: ",w[1:].flatten())
print("SST: ",SST)
print("scikit learn SSE: ",scikit_learn_SSE)
print("adadelta SSE: ",np.sum((y-Regression.numpy_predict(X,w))**2)[0])
{% endhighlight %}

    scikit learn coefficients:  [    308.16971104 -127863.12577131 -110920.90983384   34641.50131801
       -1390.73582348]
    adadelta coefficients:  [   307.09084409 -55060.95637356 -62926.0491083   21999.67276166
      -6750.40350354]
    SST:  2.37576186177e+15
    scikit learn SSE:  1.14723203834e+15
    adadelta SSE:  1.15391181724e+15


#### Not too shabby...

The process took 24.04 seconds and checked for loss rate linearity after 200 epochs. Overall, the parameters are still pretty close.



{% highlight python %}
fig = plt.figure(figsize=(10,12))
fig.suptitle("Adadelta Gradient Descent Performance", fontsize=16)
ax = plt.subplot(3,1,1)
ax.set_title("Loss")
ax.plot(loss_over_time[:])
ax = plt.subplot(3,1,2)
ax.set_title("Gradient")
ax.plot(gradient_over_time[:])
ax = plt.subplot(3,1,3)
ax.set_title("Alpha")
ax.plot(alpha_over_time[:])
plt.show()
{% endhighlight %}


<p><img src='/adadeleta/output_20_0.png' /></p>



{% highlight python %}

{% endhighlight %}
