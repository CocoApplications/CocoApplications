---
layout: post
title:  "Custom Time Series & Signal Processing Library"
date: 2017-09-10 12:00:00
author: Rohan Kotwani
excerpt: "Custom library for signal processing and time series techniques."
tags: 
- Time Series
- Signal Processing

---

## Table of Contents

1. Differencing
2. Outlier Detection
3. Detrending
4. Fourier Transforms
5. Filtering on Frequency and Magnitude
6. Trend Component (Polynomial Component)
7. Seasonal Component with Waveform Generation
8. Seasonal Regression with LASSO Feature Selection
9. Recurrent Neural Network
10. Statmodels ARIMA modeling

## Introduction

Here is a brief overview of how signal processing techniques can non-stationary time series that depend on multiple dimensions. This is constrasted with existing, prepackaged, time series modelling techniques such as the autoregressive integrated moving average, ARIMA, models. 


{% highlight python %}
import importlib
import DSP
importlib.reload(DSP)

import Regression
importlib.reload(Regression)
{% endhighlight %}




    <module 'Regression' from '/Users/rohankotwani/Documents/Complex-Time-Series-Signal-Processing/Regression.py'>




{% highlight python %}
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd

full=pd.read_csv("DATA/DSC_Time_Series_Challenge.csv",dtype = {'Day ':str,'Sessions':int,'Pageviews':int})
full['time']=[datetime.datetime.strptime(t[0],"%m/%d/%y") for t in full[['Day ']].values]
full=full.sort_values(by=['time'])
full['index']=DSP.datetime_index(full['time'])

train=full[:450].copy()
valid=full[450:].copy()
train.head(n=5)

plt.figure(figsize=(10,6))
plt.plot(full['index'],full['Pageviews'],'-')
plt.xlabel('time')
plt.ylabel('Pageviews')
plt.title('Pageviews')
plt.show()
{% endhighlight %}


<p><img src='/signal_images/output_1_0.png' /></p>


### Differenced time series


{% highlight python %}
time_diff_signal = DSP.time_diff_variable(train[['Pageviews']].values.flatten(),1)
plt.figure(figsize=(10,6))
plt.plot(train['index'][1:],time_diff_signal,'-');plt.xlabel('time');plt.ylabel('Pageviews[t] - Pageviews[t-1]');plt.title('Pageviews[t] - Pageviews[t-1]')
plt.show()
{% endhighlight %}


<p><img src='/signal_images/output_3_0.png' /></p>


### Outlier detection


{% highlight python %}
normal = np.zeros((time_diff_signal.shape[0]+1))
normal[1:] = time_diff_signal/np.std(time_diff_signal)
normal = np.exp(abs(normal)**2.0)
plt.figure(figsize=(10,6))
plt.plot(normal/np.linalg.norm(normal))
plt.show()
print(np.corrcoef(normal,train[['Pageviews']].values.flatten()))
{% endhighlight %}


<p><img src='/signal_images/output_5_0.png' /></p>


    [[ 1.          0.21752737]
     [ 0.21752737  1.        ]]


### Detrending time series using a decision tree


{% highlight python %}
O_train, O_valid = np.array(normal/np.linalg.norm(normal) > 0.9).astype(int), np.zeros(len(valid))
{% endhighlight %}


{% highlight python %}
from sklearn.tree import DecisionTreeRegressor

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_1.fit(train[['index']].values, train[['Pageviews']].values)
y_1 = regr_1.predict(train[['index']].values)

plt.figure(figsize=(10,6))
plt.plot(train[['Pageviews']].values.flatten()-y_1)
plt.show()

{% endhighlight %}


<p><img src='/signal_images/output_8_0.png' /></p>


### FFT transformation


{% highlight python %}
detrend_signal = train[['Pageviews']].values.flatten()-y_1
N = len(detrend_signal)
unfiltered  = DSP.get_frequency_domain(detrend_signal)
f,y  = unfiltered[:,0],unfiltered[:,1]
y_abs=( 2.0/N * np.abs(y[1:]))
plt.figure(figsize=(10,6))
plt.plot(f[1:],y_abs);plt.xlabel('frequency');plt.ylabel('absolute magnitude');plt.title("Absolute Magnitude of Complex Frequencies")
plt.show()
{% endhighlight %}

    /Users/rohankotwani/anaconda/envs/datasci/lib/python3.5/site-packages/numpy/core/numeric.py:531: ComplexWarning: Casting complex values to real discards the imaginary part
      return array(a, dtype, copy=False, order=order)



<p><img src='/signal_images/output_10_1.png' /></p>


### Filtering frequency domain: center, band, and threshold


{% highlight python %}
abs_filtered = np.absolute(DSP.filter_freq_domain(unfiltered, center=0.3,band=0.2,threshold=500))
print("Frequency, Magnitude")
print(abs_filtered)

period_list = set([round(1/(ft)) for ft,ht in abs_filtered if round(1/(ft))>2])
print("periods: ",period_list)
{% endhighlight %}

    Frequency, Magnitude
    [[  1.42222222e-01   3.31238582e+05]
     [  1.44444444e-01   2.20981166e+05]
     [  1.46666667e-01   1.14687470e+05]
     [  2.82222222e-01   1.17656492e+05]
     [  2.84444444e-01   2.39078889e+05]
     [  2.86666667e-01   3.87622866e+05]
     [  4.28888889e-01   1.51845485e+05]]
    periods:  {3.0, 4.0, 7.0}


### Trend component T(t) : Polynomial regression


{% highlight python %}
import math
def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def complex_lengendre(x):
    x = x - np.mean(x)
    n = len(x)
    t = np.linspace(-1,1,n)
    mat = np.zeros((n,30))
    
    for v in range(0+1,30+1):
        val = 0
        j=1
        for w in range(0,v+1):
            if w % 2 == 0:
                j=-1
            val += j * ( 1/(2**v) * nCr(v,w)**2 * (t-1)**(v-w) * (t+1)**w )
            
        normal = (val - np.min(val))/(np.max(val)- np.min(val)) * (2) - 1
        
        mat[:,v-1] = np.correlate(x,normal)
        
    return mat

mat1 = complex_lengendre(train.Pageviews.values.flatten())
t = range(0+1,30+1)
temp = np.sqrt(np.sum(mat1**2,axis=0))
normal = (temp - np.mean(temp))/np.std(temp)
normal = np.exp(normal)
plt.figure(figsize=(10,6))
plt.plot(t,normal)
plt.show()

heap = []

for i in range(1,30):
    heap.append((normal[i],i))
    
sorted_heap = Regression.heapsort(heap)

heap = []
for y in range(1,20):
    T_train = np.zeros(len(train))
    T_valid = np.zeros(len(valid))
    j = sorted_heap[-y][1]
    while j>0: 

        T_train = np.column_stack((T_train,train[['index']]**j))
        T_valid = np.column_stack((T_valid,valid[['index']]**j))
        j=j-2


    z = Regression.numpy_simple_regression(T_train[:,1:],train[['Pageviews']].values)
    SSE = Regression.numpy_SSE(T_valid[:,1:],valid[['Pageviews']],z)
    SST = np.sum((valid[['Pageviews']].values - np.mean(valid[['Pageviews']].values))**2)
    heap.append((1-SSE/SST,y))
    
Regression.heapsort(heap)
{% endhighlight %}


<p><img src='/signal_images/output_14_0.png' /></p>





    [(-6.011873941401473e+18, 17),
     (-5.4277886092490125e+17, 19),
     (-30904090423470896.0, 13),
     (-620885790410212.0, 12),
     (-79303334867452.234, 10),
     (-20041658321596.043, 7),
     (-1969692284880.6143, 14),
     (-8156007026.8742466, 8),
     (-2770024103.7658229, 9),
     (-112374464.67439486, 4),
     (-437987.28174616257, 18),
     (-47462.882029266701, 15),
     (-36467.10740261824, 6),
     (-1680.5341147097188, 5),
     (-163.91665441030392, 11),
     (-50.919645927770695, 16),
     (-0.46251447713035376, 3),
     (0.33225583607722331, 1),
     (0.37298602907873102, 2)]



### Seasonal Component S(t) : Waveform generation


{% highlight python %}
import importlib
import DSP
importlib.reload(DSP)
S_train = np.ones((len(train),1))
S_valid = np.ones((len(valid),1))
 
    
for period in [2.0]:
    x = DSP.generate_square_waves(train['index'],period)
    S_train=np.column_stack((S_train,x))

    x = DSP.generate_square_waves(valid['index'],period)
    S_valid=np.column_stack((S_valid,x))
    
    for i in range(x.shape[1]):
        plt.stem(x[:20,i])
        plt.show()

S_train,S_valid = S_train[:,1:],S_valid[:,1:]
{% endhighlight %}


<p><img src='/signal_images/output_16_0.png' /></p>



<p><img src='/signal_images/output_16_1.png' /></p>


### Trend & Seasonal Regression with LASSO feature selection - x(t) = T(t) + S(t) +R(t) + error 


{% highlight python %}
heap = []
for i in range(1,15):
    T_train = np.zeros(len(train))
    T_valid = np.zeros(len(valid))
    
    j = sorted_heap[-i][1]
    while j>0: 

        T_train = np.column_stack((T_train,train[['index']]**j))
        T_valid = np.column_stack((T_valid,valid[['index']]**j))

        j=j-2
        
    skip_list=None
    for thresh in np.linspace(500,600,10):
        abs_filtered = np.absolute(DSP.filter_freq_domain(unfiltered, center=0.25,band=0.25,threshold=thresh))

        period_list = set([round(1/(ft)) for ft,ht in abs_filtered if round(1/(ft))>1])
#                 print(period_list)
        if skip_list==period_list:
#             print("same sequences")
            continue
        skip_list = period_list

        S_train = np.ones((len(train),1))
        S_valid = np.ones((len(valid),1))
        for period in period_list:

            x = DSP.generate_impluse_waves(train['index'],period)
            S_train=np.column_stack((S_train,x))
            x = DSP.generate_sawtooth_waves(train['index'],period)
            S_train=np.column_stack((S_train,x))
            
            x = DSP.generate_impluse_waves(valid['index'],period)
            S_valid=np.column_stack((S_valid,x))
            x = DSP.generate_sawtooth_waves(valid['index'],period)
            S_valid=np.column_stack((S_valid,x))

        S_train,S_valid = S_train[:,1:],S_valid[:,1:]

#         mat = np.zeros((S_train.shape[0],S_train.shape[1]))
        
#         for s in range(S_train.shape[1]):
#             mat[:,s] = np.correlate(S_train[:,s].copy(),train[['Pageviews']].values.flatten())
            
            
#         temp = np.sqrt(np.sum(mat**2,axis=0))

#         # temp = np.sum(mat1,axis=0)

#         normal = (temp - np.mean(temp))/np.std(temp)

#         normal = np.exp(normal)

#         heapx = []

#         for i in range(S_train.shape[1]):
#             heapx.append((normal[i],i))

#         sorted_heap = Regression.heapsort(heapx)

#         print(sorted_heap)
            
        
        
        P_train_df = pd.DataFrame(np.column_stack((S_train,T_train,O_train)))
        P_valid_df = pd.DataFrame(np.column_stack((S_valid,T_valid,O_valid)))
        
        lasso_heap=[]
        for l1_penalty in np.logspace(1, 7, num=13):
            z = Regression.sklearn_lasso_regression(P_train_df,train[['Pageviews']],l1_penalty)
            SSE = Regression.numpy_SSE(P_valid_df,valid[['Pageviews']],z)
            lasso_heap.append((SSE,l1_penalty))
        penalty = Regression.heapsort(lasso_heap)[0][1]
        mask = Regression.sklearn_lasso_feature_selection(P_train_df,train[['Pageviews']],penalty)

        z = Regression.numpy_simple_regression(P_train_df[mask].values,train[['Pageviews']])
        SSE = Regression.numpy_SSE(P_valid_df[mask].values,valid[['Pageviews']],z)
        heap.append((SSE,i,thresh,penalty))

SSE,i,thresh,penalty = Regression.heapsort(heap)[0]
print("model paramters: ",i,thresh,penalty)
        
    
T_train = np.zeros(len(train))
T_valid = np.zeros(len(valid))
j = sorted_heap[-i][1]
while j>0: 

    T_train = np.column_stack((T_train,train[['index']]**j))
    T_valid = np.column_stack((T_valid,valid[['index']]**j))

    j=j-2
    
abs_filtered = np.absolute(DSP.filter_freq_domain(unfiltered, center=0.25,band=0.25,threshold=thresh))

period_list = set([round(1/(ft)) for ft,ht in abs_filtered if round(1/(ft))>2])
print(period_list)
S_train = np.ones((len(train),1))
S_valid = np.ones((len(valid),1))
for period in period_list:

        x = DSP.generate_impluse_waves(train['index'],period)
        S_train=np.column_stack((S_train,x))
        x = DSP.generate_sawtooth_waves(train['index'],period)
        S_train=np.column_stack((S_train,x))

        x = DSP.generate_impluse_waves(valid['index'],period)
        S_valid=np.column_stack((S_valid,x))
        x = DSP.generate_sawtooth_waves(valid['index'],period)
        S_valid=np.column_stack((S_valid,x))

S_train,S_valid = S_train[:,1:],S_valid[:,1:]
P_train = np.column_stack((S_train,T_train,O_train))
P_valid = np.column_stack((S_valid,T_valid,O_valid))
P_train_df = pd.DataFrame(P_train)
P_valid_df = pd.DataFrame(P_valid)

mask = Regression.sklearn_lasso_feature_selection(P_train_df,train[['Pageviews']],penalty)
z = Regression.numpy_simple_regression(P_train_df[mask].values,train[['Pageviews']])

SSE = Regression.numpy_SSE(P_valid_df[mask].values,valid[['Pageviews']],z)
SST = np.sum((valid['Pageviews']-np.mean(valid['Pageviews']))**2)
print("R-squared: ",1-SSE/SST)

fig = plt.figure(figsize=(10,12))
fig.suptitle("Datasciencecentral.com Pageviews", fontsize=16)
ax = plt.subplot(2,1,1)
predict = Regression.numpy_predict(P_valid_df[mask].values,z)
ax.plot(valid['index'],valid['Pageviews'],'b-');plt.plot(valid['index'],predict,'r-',linewidth=1.5);ax.set_title("Pageviews: valid")
ax = plt.subplot(2,1,2)
predict = Regression.numpy_predict(P_train_df[mask].values,z)
ax.plot(train['index'],train['Pageviews'],'b-');plt.plot(train['index'],predict,'r-',linewidth=1.5);ax.set_title("Pageviews: train")
plt.show()






{% endhighlight %}

    model paramters:  2 500.0 10.0
    {3.0, 4.0, 7.0}
    R-squared:  0.723189957088



<p><img src='/signal_images/output_18_1.png' /></p>


### Recurrent Neural Network


{% highlight python %}
import tensorflow as tf
import os
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn

df = pd.read_csv("DATA/DSC_Time_Series_Challenge.csv")

x  = df[['Pageviews']].values.reshape((len(df),1))
{% endhighlight %}


{% highlight python %}

TS = np.array(df[['Pageviews']].values.reshape((len(df),1)))
num_periods = 281
f_horizon = 1  #forecast horizon, one period into the future

x_data = TS[:(len(TS)-(len(TS) % num_periods))]
x_batches = x_data.reshape(-1, num_periods, 1)
print("Number of periods: ",len(TS)-(len(TS) % num_periods))
y_data = TS[1:(len(TS)-(len(TS) % num_periods))+f_horizon]
y_batches = y_data.reshape(-1, num_periods, 1)


def test_data(series,forecast,num_periods):
    test_x_setup = TS[-(num_periods + forecast):]
    testX = test_x_setup[:num_periods].reshape(-1, num_periods, 1)
    testY = TS[-(num_periods):].reshape(-1, num_periods, 1)
    return testX,testY

X_test, Y_test = test_data(TS,f_horizon,num_periods )

tf.reset_default_graph()   #We didn't have any previous graph objects running, but this would reset the graphs

num_periods = num_periods      #number of periods per vector we are using to predict one period ahead
inputs = 1            #number of vectors submitted
hidden = 500          #number of neurons we will recursively work through, can be changed to improve accuracy
output = 1            #number of output vectors

X = tf.placeholder(tf.float32, [None, num_periods, inputs])   #create variable objects
y = tf.placeholder(tf.float32, [None, num_periods, output])


basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)   #create our RNN object
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)               #choose dynamic over static

learning_rate = 0.001   #small learning rate so we don't overshoot the minimum

stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])           #change the form into a tensor
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)        #specify the type of layer (dense)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])          #shape of results
 
loss = tf.reduce_sum(tf.square(outputs - y))    #define the cost function which evaluates the quality of our model
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)          #gradient descent method
training_op = optimizer.minimize(loss)          #train the result of the application of the cost_function                                 

init = tf.global_variables_initializer()           #initialize all the variables

epochs = 1000     #number of iterations or training cycles, includes both the FeedFoward and Backpropogation

with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
        if ep % 100 == 0:
            mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
            print(ep, "\tMSE:", mse)
    
    y_pred = sess.run(outputs, feed_dict={X: X_test})
    
R2 = 1-np.sum((np.ravel(Y_test)-np.ravel(y_pred))**2)/np.sum((np.ravel(Y_test)-np.mean(Y_test.ravel()))**2)
{% endhighlight %}

    Number of periods:  562
    0 	MSE: 4.91916e+10
    100 	MSE: 2.99341e+09
    200 	MSE: 1.54519e+09
    300 	MSE: 1.327e+09
    400 	MSE: 1.1797e+09
    500 	MSE: 1.11012e+09
    600 	MSE: 1.18234e+09
    700 	MSE: 8.93702e+08
    800 	MSE: 8.51683e+08
    900 	MSE: 9.31046e+08



{% highlight python %}
print("R-squared: ", R2)
fig = plt.figure(figsize=(10,12))
fig.suptitle("Datasciencecentral.com Pageviews", fontsize=16)
ax = plt.subplot(2,1,1)
ax.set_title("Forecast vs Actual", fontsize=14)
plt.plot(pd.Series(np.ravel(Y_test)), "b-", markersize=10, label="Actual")
#plt.plot(pd.Series(np.ravel(Y_test)), "w*", markersize=10)
ax.plot(pd.Series(np.ravel(y_pred)), "r-", markersize=10, label="Forecast")
ax.legend(loc="upper left")
# fig.xlabel("Time Periods")
plt.show()
{% endhighlight %}

    R-squared:  0.826689976288



<p><img src='/signal_images/output_22_1.png' /></p>


### Statsmodels autocorrelation


{% highlight python %}
import statsmodels
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf

full=pd.read_csv("DATA/DSC_Time_Series_Challenge.csv",dtype = {'Day ':str,'Sessions':int,'Pageviews':int})
time=[datetime.datetime.strptime(t[0],"%m/%d/%y") for t in full[['Day ']].values]
full['time']=time
full['Pageviews']=full['Pageviews'].apply(float)

series = full.set_index('time')['Pageviews']
train=series[:500].copy()
valid=series[500:].copy()

ticker_data_acf_1 =  acf(train)[1:32]
test_df = pd.DataFrame([ticker_data_acf_1]).T
test_df.columns = ['ACF']
test_df.plot(kind='bar')
plt.show()

ticker_data_pacf_1 =  pacf(train)[1:32]
test_df = pd.DataFrame([ticker_data_pacf_1]).T
test_df.columns = ['PACF']
test_df.plot(kind='bar')
plt.show()
{% endhighlight %}


<p><img src='/signal_images/output_24_0.png' /></p>



<p><img src='/signal_images/output_24_1.png' /></p>


### Statsmodels ARIMA model - choose p, d, and q parameters


{% highlight python %}
%%capture --no-stdout
heap = []
import time
for p in range(0,10):
    for d in range(0,3):
        for q in range(1,10):
            model = statsmodels.tsa.arima_model.ARIMA(train, order=(p,d,q))
            try:
                model_fit = model.fit(disp=0)
                residuals = pd.DataFrame(model_fit.resid)
                predictions = model_fit.forecast(len(valid))
                SSE = np.sum((valid.values - predictions[0])**2)
                heap.append((SSE,p,d,q))
            except:
                time.sleep(0.01)
                pass


{% endhighlight %}


{% highlight python %}
SSE,p,d,q = Regression.heapsort(heap)[0]

model = statsmodels.tsa.arima_model.ARIMA(train, order=(p,d,q))
model_fit = model.fit(disp=0)
residuals = pd.DataFrame(model_fit.resid)
predictions = model_fit.forecast(len(valid))
SSE = np.sum((valid.values - predictions[0])**2)
SST = np.sum((valid.values - np.mean(valid.values))**2)
print("R-squared: ",1-SSE/SST)
plt.plot(valid.values,'b.');plt.plot(predictions[0],'r-',linewidth=1.5);plt.title("ARIMA: validation data")
plt.show()
{% endhighlight %}

    R-squared:  0.369187234626


    /Users/rohankotwani/anaconda/envs/datasci/lib/python3.5/site-packages/statsmodels/base/model.py:466: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      "Check mle_retvals", ConvergenceWarning)



<p><img src='/signal_images/output_27_2.png' /></p>


