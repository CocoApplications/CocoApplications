---
layout: post
title:  "Kaggle Competition: Draper Satellite Image Chronology"
date: 2017-07-06 12:00:00
author: Rohan Kotwani
excerpt: "Quantify the difference between days with image stitching for Draper Satellite Image Chronology"
tags: 
- Seam Carving
- Object Detection
- Filtering
- Phase Correlation
- Perspective transforms

---


{% highlight python %}
import importlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import Regression
import CVision

{% endhighlight %}

# Training Images - Day 5 Compared to Days 1-4 


{% highlight python %}
importlib.reload(CVision)
importlib.reload(Regression)
tag1 = cv2.imread("draper/train/set160_5.jpeg")

rsq_heap=[]
for i in range(1,5):
    tag2 = cv2.imread("draper/train/set160_"+str(i)+".jpeg")
    img1,img2,is_match = CVision.im_stitcher(tag1,tag2,plot=False,warp_threshold=5000)
    msk = img2==0
    img1[msk] = 0
    
    n = len(img2[~msk].flatten())
    
    # Make sure there is data to match
    if n <1000:
        continue
        
    z = Regression.numpy_simple_regression(img2[~msk].astype(np.int16).reshape(n,1),img1[~msk].astype(np.int16).flatten())
    SSE = Regression.numpy_SSE(img2[~msk].astype(np.int16).reshape(n,1),img1[~msk].astype(np.int16).flatten(),z)
    SST = np.sum((img1[~msk].astype(np.int16)-np.mean(img1[~msk].astype(np.int16)))**2)
    
    #Ignore low R-square values
    if 1-SSE/SST<0.05:
        print("R-squared < 0.05")
        continue
    
    plt.figure(figsize=(10,7))
    plt.title("Comparision ,Original image")
    CVision.cv2_to_plt(np.hstack([tag1,tag2]))
    
    
    print("image number: ",i)
    print("SSE: ",SSE)
    print("SST: ",SST)
    print("R-squared: ",1-SSE/SST)
    print("coefficients: ",z)
    
    plt.figure(figsize=(10,7))
    plt.title("Masked Comparion, Warped image")
    CVision.cv2_to_plt(np.hstack([img1,img2]))



    rsq_heap.append((1-SSE/SST,i))
    
Regression.heapsort(rsq_heap)
{% endhighlight %}

    Warp Threshold 1312.02



<p><img src='/draper-satellite-example/output_2_1.png' /></p>


    image number:  1
    SSE:  238931148.968
    SST:  455256365.98
    R-squared:  0.475172305491
    coefficients:  [ 38.27697934   0.66404544]



<p><img src='/draper-satellite-example/output_2_3.png' /></p>


    Warp Threshold 1990.67



<p><img src='/draper-satellite-example/output_2_5.png' /></p>


    image number:  2
    SSE:  184863799.439
    SST:  374308395.391
    R-squared:  0.506119013852
    coefficients:  [ 34.93400105   0.67871176]



<p><img src='/draper-satellite-example/output_2_7.png' /></p>


    Warp Threshold 2066.28



<p><img src='/draper-satellite-example/output_2_9.png' /></p>


    image number:  3
    SSE:  369589401.984
    SST:  678684943.588
    R-squared:  0.455433031961
    coefficients:  [ 39.61894188   0.67145665]



<p><img src='/draper-satellite-example/output_2_11.png' /></p>


    Warp Threshold 1170.92



<p><img src='/draper-satellite-example/output_2_13.png' /></p>


    image number:  4
    SSE:  213825041.258
    SST:  649302659.207
    R-squared:  0.670685098504
    coefficients:  [ 16.41474607   0.82778366]



<p><img src='/draper-satellite-example/output_2_15.png' /></p>





    [(0.45543303196090956, 3),
     (0.47517230549053879, 1),
     (0.50611901385244318, 2),
     (0.67068509850402169, 4)]



# Training Image Day 5 Compared to Test Images 


{% highlight python %}
importlib.reload(CVision)
importlib.reload(Regression)
tag1 = cv2.imread("draper/train/set160_5.jpeg")

rsq_heap=[]
for i in range(1,6):
    tag2 = cv2.imread("draper/test/set74_"+str(i)+".jpeg")
    img1,img2,is_match = CVision.im_stitcher(tag1,tag2,plot=False,warp_threshold=5000)
    msk = img2==0
    img1[msk] = 0
    
    n = len(img2[~msk].flatten())
    
    # Make sure there is data to match
    if n <1000:
        continue
        
    z = Regression.numpy_simple_regression(img2[~msk].astype(np.int16).reshape(n,1),img1[~msk].astype(np.int16).flatten())
    SSE = Regression.numpy_SSE(img2[~msk].astype(np.int16).reshape(n,1),img1[~msk].astype(np.int16).flatten(),z)
    SST = np.sum((img1[~msk].astype(np.int16)-np.mean(img1[~msk].astype(np.int16)))**2)
    
    #Ignore low R-square values
    if 1-SSE/SST<0.05:
        print("R-squared < 0.05")
        continue
    
    plt.figure(figsize=(10,7))
    plt.title("Comparision ,Original image")
    CVision.cv2_to_plt(np.hstack([tag1,tag2]))
    
    
    print("image number: ",i)
    print("SSE: ",SSE)
    print("SST: ",SST)
    print("R-squared: ",1-SSE/SST)
    print("coefficients: ",z)
    
    plt.figure(figsize=(10,7))
    plt.title("Masked Comparion, Warped image")
    CVision.cv2_to_plt(np.hstack([img1,img2]))



    rsq_heap.append((1-SSE/SST,i))
    
Regression.heapsort(rsq_heap)
{% endhighlight %}

    Warp Threshold 3424.36



<p><img src='/draper-satellite-example/output_4_1.png' /></p>


    image number:  1
    SSE:  60733943.6588
    SST:  103435675.306
    R-squared:  0.412833691284
    coefficients:  [ 63.08725456   0.61048454]



<p><img src='/draper-satellite-example/output_4_3.png' /></p>


    Warp Threshold 3944.86



<p><img src='/draper-satellite-example/output_4_5.png' /></p>


    image number:  2
    SSE:  64836512.4533
    SST:  240181894.443
    R-squared:  0.73005245627
    coefficients:  [-20.32274366   1.17865269]



<p><img src='/draper-satellite-example/output_4_7.png' /></p>


    Warp Threshold 3017.09



<p><img src='/draper-satellite-example/output_4_9.png' /></p>


    image number:  3
    SSE:  83257164.9219
    SST:  367015321.462
    R-squared:  0.773150710466
    coefficients:  [-7.81895735  1.13657705]



<p><img src='/draper-satellite-example/output_4_11.png' /></p>


    Warp Threshold 2631.52



<p><img src='/draper-satellite-example/output_4_13.png' /></p>


    image number:  4
    SSE:  101708966.02
    SST:  254819881.518
    R-squared:  0.600859377948
    coefficients:  [ 36.12586612   0.8096621 ]



<p><img src='/draper-satellite-example/output_4_15.png' /></p>


    Warp Threshold 4349.42



<p><img src='/draper-satellite-example/output_4_17.png' /></p>


    image number:  5
    SSE:  60868739.299
    SST:  112969908.621
    R-squared:  0.461195109015
    coefficients:  [ 39.82375739   0.70391999]



<p><img src='/draper-satellite-example/output_4_19.png' /></p>





    [(0.41283369128398539, 1),
     (0.46119510901454452, 5),
     (0.60085937794846567, 4),
     (0.73005245626978721, 2),
     (0.77315071046573314, 3)]




{% highlight python %}

{% endhighlight %}

