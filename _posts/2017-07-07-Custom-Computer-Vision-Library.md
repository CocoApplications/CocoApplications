---
layout: post
title:  "Custom Computer Vision Library"
date: 2017-07-06 12:00:00
author: Rohan Kotwani
excerpt: "Custom library for computer vision algorithms"
tags: 
- Seam Carving
- Object Detection
- Filtering
- Phase Correlation
- Perspective transforms

---

{% highlight python %}
import importlib
import CVision
importlib.reload(CVision)
{% endhighlight %}




    <module 'CVision' from '/Users/rohankotwani/Documents/ComputerVision/CVision.py'>




{% highlight python %}
import cv2
import matplotlib.pyplot as plt
import numpy as np
import Regression
{% endhighlight %}

### K-means clustering & Image Segmentation 


{% highlight python %}
img = cv2.imread("image_dump/gr175.jpg") 
img = CVision.cv2_resize(img,(800,1300))

plt.figure(figsize=(7,5))
CVision.cv2_to_plt(img)

cluster = CVision.kmeans_color_clustering(img,n=9)
print(cluster)
for c in cluster:
    CVision.color_segmentation(img,c,threshold=100)
{% endhighlight %}


<p><img src='computervision_images/output_3_0.png' /></p>



<p><img src='computervision_images/output_3_1.png' /></p>


    [[  68.22302368  112.6797825     7.62179926]
     [ 182.0180663   197.97181799  153.18534819]
     [ 110.89770443  158.67313526   41.34804296]
     [ 237.92677073  241.17186698  237.66669425]
     [  56.1333727    57.59788593   30.87111824]
     [  88.26041097  137.89178132   15.93635156]
     [ 139.9548802   182.86410616   73.80964718]
     [ 198.03967624  102.9776226     1.88938264]
     [  62.27309948   35.4978422   180.41130906]]



<p><img src='computervision_images/output_3_3.png' /></p>



<p><img src='computervision_images/output_3_4.png' /></p>



<p><img src='computervision_images/output_3_5.png' /></p>



<p><img src='computervision_images/output_3_6.png' /></p>



<p><img src='computervision_images/output_3_7.png' /></p>



<p><img src='computervision_images/output_3_8.png' /></p>



<p><img src='computervision_images/output_3_9.png' /></p>



<p><img src='computervision_images/output_3_10.png' /></p>



<p><img src='computervision_images/output_3_11.png' /></p>


### Object Detection with Canny Filter


{% highlight python %}

object_list = ["y95.jpg","115.jpg","stop-sign.jpeg","ghost-shark.jpeg"]
object_dir = ["image_dump/"+obj for obj in object_list]
for d in object_dir:
    obj = cv2.imread(d)         
    obj = cv2.resize(obj,(700,500))  
    p = CVision.get_object(obj,canny_param_1=35,canny_param_2=135,min_area=5500,min_perimeter=300)
{% endhighlight %}

    converting image to grayscale
    top left point:  (348, 291) width and height:  (98, 134) contour area:  9795.5  perimeter:  455.60511887073517  is convex: False  square area:  11883.12688961171



<p><img src='computervision_images/output_5_1.png' /></p>


    top left point:  (348, 294) width and height:  (94, 134) contour area:  9735.5  perimeter:  397.8061306476593  is convex: False  square area:  11430.213565401267



<p><img src='computervision_images/output_5_3.png' /></p>


    top left point:  (1, 1) width and height:  (229, 353) contour area:  254.0  perimeter:  1052.8468807935715  is convex: False  square area:  2528.2057477928465
    top left point:  (1, 445) width and height:  (254, 421) contour area:  248.5  perimeter:  1066.1778362989426  is convex: False  square area:  2937.6794449072477
    top left point:  (7, 635) width and height:  (34, 32) contour area:  152.5  perimeter:  210.55129599571228  is convex: False  square area:  645.3529283951066
    top left point:  (1, 587) width and height:  (112, 73) contour area:  126.5  perimeter:  464.0559091567993  is convex: False  square area:  2432.355919080146
    top left point:  (26, 648) width and height:  (21, 9) contour area:  74.0  perimeter:  59.94112479686737  is convex: False  square area:  147.54097467692827
    top left point:  (27, 631) width and height:  (9, 10) contour area:  44.0  perimeter:  36.62741661071777  is convex: False  square area:  59.9999972719379
    top left point:  (31, 649) width and height:  (10, 8) contour area:  36.5  perimeter:  35.21320307254791  is convex: False  square area:  56.69998617563897
    top left point:  (1, 1) width and height:  (55, 35) contour area:  34.0  perimeter:  136.16652035713196  is convex: False  square area:  99.893026471138
    converting image to grayscale
    top left point:  (284, 288) width and height:  (84, 117) contour area:  7638.5  perimeter:  348.20815098285675  is convex: False  square area:  8793.360398303135



<p><img src='computervision_images/output_5_5.png' /></p>


    top left point:  (284, 288) width and height:  (84, 117) contour area:  7601.5  perimeter:  345.86500453948975  is convex: False  square area:  8793.360398303135



<p><img src='computervision_images/output_5_7.png' /></p>


    top left point:  (1, 1) width and height:  (229, 355) contour area:  247.0  perimeter:  1086.160588979721  is convex: False  square area:  2667.6332785105915
    top left point:  (1, 446) width and height:  (139, 225) contour area:  133.5  perimeter:  566.0802954435349  is convex: False  square area:  760.8810815757606
    top left point:  (227, 581) width and height:  (118, 191) contour area:  115.0  perimeter:  489.26911330223083  is convex: False  square area:  919.5175184488762
    top left point:  (1, 589) width and height:  (110, 71) contour area:  75.5  perimeter:  380.315793633461  is convex: False  square area:  3720.986462806177
    top left point:  (8, 637) width and height:  (62, 32) contour area:  72.5  perimeter:  421.34523236751556  is convex: False  square area:  1552.200112618957
    top left point:  (1, 1) width and height:  (56, 36) contour area:  34.5  perimeter:  139.58073389530182  is convex: False  square area:  95.46388724313147
    top left point:  (30, 651) width and height:  (9, 6) contour area:  22.0  perimeter:  24.14213538169861  is convex: False  square area:  35.99999202427716
    top left point:  (1, 650) width and height:  (17, 20) contour area:  21.0  perimeter:  124.50966536998749  is convex: False  square area:  301.4483073033407
    converting image to grayscale
    top left point:  (61, 411) width and height:  (160, 249) contour area:  22663.5  perimeter:  745.5117597579956  is convex: False  square area:  33643.98628187552



<p><img src='computervision_images/output_5_9.png' /></p>


    top left point:  (61, 411) width and height:  (160, 194) contour area:  22474.5  perimeter:  679.9970378875732  is convex: False  square area:  27898.05095772352



<p><img src='computervision_images/output_5_11.png' /></p>


    top left point:  (90, 418) width and height:  (145, 156) contour area:  18131.0  perimeter:  499.2447339296341  is convex: False  square area:  22197.62906224467



<p><img src='computervision_images/output_5_13.png' /></p>


    top left point:  (90, 418) width and height:  (145, 156) contour area:  17971.0  perimeter:  496.90158772468567  is convex: False  square area:  22197.62906224467



<p><img src='computervision_images/output_5_15.png' /></p>


    top left point:  (304, 503) width and height:  (173, 149) contour area:  3579.0  perimeter:  5336.025050520897  is convex: False  square area:  24709.63999960944
    top left point:  (135, 489) width and height:  (29, 62) contour area:  1478.0  perimeter:  159.25483298301697  is convex: False  square area:  1689.587651855647
    top left point:  (135, 489) width and height:  (29, 62) contour area:  1456.0  perimeter:  156.9116872549057  is convex: False  square area:  1689.587651855647
    top left point:  (320, 230) width and height:  (165, 65) contour area:  1411.0  perimeter:  2255.797234773636  is convex: False  square area:  10293.188913546153
    top left point:  (313, 543) width and height:  (49, 53) contour area:  1117.5  perimeter:  655.4528785943985  is convex: False  square area:  2190.8465570150584
    top left point:  (391, 1) width and height:  (260, 40) contour area:  1076.0  perimeter:  1018.3574266433716  is convex: False  square area:  7361.300538337091
    converting image to grayscale
    top left point:  (86, 140) width and height:  (385, 346) contour area:  6306.0  perimeter:  3697.594474554062  is convex: False  square area:  87781.91193014849



<p><img src='computervision_images/output_5_17.png' /></p>


    top left point:  (185, 140) width and height:  (124, 87) contour area:  5203.0  perimeter:  1090.0529463291168  is convex: False  square area:  8650.639746287197
    top left point:  (124, 190) width and height:  (36, 24) contour area:  180.0  perimeter:  187.13708317279816  is convex: False  square area:  686.7924151702027
    top left point:  (140, 217) width and height:  (39, 29) contour area:  148.0  perimeter:  309.5634890794754  is convex: False  square area:  1031.128770035073
    top left point:  (160, 301) width and height:  (21, 16) contour area:  135.0  perimeter:  107.39696860313416  is convex: False  square area:  221.00000204497337
    top left point:  (279, 324) width and height:  (41, 59) contour area:  73.0  perimeter:  341.3036028146744  is convex: False  square area:  1929.2229092003836
    top left point:  (173, 245) width and height:  (52, 20) contour area:  70.5  perimeter:  312.43354630470276  is convex: False  square area:  944.939045954161
    top left point:  (232, 265) width and height:  (97, 109) contour area:  69.5  perimeter:  343.6883783340454  is convex: False  square area:  3506.6290912449476
    top left point:  (224, 363) width and height:  (52, 86) contour area:  56.5  perimeter:  289.26197266578674  is convex: False  square area:  1304.5324749458669
    top left point:  (152, 231) width and height:  (11, 10) contour area:  53.5  perimeter:  46.870057225227356  is convex: False  square area:  88.8461571615735



{% highlight python %}

{% endhighlight %}

### 2D convolution with Contant Border = 0


{% highlight python %}
def apply_3_channels(img, function, args):
    assert len(img.shape[:])==3
    n = img.shape[2]
#     print(*args)
    for i in range(n):
        img[:,:,i] = function(img[:,:,i],*args)
    return img


img = cv2.imread("image_dump/snake.jpeg")       
# img = CVision.cv2_resize(img,(1500,1200))
plt.figure(figsize=(7,5))
CVision.cv2_to_plt(img)

sobel_x_kernel = np.array(
                    [[-1.,0.,1.],
                    [-2.,0.,2.],
                    [-1.,0.,1.]],np.float64)

sobel_y_kernel = np.array(
                    [[-1.,-2.,-1.],
                    [0.,0.,0.],
                    [1.,2.,1.]],np.float64)

identity_kernel = np.array(
                    [[-0.,-0.,-0.],
                    [0.,1.,0.],
                    [0.,0.,0.]],np.float64)

edge0_kernel = np.array(
                    [[0.,1.,0.],
                    [1.,-4.,1.],
                    [0.,1.,0.]],np.float64)

edge1_kernel = np.array(
                    [[-1.,-1.,-1.],
                    [-1.,8.,-1.],
                    [-1.,-1.,-1.]],np.float64)

sharpen_kernel = np.array(
                    [[0.,-1.,0.],
                    [-1.,5.,-1.],
                    [0.,-1.,0.]],np.float64)

unsharp_mask_kernel = np.array(
                    [[1.,4.,6.,4.,1.],
                    [4.,16.,24.,16.,4.],
                    [6.,24.,36.,24.,6.],
                    [4.,16.,24.,16.,4.],
                    [1.,4.,6.,4.,1.]],np.float64)

outx=CVision.custom_2d_kernel_conv(img, sobel_x_kernel.astype(np.float64))
plt.figure(figsize=(7,5))
plt.title("Sobel x")
CVision.cv2_to_plt(outx)

outy=CVision.custom_2d_kernel_conv(img, sobel_y_kernel.astype(np.float64))
plt.figure(figsize=(7,5))
plt.title("Sobel y")
CVision.cv2_to_plt(outy)

edge0=CVision.custom_2d_kernel_conv(img, edge0_kernel.astype(np.float64))
plt.figure(figsize=(7,5))
plt.title("Edge 0")
CVision.cv2_to_plt(edge0)

edge1=CVision.custom_2d_kernel_conv(img, edge1_kernel.astype(np.float64))
plt.figure(figsize=(7,5))
plt.title("Edge 1")
CVision.cv2_to_plt(edge1)

sharpen=CVision.custom_2d_kernel_conv(img, sharpen_kernel.astype(np.float64)).astype(np.uint8)
sharpen = apply_3_channels(img.copy(), cv2.addWeighted, (1.5,sharpen.copy(),-0.5,0))
plt.figure(figsize=(7,5))
plt.title("Sharpen")
CVision.cv2_to_plt(sharpen)

unsharp_mask=CVision.custom_2d_kernel_conv(img, unsharp_mask_kernel.astype(np.float64)).astype(np.uint8)
unsharp_mask = apply_3_channels(img.copy(), cv2.addWeighted, (1.5,unsharp_mask.copy(),-0.5,0))
plt.figure(figsize=(7,5))
plt.title("Unsharp mask")
CVision.cv2_to_plt(unsharp_mask)
{% endhighlight %}


<p><img src='computervision_images/output_8_0.png' /></p>


    converting image to grayscale



<p><img src='computervision_images/output_8_2.png' /></p>


    converting image to grayscale



<p><img src='computervision_images/output_8_4.png' /></p>


    converting image to grayscale



<p><img src='computervision_images/output_8_6.png' /></p>


    converting image to grayscale



<p><img src='computervision_images/output_8_8.png' /></p>


    converting image to grayscale



<p><img src='computervision_images/output_8_10.png' /></p>


    converting image to grayscale



<p><img src='computervision_images/output_8_12.png' /></p>


### Spectral phase correlation & Image matching 


{% highlight python %}
img1 = cv2.imread("image_dump/football.png")
img2 = cv2.imread("image_dump/football_subset.png")

CVision.cv2_to_plt(img1)
CVision.cv2_to_plt(img2)
{% endhighlight %}


<p><img src='computervision_images/output_10_0.png' /></p>



<p><img src='computervision_images/output_10_1.png' /></p>



{% highlight python %}
corr,pt1,pt2=CVision.sliding_phase_correlation(img1, img2, min_corr=0.95,plot=False)
{% endhighlight %}

    converting image to grayscale
    converting image to grayscale



{% highlight python %}
CVision.cv2_to_plt(cv2.rectangle(img1.copy(),pt1[2],pt2[2],color=(100,100,20),thickness=2))
plt.close()
{% endhighlight %}


<p><img src='computervision_images/output_12_0.png' /></p>



{% highlight python %}
img1 = cv2.imread("image_dump/text.jpeg")
img2 = cv2.imread("image_dump/d.jpeg")

CVision.cv2_to_plt(img1)
CVision.cv2_to_plt(img2)

corr,pt1,pt2=CVision.sliding_phase_correlation(img1, img2, min_corr=0.99,plot=False)
{% endhighlight %}


<p><img src='computervision_images/output_13_0.png' /></p>



<p><img src='computervision_images/output_13_1.png' /></p>


    converting image to grayscale
    converting image to grayscale



{% highlight python %}
[(c,(y1,x1)) for (c,(y1,x1)) in zip(corr,pt1)]
{% endhighlight %}




    [(0.99619059573419388, (437, 61)),
     (0.99504682782448051, (704, 101)),
     (0.99786064030551735, (150, 182)),
     (0.99850176916051248, (457, 182)),
     (0.99775032526574647, (472, 222)),
     (0.99626276035797323, (616, 222)),
     (0.99927231374288739, (317, 262)),
     (0.99695683843849492, (569, 262)),
     (1.0, (341, 303)),
     (0.99517423398257021, (49, 383)),
     (0.99671320239389627, (558, 383)),
     (0.99932607080059455, (269, 424)),
     (0.99731579729987352, (444, 424))]




{% highlight python %}
img3 = img1.copy()
for p1,p2 in zip(pt1,pt2):
    img3 = cv2.rectangle(img3,p1,p2,color=(100,100,20),thickness=2)

CVision.cv2_to_plt(img3)
plt.close()
{% endhighlight %}


<p><img src='computervision_images/output_15_0.png' /></p>


### Image signiture with average brightnesses (Image database or Exact matching)


{% highlight python %}
import urllib.request
file = open("/Users/rohankotwani/Downloads/craters.txt").read().split("\n")
i=0

opener=urllib.request.build_opener()
opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
urllib.request.install_opener(opener)

for link in file[:5]:
    print("link: ",link)
    req = urllib.request.urlopen(link)
    assert req.code==200
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    im_arr=cv2.imdecode(arr,1)
#     CVision.cv2_to_plt(im_arr)
    upperright,upperleft,lowerleft,lowerright=CVision.four_quadrant_indentifier(im_arr,plot=True)
    print("Average bright by quadrant")
    print("upper right: ",upperright,"upper left: ",upperleft,"lower left: ",lowerleft,"lower right: ",lowerright)
    print()
{% endhighlight %}

    link:  http://cdn.touropia.com/gfx/d/impact-craters-on-earth/roter_kamm_crater.jpg?v=186603f7e16e01c8707cffc7a3f316a2
    converting image to grayscale



<p><img src='computervision_images/output_17_1.png' /></p>


    Average bright by quadrant
    upper right:  133.377649273 upper left:  152.207747028 lower left:  125.235733627 lower right:  100.514892836
    
    link:  http://cdn.touropia.com/gfx/d/impact-craters-on-earth/kaali_crater.jpg?v=703f32a266af8e4060de68de27bf8df5
    converting image to grayscale



<p><img src='computervision_images/output_17_3.png' /></p>


    Average bright by quadrant
    upper right:  98.9226398642 upper left:  50.2815732709 lower left:  64.0264720731 lower right:  93.63510028
    
    link:  http://cdn.touropia.com/gfx/d/impact-craters-on-earth/tenoumer_crater.jpg?v=2dba62745baca63a37aa6cd2360b47b4
    converting image to grayscale



<p><img src='computervision_images/output_17_5.png' /></p>


    Average bright by quadrant
    upper right:  150.213603762 upper left:  141.676801881 lower left:  141.486725456 lower right:  177.433709583
    
    link:  http://cdn.touropia.com/gfx/d/impact-craters-on-earth/lonar_crater_lake.jpg?v=8d27e20391475940eba9f11d3386cd83
    converting image to grayscale



<p><img src='computervision_images/output_17_7.png' /></p>


    Average bright by quadrant
    upper right:  167.209426504 upper left:  174.706708022 lower left:  68.1118530682 lower right:  76.9298789759
    
    link:  http://cdn.touropia.com/gfx/d/impact-craters-on-earth/monturaqui_crater.jpg?v=1
    converting image to grayscale



<p><img src='computervision_images/output_17_9.png' /></p>


    Average bright by quadrant
    upper right:  114.12519703 upper left:  104.402924696 lower left:  73.1991253246 lower right:  98.5357933579
    


### Warping an Image


{% highlight python %}
cage = cv2.imread("image_dump/cage.jpeg")
for k in np.linspace(0,1,2):
    for i in np.linspace(.0,.2,2):
        for j in np.linspace(.0,.2,2):
            print(i,j,"flip: ",k)
            CVision.cv2_to_plt(CVision.im_warp_right(CVision.im_flip(cage,k),j,i))
{% endhighlight %}

    0.0 0.0 flip:  0.0



<p><img src='computervision_images/output_19_1.png' /></p>


    0.0 0.2 flip:  0.0



<p><img src='computervision_images/output_19_3.png' /></p>


    0.2 0.0 flip:  0.0



<p><img src='computervision_images/output_19_5.png' /></p>


    0.2 0.2 flip:  0.0



<p><img src='computervision_images/output_19_7.png' /></p>


    0.0 0.0 flip:  1.0



<p><img src='computervision_images/output_19_9.png' /></p>


    0.0 0.2 flip:  1.0



<p><img src='computervision_images/output_19_11.png' /></p>


    0.2 0.0 flip:  1.0



<p><img src='computervision_images/output_19_13.png' /></p>


    0.2 0.2 flip:  1.0



<p><img src='computervision_images/output_19_15.png' /></p>


### Object Matching & Image stitching


{% highlight python %}
importlib.reload(CVision)
tag1 = cv2.imread("star_bucks/star-bucks1.jpeg")
plt.figure(figsize=(7,5))
plt.title("Comparison image")
CVision.cv2_to_plt(tag1)

rsq_heap=[]
for i in range(2,9):
    tag2 = cv2.imread("star_bucks/star-bucks"+str(i)+".jpeg")
    img1,img2,is_match = CVision.im_stitcher(tag1,tag2,plot=False,warp_threshold=50000)
    msk = img2==0
    img1[msk] = 0
    
    
    n = len(img2[~msk].flatten())
    z = Regression.numpy_simple_regression(img2[~msk].astype(np.int16).reshape(n,1),img1[~msk].astype(np.int16).flatten())
    SSE = Regression.numpy_SSE(img2[~msk].astype(np.int16).reshape(n,1),img1[~msk].astype(np.int16).flatten(),z)
    SST = np.sum((img1[~msk].astype(np.int16)-np.mean(img1[~msk].astype(np.int16)))**2)
    

    plt.figure(figsize=(7,5))
    plt.title("Original image")
    CVision.cv2_to_plt(tag2)
    plt.figure(figsize=(7,5))
    plt.title("Warped image")
    CVision.cv2_to_plt(img2)
    print("image number: ",i)
    print("SSE: ",SSE)
    print("SST: ",SST)
    print("R-squared: ",1-SSE/SST)
    print("coefficients: ",z)
    plt.figure(figsize=(7,5))
    plt.title("Masked Comparision Image")
    CVision.cv2_to_plt(img1)

    rsq_heap.append((1-SSE/SST,i))
    
Regression.heapsort(rsq_heap)
{% endhighlight %}


<p><img src='computervision_images/output_21_0.png' /></p>


    converting image to grayscale
    converting image to grayscale



<p><img src='computervision_images/output_21_2.png' /></p>



<p><img src='computervision_images/output_21_3.png' /></p>


    image number:  2
    SSE:  3551685886.52
    SST:  5292258798.44
    R-squared:  0.328890361983
    coefficients:  [ 29.83408577   0.609802  ]



<p><img src='computervision_images/output_21_5.png' /></p>


    converting image to grayscale
    converting image to grayscale



<p><img src='computervision_images/output_21_7.png' /></p>



<p><img src='computervision_images/output_21_8.png' /></p>


    image number:  3
    SSE:  5104661730.63
    SST:  5266905152.71
    R-squared:  0.0308043181662
    coefficients:  [ 64.9953785    0.20453586]



<p><img src='computervision_images/output_21_10.png' /></p>


    converting image to grayscale
    converting image to grayscale



<p><img src='computervision_images/output_21_12.png' /></p>



<p><img src='computervision_images/output_21_13.png' /></p>


    image number:  4
    SSE:  1753778251.75
    SST:  5184161415.25
    R-squared:  0.661704543653
    coefficients:  [ 21.07826597   0.78775359]



<p><img src='computervision_images/output_21_15.png' /></p>


    converting image to grayscale
    converting image to grayscale



<p><img src='computervision_images/output_21_17.png' /></p>



<p><img src='computervision_images/output_21_18.png' /></p>


    image number:  5
    SSE:  2123726480.48
    SST:  2879680660.17
    R-squared:  0.262513198128
    coefficients:  [ 45.64177078   0.62979106]



<p><img src='computervision_images/output_21_20.png' /></p>


    converting image to grayscale
    converting image to grayscale



<p><img src='computervision_images/output_21_22.png' /></p>



<p><img src='computervision_images/output_21_23.png' /></p>


    image number:  6
    SSE:  1704742749.94
    SST:  1857715067.02
    R-squared:  0.0823443378382
    coefficients:  [ 83.04531061   0.35370221]



<p><img src='computervision_images/output_21_25.png' /></p>


    converting image to grayscale
    converting image to grayscale



<p><img src='computervision_images/output_21_27.png' /></p>



<p><img src='computervision_images/output_21_28.png' /></p>


    image number:  7
    SSE:  3834307772.0
    SST:  4052066526.41
    R-squared:  0.0537401725721
    coefficients:  [ 68.87566674   0.3136739 ]



<p><img src='computervision_images/output_21_30.png' /></p>


    converting image to grayscale
    converting image to grayscale



<p><img src='computervision_images/output_21_32.png' /></p>



<p><img src='computervision_images/output_21_33.png' /></p>


    image number:  8
    SSE:  3204617723.84
    SST:  4088705822.96
    R-squared:  0.21622687897
    coefficients:  [ 30.38643206   0.50402533]



<p><img src='computervision_images/output_21_35.png' /></p>





    [(0.030804318166220535, 3),
     (0.053740172572137146, 7),
     (0.082344337838234805, 6),
     (0.21622687896985049, 8),
     (0.2625131981279778, 5),
     (0.32889036198268728, 2),
     (0.66170454365331399, 4)]



### Seam Carving with Sobel energy matrix and dynamic programming


{% highlight python %}
import importlib
import CVision
importlib.reload(CVision)

img = cv2.imread("image_dump/tower.jpeg")
img = CVision.cv2_resize(img,(700,1000))
plt.figure(figsize=(7,5))
CVision.cv2_to_plt(img)
img2,seams = CVision.remove_n_vertical_seams(img.copy(),n=200,plot=True)
plt.figure(figsize=(7,5))
CVision.cv2_to_plt(img2)
{% endhighlight %}


<p><img src='computervision_images/output_23_0.png' /></p>



<p><img src='computervision_images/output_23_1.png' /></p>



<p><img src='computervision_images/output_23_2.png' /></p>



{% highlight python %}

{% endhighlight %}

