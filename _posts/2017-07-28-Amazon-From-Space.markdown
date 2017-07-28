---
layout: post
title:  "Kaggle Competition: Understanding the Amazon from Space"
date: 2017-07-06 12:00:00
author: Rohan Kotwani
excerpt: "Classification of Land Sat Image with Simulated Images and Cross Correlation"
tags: 
- Seam Carving
- Object Detection
- Filtering
- Phase Correlation
- Perspective transforms

---

{% highlight python %}
import cv2
import matplotlib.pyplot as plt
from os import walk
import numpy as np
import pandas as pd

df = pd.read_csv("train_v2.csv")
df['count']=1
df.head(5)
{% endhighlight %}


{% highlight python %}
import importlib
import CVision
importlib.reload(CVision)
import Regression
importlib.reload(Regression)
{% endhighlight %}




    <module 'Regression' from 'C:\\Users\\Wscraper\\Documents\\Kaggle_Amazon_Forest\\Regression.py'>


### Define the Categories

{% highlight python %}
artisinal_mines=[x for x in df[['tags','count']].groupby(['tags']).sum().reset_index()['tags'] if 'artisinal_mine' in x and 'cloudy' not in x]
clear_slash_burn=[x for x in df[['tags','count']].groupby(['tags']).sum().reset_index()['tags'] if 'slash_burn' in x and 'clear' in x]

clear_agriculture_habitation=[x for x in df[['tags','count']].groupby(['tags']).sum().reset_index()['tags'] if 'agriculture' in x
                   and 'clear' in x and 'habitation' in x]
cloudy = ['cloudy']
{% endhighlight %}

### Simulation for each Category


{% highlight python %}
def get_simulation(tag_list):
    data = np.zeros((len(df[df.tags.isin(tag_list)]),256*256))
    m,n=data.shape[0],data.shape[1]
    print(m,n)
    i=0
    for pic,tag,count in df[df.tags.isin(tag_list)].values:
        y = cv2.imread("train-jpg/"+pic+".jpg",0)

        #plt.imshow(y,cmap='gray')
        #plt.show()
        data[i,:] = y[:,:].flatten()
        i+=1

    random_weight = np.random.lognormal(size=m).reshape((m,1))
    norm_weight = random_weight/np.sum(random_weight)
    norm_weight.shape[:]

    data=np.multiply(data,norm_weight)
    simulation=np.sum(data,axis=0)
    return simulation.reshape((256,256))


print("Clear Agriculture Habitation")
clear_agriculture_habitation_simulation =  get_simulation(clear_agriculture_habitation)
CVision.cv2_to_plt(clear_agriculture_habitation_simulation)

print("Cloudy")
cloudy_simulation =  get_simulation(cloudy)
CVision.cv2_to_plt(cloudy_simulation)

print("Artisinal Mines Not Cloudy")
artisinal_mines_simulation =  get_simulation(artisinal_mines)
CVision.cv2_to_plt(artisinal_mines_simulation)

print("Clear Slash Burn")
clear_slash_burn_simulation =  get_simulation(clear_slash_burn)
CVision.cv2_to_plt(clear_slash_burn_simulation)

{% endhighlight %}

    Clear Agriculture Habitation
    2301 65536
    


<p><img src='/amazon_images/output_3_1.png' /></p>


    Cloudy
    2089 65536
    


<p><img src='/amazon_images/output_3_3.png' /></p>


    Artisinal Mines Not Cloudy
    312 65536
    


<p><img src='/amazon_images/output_3_5.png' /></p>


    Clear Slash Burn
    173 65536
    


<p><img src='/amazon_images/output_3_7.png' /></p>




### Cross Correlation of Images with Simulated Categories

{% highlight python %}
for pic,tag,count in df[df.tags.isin(artisinal_mines+cloudy+clear_agriculture_habitation+clear_slash_burn)].values[:]:
    y = cv2.imread("train-jpg/"+pic+".jpg",0)
    assert y.shape[0]==y.shape[1]
    corr,pt1,pt2=CVision.sliding_phase_correlation(artisinal_mines_simulation, y, min_corr=0,plot=False)
    
    if len(corr)==0:
        continue
    
    corr_heap = []
    if corr[0]>0.08:
        corr_heap.append((corr,"artisinal mine"))
        
    corr,pt1,pt2=CVision.sliding_phase_correlation(cloudy_simulation, y, min_corr=0,plot=False)
    if corr[0]>0.08:
        corr_heap.append((corr,'cloudy"'))
    
    corr,pt1,pt2=CVision.sliding_phase_correlation(clear_agriculture_habitation_simulation, y, min_corr=0,plot=False)
    if corr[0]>0.08:
        corr_heap.append((corr,"clear agriculture habitation"))
        
    corr,pt1,pt2=CVision.sliding_phase_correlation(clear_slash_burn_simulation, y, min_corr=0,plot=False)
    if corr[0]>0.08:
        corr_heap.append((corr,"clear slash burn"))
    if len(corr_heap)>0:
        CVision.cv2_to_plt(y)
        print("Actual class: ",tag)
        print(Regression.heapsort(corr_heap))
{% endhighlight %}


<p><img src='/amazon_images/output_5_0.png' /></p>


    Actual class:  cloudy
    [([0.13063403099471033], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_2.png' /></p>


    Actual class:  cloudy
    [([0.087118535027878113], 'artisinal mines'), ([0.13656031401499211], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_4.png' /></p>


    Actual class:  cloudy
    [([0.096529488279200909], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_6.png' /></p>


    Actual class:  cloudy
    [([0.084277523097568155], 'artisinal mines'), ([0.18080938665164795], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_8.png' /></p>


    Actual class:  agriculture clear primary road slash_burn
    [([0.10418069071704575], 'clear slash burn')]
    


<p><img src='/amazon_images/output_5_10.png' /></p>


    Actual class:  cloudy
    [([0.12918492230301232], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_12.png' /></p>


    Actual class:  cloudy
    [([0.13520188856718429], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_14.png' /></p>


    Actual class:  artisinal_mine clear water
    [([0.10780454901105828], 'artisinal mines')]
    


<p><img src='/amazon_images/output_5_16.png' /></p>


    Actual class:  cloudy
    [([0.11771571067672056], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_18.png' /></p>


    Actual class:  agriculture clear cultivation primary slash_burn
    [([0.089064334010422463], 'clear slash burn')]
    


<p><img src='/amazon_images/output_5_20.png' /></p>


    Actual class:  cloudy
    [([0.090017785803405262], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_22.png' /></p>


    Actual class:  cloudy
    [([0.083445603864388096], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_24.png' /></p>


    Actual class:  agriculture artisinal_mine clear cultivation habitation primary road water
    [([0.15143355595183561], 'artisinal mines')]
    


<p><img src='/amazon_images/output_5_26.png' /></p>


    Actual class:  cloudy
    [([0.11690010567474317], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_28.png' /></p>


    Actual class:  cloudy
    [([0.081510632603191191], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_30.png' /></p>


    Actual class:  cloudy
    [([0.099232441721483183], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_32.png' /></p>


    Actual class:  agriculture clear primary slash_burn
    [([0.11852907359189235], 'clear slash burn')]
    


<p><img src='/amazon_images/output_5_34.png' /></p>


    Actual class:  agriculture clear habitation primary road
    [([0.090623408943844713], 'clear agriculture habitation')]
    


<p><img src='/amazon_images/output_5_36.png' /></p>


    Actual class:  cloudy
    [([0.085530640931990032], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_38.png' /></p>


    Actual class:  artisinal_mine bare_ground clear primary water
    [([0.16135650901161075], 'artisinal mines')]
    


<p><img src='/amazon_images/output_5_40.png' /></p>


    Actual class:  cloudy
    [([0.091506313517301124], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_42.png' /></p>


    Actual class:  agriculture clear primary road slash_burn water
    [([0.08446619827329381], 'clear slash burn')]
    


<p><img src='/amazon_images/output_5_44.png' /></p>


    Actual class:  cloudy
    [([0.12751763487196005], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_46.png' /></p>


    Actual class:  agriculture clear habitation primary road water
    [([0.083345853863848587], 'clear agriculture habitation')]
    


<p><img src='/amazon_images/output_5_48.png' /></p>


    Actual class:  cloudy
    [([0.14580039231987857], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_50.png' /></p>


    Actual class:  agriculture clear cultivation habitation primary road slash_burn
    [([0.096473949399975598], 'clear slash burn')]
    


<p><img src='/amazon_images/output_5_52.png' /></p>


    Actual class:  cloudy
    [([0.10523935600677808], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_54.png' /></p>


    Actual class:  clear cultivation primary slash_burn
    [([0.10301980916845993], 'clear slash burn')]
    


<p><img src='/amazon_images/output_5_56.png' /></p>


    Actual class:  cloudy
    [([0.12343897364504654], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_58.png' /></p>


    Actual class:  agriculture clear habitation primary road
    [([0.1357053344177622], 'clear agriculture habitation')]
    


<p><img src='/amazon_images/output_5_60.png' /></p>


    Actual class:  cloudy
    [([0.11820472090508875], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_62.png' /></p>


    Actual class:  cloudy
    [([0.09755289260325177], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_64.png' /></p>


    Actual class:  cloudy
    [([0.13162684290199489], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_66.png' /></p>


    Actual class:  cloudy
    [([0.081955272671638282], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_68.png' /></p>


    Actual class:  cloudy
    [([0.089599858307739533], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_70.png' /></p>


    Actual class:  cloudy
    [([0.094947669192607054], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_72.png' /></p>


    Actual class:  artisinal_mine clear primary road water
    [([0.089335582423688678], 'artisinal mines')]
    


<p><img src='/amazon_images/output_5_74.png' /></p>


    Actual class:  cloudy
    [([0.089476648776565845], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_76.png' /></p>


    Actual class:  cloudy
    [([0.10447321762521956], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_78.png' /></p>


    Actual class:  cloudy
    [([0.12125063520650525], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_80.png' /></p>


    Actual class:  cloudy
    [([0.12194140852334218], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_82.png' /></p>


    Actual class:  cloudy
    [([0.088562982168245319], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_84.png' /></p>


    Actual class:  cloudy
    [([0.091635314162840861], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_86.png' /></p>


    Actual class:  agriculture clear habitation primary road
    [([0.093081669761985675], 'clear agriculture habitation')]
    


<p><img src='/amazon_images/output_5_88.png' /></p>


    Actual class:  cloudy
    [([0.080246884182431102], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_90.png' /></p>


    Actual class:  artisinal_mine clear primary road water
    [([0.16770430765394156], 'artisinal mines')]
    


<p><img src='/amazon_images/output_5_92.png' /></p>


    Actual class:  cloudy
    [([0.081231022266089456], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_94.png' /></p>


    Actual class:  cloudy
    [([0.08919447814942047], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_96.png' /></p>


    Actual class:  cloudy
    [([0.10856759438112071], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_98.png' /></p>


    Actual class:  artisinal_mine clear primary water
    [([0.12964161736779997], 'artisinal mines')]
    


<p><img src='/amazon_images/output_5_100.png' /></p>


    Actual class:  cloudy
    [([0.12152410306326281], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_102.png' /></p>


    Actual class:  cloudy
    [([0.083996267037414168], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_104.png' /></p>


    Actual class:  agriculture clear habitation primary road
    [([0.27742388041341481], 'clear agriculture habitation')]
    


<p><img src='/amazon_images/output_5_106.png' /></p>


    Actual class:  artisinal_mine clear primary water
    [([0.15136975810489844], 'artisinal mines')]
    


<p><img src='/amazon_images/output_5_108.png' /></p>


    Actual class:  cloudy
    [([0.086632218964269203], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_110.png' /></p>


    Actual class:  cloudy
    [([0.094778162550398587], 'artisinal mines'), ([0.16963852945329347], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_112.png' /></p>


    Actual class:  artisinal_mine clear primary road water
    [([0.12077605716929062], 'artisinal mines')]
    


<p><img src='/amazon_images/output_5_114.png' /></p>


    Actual class:  cloudy
    [([0.083516495684947553], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_116.png' /></p>


    Actual class:  artisinal_mine clear primary water
    [([0.094576911517528062], 'artisinal mines')]
    


<p><img src='/amazon_images/output_5_118.png' /></p>


    Actual class:  agriculture clear habitation primary slash_burn
    [([0.4567353309590792], 'clear slash burn')]
    


<p><img src='/amazon_images/output_5_120.png' /></p>


    Actual class:  cloudy
    [([0.086958255868012369], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_122.png' /></p>


    Actual class:  artisinal_mine clear primary water
    [([0.17398256717901797], 'artisinal mines')]
    


<p><img src='/amazon_images/output_5_124.png' /></p>


    Actual class:  cloudy
    [([0.082597380512869911], 'artisinal mines'), ([0.14919842611747028], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_126.png' /></p>


    Actual class:  cloudy
    [([0.093006383921766606], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_128.png' /></p>


    Actual class:  cloudy
    [([0.085156552584637391], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_130.png' /></p>


    Actual class:  cloudy
    [([0.10593210350302112], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_132.png' /></p>


    Actual class:  cloudy
    [([0.10728089593737271], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_134.png' /></p>


    Actual class:  cloudy
    [([0.091264448616214577], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_136.png' /></p>


    Actual class:  cloudy
    [([0.11338740645434967], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_138.png' /></p>


    Actual class:  artisinal_mine clear primary road water
    [([0.22877258810028245], 'artisinal mines')]
    


<p><img src='/amazon_images/output_5_140.png' /></p>


    Actual class:  agriculture clear cultivation habitation primary road
    [([0.10224372610104372], 'clear agriculture habitation')]
    


<p><img src='/amazon_images/output_5_142.png' /></p>


    Actual class:  cloudy
    [([0.085904890033401071], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_144.png' /></p>


    Actual class:  cloudy
    [([0.084749742229728392], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_146.png' /></p>


    Actual class:  agriculture bare_ground clear habitation primary road
    [([0.086543645090960394], 'clear agriculture habitation')]
    


<p><img src='/amazon_images/output_5_148.png' /></p>


    Actual class:  agriculture clear cultivation habitation primary slash_burn
    [([0.10399066698934538], 'clear slash burn')]
    


<p><img src='/amazon_images/output_5_150.png' /></p>


    Actual class:  cloudy
    [([0.17840229681715872], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_152.png' /></p>


    Actual class:  cloudy
    [([0.11551638995216192], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_154.png' /></p>


    Actual class:  cloudy
    [([0.090519348539475949], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_156.png' /></p>


    Actual class:  cloudy
    [([0.11794497975454787], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_158.png' /></p>


    Actual class:  cloudy
    [([0.10902975791640905], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_160.png' /></p>


    Actual class:  cloudy
    [([0.097573811284065046], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_162.png' /></p>


    Actual class:  agriculture bare_ground clear habitation road
    [([0.098883408333931486], 'clear agriculture habitation')]
    


<p><img src='/amazon_images/output_5_164.png' /></p>


    Actual class:  cloudy
    [([0.097630454734436059], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_166.png' /></p>


    Actual class:  cloudy
    [([0.10473454421043207], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_168.png' /></p>


    Actual class:  cloudy
    [([0.12952979608823628], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_170.png' /></p>


    Actual class:  agriculture clear habitation primary water
    [([0.12102033130289222], 'clear agriculture habitation')]
    


<p><img src='/amazon_images/output_5_172.png' /></p>


    Actual class:  clear cultivation primary slash_burn water
    [([0.11748752728886377], 'clear slash burn')]
    


<p><img src='/amazon_images/output_5_174.png' /></p>


    Actual class:  artisinal_mine clear primary road
    [([0.10950161244168227], 'artisinal mines')]
    


<p><img src='/amazon_images/output_5_176.png' /></p>


    Actual class:  cloudy
    [([0.088266751240256305], 'cloudy"')]
    


<p><img src='/amazon_images/output_5_178.png' /></p>


    Actual class:  agriculture clear habitation primary road
    [([0.098697201381162503], 'clear agriculture habitation')]
    
