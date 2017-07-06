---
layout: post
title:  "Web Crawler / MySQL Backend"
date: 2017-07-06 12:00:00
author: Rohan Kotwani
excerpt: "Web data collection, ETL process into MySQL database"
tags: 
- Unstructured data
- ETL
- Web Crawler
- MySQL

---

```python
import datetime
import time
import pandas as pd
import httplib2
from itertools import zip_longest
import bs4
import codecs
import urllib
import re
import os
from os import walk
from time import gmtime, strftime 
import os.path
import ast
import numpy as np
import cv2
from ast import literal_eval
import pymysql
from sqlalchemy import create_engine
import json
import socket
import socks
import pandas.io.sql as psql
```


```python
import importlib
import Crawl
importlib.reload(Crawl)
```




    <module 'Crawl' from 'C:\\Users\\Wscraper\\Documents\\WebCrawler\\Crawl.py'>



### Yahoo news raw HTML database


```python
url = "https://www.yahoo.com/news"
scrape_date=strftime("%Y-%m-%d", gmtime())
        
def yahoo_news_meta_data(url, max_count=1000):
    df1 = pd.DataFrame()
    
    response = urllib.request.urlopen(url)
    assert response.code == 200
    html = response.read()
    print("reading pages")
    soup = bs4.BeautifulSoup(html,"html.parser")
    print(url)
    results = re.findall(re.compile('href="(.+?\.html)"'),str(soup))
    ad_count=0
    for x in results:
        if "news" in  x and "https://" in x:
            df1.loc[ad_count,"Link"] = x
            #print(x)
            ad_count+=1
        elif "news" in x and "https://" not in x:
            df1.loc[ad_count,"Link"] = url+x
            #print(x)
            ad_count+=1
    return df1

```


```python
#default_proxy()    

#connect_tor()

def scrape_page(url,meta_data):

    def get_raw_data(links,max_count=1000):
        ad_count= 0
        df2 = pd.DataFrame(columns=('Link','Ad_Source'))
        for url in links:
            try:
                response = urllib.request.urlopen(url)
                html = response.read()
                Ad_Source = bs4.BeautifulSoup(html,"html.parser")
                Ad_Link= url

                df2.loc[ad_count]=[Ad_Link,Ad_Source]
                ad_count = ad_count+1
            except:
                pass
                #print(url)

        time.sleep(np.random.randint(0,1,1)[0])
        return df2

    df1 = Crawl.webpage(url,yahoo_news_meta_data)
    print("len(df1):"+str(len(df1)))
    print("deleting duplicates")
    links = df1[['Link']].groupby(['Link']).count().reset_index()['Link']

    df2=pd.DataFrame()

    #split links into n chunks
    chunk_number = 1
    if len(links)>chunk_number:
        ranges = list(range(0, len(links), len(links) // chunk_number))
        split_ = list(zip_longest(ranges, ranges[1:], fillvalue=len(links)))

        for i in range(0,len(split_)):
            df2= pd.concat([df2,Crawl.webpage(links[split_[i][0]:split_[i][1]],get_raw_data)])
            print("len(df2):"+str(len(df2)))
    else: 
        df2= pd.concat([df2,Crawl.webpage(links,get_raw_data)])
        print("len(df2):"+str(len(df2)))
        
    return df2

def upload_database(df2,db_name):
    #default_proxy() 
    
    scrape_date=strftime("%Y-%m-%d", gmtime())
    connection = pymysql.connect(host='127.0.0.1',
                             user='rohank',
                             password='rohank',
                             db='local_database',
                             charset='utf8mb4',
                             autocommit=True,
                             cursorclass=pymysql.cursors.DictCursor)
    with connection.cursor() as cursor:
            # Create a new record
            sql = "CREATE TABLE IF NOT EXISTS `"+db_name+"` (`Scrape_Date` text,`Website` text,`JSON_Package` json) ;"
            cursor.execute(sql)
    connection.commit()

    df2['Scrape_Date'] = scrape_date
    df2['Website'] = url

    print("len(df2):"+str(len(df2)))
    for i in range(0,len(df2)):
        dict_upload={
          "Link":str(df2.ix[i].Link),
          "Ad_Source": str(df2.ix[i].Ad_Source)}
        #print(dict_upload)
        json_value = json.dumps(dict_upload)

        with connection.cursor() as cursor:
            # Create a new record
            cursor.execute("INSERT INTO `"+db_name+"` (`Scrape_Date`,`Website`, `JSON_Package`) VALUES (%s,%s,%s)", (df2.ix[i].Scrape_Date,df2.ix[i].Website,json_value))
        connection.commit()
    connection.close()

    #connect_tor()
    
url = "https://www.yahoo.com/news"
pandas_df = scrape_page(url,yahoo_news_meta_data)
upload_database(pandas_df,db_name='yahoo_news_database_raw')
```

    reading pages
    https://www.yahoo.com/news
    len(df1):75
    deleting duplicates
    len(df2):34
    len(df2):34


### Yahoo finance top mutual funds


```python

def crawl_pages(max_pages=1):
    scrape_date=strftime("%Y-%m-%d", gmtime())
    df = pd.DataFrame(columns=('Pad_Left','Symbol','Company','Change','Percent_Change','Price','50_Day_Avg','200_Day_Avg','3-Mo Return','YTD Return','Pad_Right'))
    #def get_stocks():
    j=0
    i=0
    while True:
        url = "https://finance.yahoo.com/screener/predefined/top_mutual_funds?offset="+str(i*100)+"&count=100"
        if i == max_pages:
            break
        elif i < max_pages:
            print(url)
            response = urllib.request.urlopen(url)
            html = response.read()
            soup = bs4.BeautifulSoup(html,"html.parser")
            table = soup.find("table",{"class":re.compile('screener*')})
            for row in table.findAll('tr',{"class":re.compile('data*')}):
                stock_info = [col.getText().strip() for col in row.findAll('td')]
                
                try:
                    assert len(stock_info)==len(df.columns)
                except:
                    continue
                
                df.loc[j] = [col.getText().strip() for col in row.findAll('td')]

                j+=1
            i+=1

        time.sleep(1)
    df['Scrape_Date'] = scrape_date
    return df

def upload_database(df2,db_name):
    #default_proxy() 
    
    scrape_date=strftime("%Y-%m-%d", gmtime())
    df2['Scrape_Date'] = scrape_date
    
    connection_s='mysql+pymysql://rohank:rohank@127.0.0.1:3306/local_database?charset=utf8mb4'
    engine = create_engine(connection_s, echo=False)
    df2.to_sql(name=db_name, con=engine, if_exists = 'append', chunksize=100, index=False)


stock_df = crawl_pages(max_pages=10)

def quick_format(x):
    try:
        return float(str(x).replace('%',''))/10
    except:
        print(x)
        return np.nan

stock_df['3-Mo Return'] = stock_df['3-Mo Return'].apply(lambda x: quick_format(x)) 
stock_df['Percent_Change'] = stock_df['Percent_Change'].apply(lambda x: quick_format(x)) 
stock_df['YTD Return'] = stock_df['YTD Return'].apply(lambda x: quick_format(x)) 

upload_database(stock_df,db_name='yahoo_stock_database')
stock_df.head()
```

    https://finance.yahoo.com/screener/predefined/top_mutual_funds?offset=0&count=100
    https://finance.yahoo.com/screener/predefined/top_mutual_funds?offset=100&count=100
    https://finance.yahoo.com/screener/predefined/top_mutual_funds?offset=200&count=100
    https://finance.yahoo.com/screener/predefined/top_mutual_funds?offset=300&count=100
    https://finance.yahoo.com/screener/predefined/top_mutual_funds?offset=400&count=100
    https://finance.yahoo.com/screener/predefined/top_mutual_funds?offset=500&count=100
    https://finance.yahoo.com/screener/predefined/top_mutual_funds?offset=600&count=100
    https://finance.yahoo.com/screener/predefined/top_mutual_funds?offset=700&count=100
    https://finance.yahoo.com/screener/predefined/top_mutual_funds?offset=800&count=100
    https://finance.yahoo.com/screener/predefined/top_mutual_funds?offset=900&count=100
    N/A
    N/A
    N/A
    N/A
    N/A
    N/A
    N/A
    N/A
    N/A
    N/A





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pad_Left</th>
      <th>Symbol</th>
      <th>Company</th>
      <th>Change</th>
      <th>Percent_Change</th>
      <th>Price</th>
      <th>50_Day_Avg</th>
      <th>200_Day_Avg</th>
      <th>3-Mo Return</th>
      <th>YTD Return</th>
      <th>Pad_Right</th>
      <th>Scrape_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td></td>
      <td>MIDNX</td>
      <td>Matthews India Instl</td>
      <td>0.40</td>
      <td>0.129</td>
      <td>31.12</td>
      <td>31.22</td>
      <td>29.07</td>
      <td>0.950</td>
      <td>2.033</td>
      <td></td>
      <td>2017-07-03</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>MINDX</td>
      <td>Matthews India Investor</td>
      <td>0.39</td>
      <td>0.126</td>
      <td>30.95</td>
      <td>31.05</td>
      <td>28.93</td>
      <td>0.944</td>
      <td>2.027</td>
      <td></td>
      <td>2017-07-03</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>FSCHX</td>
      <td>Fidelity Select Chemicals</td>
      <td>1.73</td>
      <td>0.110</td>
      <td>157.50</td>
      <td>157.41</td>
      <td>157.88</td>
      <td>-0.061</td>
      <td>0.951</td>
      <td></td>
      <td>2017-07-03</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>CNPIX</td>
      <td>ProFunds Consumer Goods UltraSector Inv</td>
      <td>1.15</td>
      <td>0.110</td>
      <td>104.97</td>
      <td>106.21</td>
      <td>100.72</td>
      <td>0.494</td>
      <td>1.609</td>
      <td></td>
      <td>2017-07-03</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>CNPSX</td>
      <td>ProFunds Consumer Goods UltraSector</td>
      <td>1.05</td>
      <td>0.109</td>
      <td>96.31</td>
      <td>97.50</td>
      <td>92.65</td>
      <td>0.469</td>
      <td>1.561</td>
      <td></td>
      <td>2017-07-03</td>
    </tr>
  </tbody>
</table>
</div>



### Yahoo news- Extract Transform Load


```python
def get_title(s):
    s = s.decode('utf-8')
    soup1=bs4.BeautifulSoup(s,"html.parser")
    try:
        tmp=soup1.find("h1",{"class":re.compile("Lh*")}).getText().strip()
        Field = tmp.replace(r'\r',' ').replace(r'\n',' ').replace(r'\t',' ')
    except AttributeError:
        Field=""
    return Field

def get_author(s):
    s = s.decode('utf-8')
    soup1=bs4.BeautifulSoup(s,"html.parser")
    try:
        tmp=soup1.find("a",{"class":re.compile("author-link*")},'itemprop="name"').getText().strip()
        Field = tmp.replace(r'\r',' ').replace(r'\n',' ').replace(r'\t',' ')
    except AttributeError:
        Field=""
    return Field

def get_post_date(s):
    s = s.decode('utf-8')
    soup1=bs4.BeautifulSoup(s,"html.parser")
    try:
        tmp=soup1.find("time",{"class":re.compile("Fz*")},'itemprop="datePublished"')["datetime"]
        Field = tmp.replace(r'\r','').replace(r'\n','').replace(r'\t','')
        Field= datetime.datetime.strptime(Field,'%Y-%m-%dT%H:%M:%S.%fZ').date()
    except AttributeError:
        Field="NULL"
    return Field


def get_post_time(s):
    s = s.decode('utf-8')
    soup1=bs4.BeautifulSoup(s,"html.parser")
    try:
        tmp=soup1.find("time",{"class":re.compile("Fz*")},'itemprop="datePublished"')["datetime"]
        Field = tmp.replace(r'\r','').replace(r'\n','').replace(r'\t','')
        Field= datetime.datetime.strptime(Field,'%Y-%m-%dT%H:%M:%S.%fZ').time()
    except AttributeError:
        Field="NULL"
    return Field

def get_post(s):
    s = s.decode('utf-8')
    soup1=bs4.BeautifulSoup(s,"html.parser")
    try:
        tmp=soup1.find("div",{"class":re.compile("canvas-body*")}).getText().strip()
        Field = tmp.replace(r'\r',' ').replace(r'\n',' ').replace(r'\t',' ')
    except AttributeError:
        Field="NULL"
    return Field

def get_provider(s):
    s = s.decode('utf-8')
    soup1=bs4.BeautifulSoup(s,"html.parser")
    try:
        tmp=soup1.find("span",{"class":re.compile("provider*")}).getText().strip()
        Field = tmp.replace(r'\r',' ').replace(r'\n',' ').replace(r'\t',' ')
    except AttributeError:
        Field="NULL"
    return Field

def get_images(s):
    s = s.decode('ascii','ignore')
    soup1=bs4.BeautifulSoup(s,"html.parser")
    try:
        Poster_images=[]
        tmp=soup1.findAll("img")
        for each in tmp:
            Poster_images.append(each["src"])
    except AttributeError:
        Poster_images="NULL"
    return str(Poster_images)

```


```python

extract_pipeline = [("Post",get_post),
                    ("Images",get_images),
                    ("Author",get_author),
                    ("Provider",get_provider),
                    ("Title",get_title),
                    ("Post_Date",get_post_date),
                    ("Post_Time",get_post_time)]

connection_s='mysql+pymysql://rohank:rohank@127.0.0.1:3306/local_database?charset=utf8mb4'
engine = create_engine(connection_s, echo=False)


to_db_name = "yahoo_news_database_ads"
from_db_name = "yahoo_news_database_raw"
chunk_size = 10
try:
    sql = "SELECT count(*) as set_point FROM "+to_db_name;
    offset = psql.read_sql(sql, engine)['set_point'].ix[0]
except:
    offset=0
print("offset: ",offset)
print("chunk size:",chunk_size)

def encode_utf8(s):
    return codecs.encode(s, 'utf-8')

while True:
    sql = "SELECT * FROM "+from_db_name+" limit %d offset %d" % (chunk_size,offset) 
    chunk=psql.read_sql(sql, engine)


    d = ast.literal_eval(chunk["JSON_Package"][0])
    #print(codecs.encode(d['Ad_Source'],'utf-8'))
    chunk['JSON_Package']=chunk['JSON_Package'].apply(lambda x: ast.literal_eval(x))
    

    chunk['Link'] = chunk['JSON_Package'].apply(lambda x: x['Link'])
    chunk['Ad_Source'] = chunk['JSON_Package'].apply(lambda x: x['Ad_Source'])
    chunk = chunk.drop(["JSON_Package"],axis=1)    
    chunk['Ad_Source']=chunk['Ad_Source'].apply(lambda x: encode_utf8(x))
    
    for field_name,function in extract_pipeline:
        chunk[field_name]=chunk['Ad_Source'].apply(lambda x: function(x))
    #chunk['Author']=chunk['Ad_Source'].apply(lambda x: get_author(x))
    chunk = chunk.drop(["Ad_Source"],axis=1)   
    #print(chunk['Ad_Source'][0])

    #print(chunk)
    
    chunk.to_sql(name=to_db_name, con=engine, if_exists = 'append', chunksize=10, index=False)
    offset += chunk_size
    if len(chunk) < chunk_size:
        break
```

    offset:  0
    chunk size: 10



```python
pd.read_sql(to_db_name, con=engine).head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Scrape_Date</th>
      <th>Website</th>
      <th>Link</th>
      <th>Post</th>
      <th>Images</th>
      <th>Author</th>
      <th>Provider</th>
      <th>Title</th>
      <th>Post_Date</th>
      <th>Post_Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-07-03</td>
      <td>https://www.yahoo.com/news</td>
      <td>https://www.yahoo.com/news/news/bill-create-pa...</td>
      <td>For months, House Minority Leader Nancy Pelosi...</td>
      <td>['https://s.yimg.com/uu/api/res/1.2/EgCSY2qYak...</td>
      <td>Michael Isikoff</td>
      <td>Yahoo News</td>
      <td>Bill to create panel that could remove Trump f...</td>
      <td>2017-06-30</td>
      <td>12:45:21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-07-03</td>
      <td>https://www.yahoo.com/news</td>
      <td>https://www.yahoo.com/news/news/body-missing-5...</td>
      <td>The Los Angeles County Sheriff’s Department co...</td>
      <td>['https://s.yimg.com/ny/api/res/1.2/NwJPdDKIEs...</td>
      <td></td>
      <td>International Business Times</td>
      <td>Body Of Missing 5-Year-Old Boy Found Near Lake...</td>
      <td>2017-07-02</td>
      <td>07:33:22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-07-03</td>
      <td>https://www.yahoo.com/news</td>
      <td>https://www.yahoo.com/news/news/chicago-school...</td>
      <td>CHICAGO (Reuters) - The cash-strapped Chicago ...</td>
      <td>['https://s.yimg.com/ny/api/res/1.2/1v57GsbtrM...</td>
      <td></td>
      <td>Reuters</td>
      <td>Chicago schools make partial payment to teache...</td>
      <td>2017-07-01</td>
      <td>00:41:36</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-07-03</td>
      <td>https://www.yahoo.com/news</td>
      <td>https://www.yahoo.com/news/news/chinas-heavy-l...</td>
      <td>China's new heavy-lift rocket launch fails in ...</td>
      <td>['https://s.yimg.com/g/images/spaceball.gif', ...</td>
      <td></td>
      <td>Yahoo News Video</td>
      <td>China's new heavy-lift rocket launch fails in ...</td>
      <td>2017-07-02</td>
      <td>18:24:27</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-07-03</td>
      <td>https://www.yahoo.com/news</td>
      <td>https://www.yahoo.com/news/news/congress-cool-...</td>
      <td>PORTLAND, Maine (AP) -- The summer air is sizz...</td>
      <td>['https://s.yimg.com/ny/api/res/1.2/jFJ4GVeXE6...</td>
      <td></td>
      <td>Associated Press</td>
      <td>Congress is cool to Trump's proposal to end he...</td>
      <td>2017-07-02</td>
      <td>23:59:42</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```
