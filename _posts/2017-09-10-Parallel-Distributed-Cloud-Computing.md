---
layout: post
title:  "Cloud computing & Parallel Programming with Jupyter notebooks"
date: 2017-09-10 12:00:00
author: Rohan Kotwani
excerpt: ""
tags: 
- Jupyter notebook
- Parallel Programming

---

## Table of Contents

1. Parallel Programming
2. Remote Access
3. PostgreSQL remote Access


## Introduction

This post demostrates Jupyter notebook's capabilities to access remote servers for parallel programming, file management, and database management.



### My ipcluster_config.py file has the following:
    
    c.IPClusterEngines.engine_launcher = \
    'IPython.parallel.apps.launcher.SSHEngineSetLauncher'


    c.SSHEngineSetLauncher.engines = { 
        'host@xxx.xxx.xxx.x02' : (2, ['--profile_dir=/home/to/profile'])
        }
        
### My ipcontroller_config.py file has the following:

    c.HubFactory.ip = '*'
    

#### upgrading some of your tools might fix the problem.

    pip install --upgrade ipython
    pip install --upgrade setuptools pip
    
    pip install  ipython[all]
    
#### The basic idea is that an sshserver arg to a Client is only for when the Controller is not directly accessible from the Client (remote location). Ssh tunnels are -required- when the machines are not accessible to each other.

First start the ipcontroller. (On MAC):

    (ipcontroller --profile=ssh --ip=* &)
    
Then start the ipcluster and ssh profile for the remote computer:

    (ipcluster start --profile=ssh &)
    
The remote clusters can also be stopped if the engines have died:

    (ipcluster stop --profile=ssh &)
    
Finally start local engines with correct parameters:

    (ipengine --profile=ssh &)

The security file might need to be shared from the controller to the engine:

    /Users/rohankotwani/.ipython/profile_ssh/security


{% highlight python %}
import ipyparallel as ipp
{% endhighlight %}


{% highlight python %}
c = ipp.Client(profile='ssh')
{% endhighlight %}


{% highlight python %}
c.ids
{% endhighlight %}




    [0, 1]




{% highlight python %}
c[:].apply_sync(lambda : "Hello, World")
{% endhighlight %}




    ['Hello, World', 'Hello, World']



# Steps to connect remotely to jupyter notebook

1. jupyter notebook --no-browser --port=8889 --ip=127.0.0.1 (remote host)
2. ssh -N -f -L 127.0.0.1:8889:127.0.0.1:8889 rohankotwani@probably-engine (local host)
3. http://127.0.0.1:8889/tree (local host web browser)



# Connect to Postgresql database remotely

1. modify the following file: /etc/postgresql/9.5/main/postgresql.conf
2. edit the following lines: listen_addresses = '*'
    
3. possibly modify the following file: /etc/postgresql/9.5/main/pg_hba.conf


{% highlight python %}
from sqlalchemy import create_engine
import pandas.io.sql as psql
connection_s='postgresql://username:password@ip-address:5432/mytestdb'
engine = create_engine(connection_s, connect_args={'sslmode':'require'}, echo=False)
psql.execute("CREATE TABLE AUG31 (first_column text,second_column integer);",engine) 
{% endhighlight %}




    <sqlalchemy.engine.result.ResultProxy at 0x109554208>




{% highlight python %}
import pandas as pd
pd.read_sql('select * from AUG31', con=engine).head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>first_column</th>
      <th>second_column</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




{% highlight python %}

{% endhighlight %}

