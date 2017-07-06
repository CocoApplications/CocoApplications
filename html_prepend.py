import sys


---
layout: post
title: Why and What
date: '2017-07-06T10:54:00.002-07:00'
author: Alex Rogozhnikov
tags: 
modified_time: '2013-07-20T11:37:54.534-07:00'
blogger_id: tag:blogger.com,1999:blog-307916792578626510.post-8650969116020988915
blogger_orig_url: http://brilliantlywrong.blogspot.com/2013/07/blog-post.html
---

with open(filename, 'r+') as f:
    content = f.read()
    f.seek(0, 0)
    f.write(sys.agrv[1].rstrip('\r\n') + '\n' + content)