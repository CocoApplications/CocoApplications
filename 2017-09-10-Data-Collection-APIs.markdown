---
layout: post
title:  "Data Collection APIs for PredictHQ and Youtube"
date: 2017-09-10 12:00:00
author: Rohan Kotwani
excerpt: "Code"
tags: 
- Data Collection
- APIs
- Text Analysis

---

# Table of Contents

1. PredictHQ event data collection by location
2. Youtube API
3. Video metadata and comment collection
4. Text Analysis


# PredictHQ's event search API


{% highlight python %}
import pandas as pd
import datetime
{% endhighlight %}


{% highlight python %}
import pandas as pd
coords = pd.read_csv("major_us_cities.csv")
coords[coords.City=='New Orleans']
{% endhighlight %}


{% highlight python %}
lat,lng = 29.951066,-90.071532
{% endhighlight %}


{% highlight python %}
import datetime

start_datetime = datetime.datetime(2017, 7, 3, 10,0,0)
end_datetime = datetime.datetime(2017, 7, 4, 4,0,0)
America_TZ_date = datetime.date(2017, 7, 3)

date_range=[]
date_range=[]

while America_TZ_date <= datetime.date(2017, 7, 4):
    date_range.append((America_TZ_date,start_datetime,end_datetime))
    America_TZ_date+= datetime.timedelta(1)
    start_datetime+= datetime.timedelta(1)
    end_datetime+= datetime.timedelta(1)
{% endhighlight %}


{% highlight python %}
import requests
df = pd.DataFrame()

for rank in range(1,6):
    for date,start,end in date_range:
        response = requests.get(
            url="https://api.predicthq.com/v1/events/",
            headers={
              "Authorization": "Bearer P1glaeNO672SVMZv6KYdoR41dCXDPv",
              "Accept": "application/json"
            },
            params={
                "country": "US",
                "rank_level":str(rank),
                "within":"10km@"+str(lat)+","+str(lng),
                "active.gte":str(start).replace(" ","T"),
                "active.lte":str(end).replace(" ","T")
            }
        )
        tmp = pd.DataFrame(response.json()['results'])
        tmp['Date'] = date
        df = pd.concat([df,tmp])

df=df.reset_index(drop=True)
{% endhighlight %}


{% highlight python %}
import ast
df['lat'] = df['location'].apply(lambda x: ast.literal_eval(str(x))[1])
df['lng'] = df['location'].apply(lambda x: ast.literal_eval(str(x))[0])
{% endhighlight %}


{% highlight python %}
import numpy as np
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = [x*np.pi/180 for x in [lon1, lat1, lon2, lat2]] 

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


city = coords[coords.City.isin(['New Orleans'])].values.flatten()
city_name = city[1:2]
city_coords = city[2:].astype(np.float)

print("Kilometers of events from New Orleans' center")
for i in range(0,len(df)):
    lat1,lng1 = [x for x in city_coords]
    lat2,lng2 = df.ix[i][['lat','lng']].values.astype(np.float)
    print(haversine(lng1, lat1, lng2, lat2))

{% endhighlight %}

# Youtube API


{% highlight python %}
import youtube_api
reload(youtube_api)
{% endhighlight %}




    <module 'youtube_api' from 'youtube_api.pyc'>




{% highlight python %}
videoId = 'Xq6i78KGk9w'
df = pd.DataFrame()
pageToken = None

while True:
    x = youtube_api.comment_threads_list_by_video_id(part='snippet,replies',videoId='Xq6i78KGk9w',pageToken=pageToken)

    
    if u'nextPageToken' in x.keys():
        pageToken = x['nextPageToken']
        
    pages = x['pageInfo']['totalResults']
    assert len(x['items'])==x['pageInfo']['totalResults']


    for i in range(pages):
        item = x['items'][i]
        top_comment_df = pd.DataFrame(item['snippet']['topLevelComment']['snippet'])

        df = pd.concat([df,top_comment_df])
        if u'replies' in item.keys():
            
            for reply in item['replies']['comments']:
                reply_df = pd.DataFrame(reply['snippet'])
                df = pd.concat([df,reply_df])
                
    if u'nextPageToken' not in x.keys():
        print("not in keys")
        break
                
{% endhighlight %}


{% highlight python %}
def pandas_get_comments(videoID):
    df  = pd.DataFrame()
    pageToken = None
    while True:
        x = youtube_api.comment_threads_list_by_video_id(part='snippet,replies',videoId=videoID,pageToken=pageToken)


        if u'nextPageToken' in x.keys():
            pageToken = x['nextPageToken']

        pages = x['pageInfo']['totalResults']
        assert len(x['items'])==x['pageInfo']['totalResults']


        for i in range(pages):
            item = x['items'][i]
            top_comment_df = pd.DataFrame(item['snippet']['topLevelComment']['snippet'])

            df = pd.concat([df,top_comment_df])
            if u'replies' in item.keys():

                for reply in item['replies']['comments']:
                    reply_df = pd.DataFrame(reply['snippet'])
                    df = pd.concat([df,reply_df])

        if u'nextPageToken' not in x.keys():
            print("not in keys")
            return df
def pandas_get_playlist(playlistId):
    df = pd.DataFrame()
    pageToken = None
    while True:

        playlist = youtube_api.playlist_items_list_by_playlist_id(part='snippet,contentDetails',
                maxResults=1,playlistId=playlistId,pageToken=pageToken)

        if u'nextPageToken' in playlist.keys():
            pageToken = playlist['nextPageToken']

        if u'resourceId' not in playlist['items'][0]['snippet'].keys():
            break

        videoId = playlist['items'][0]['snippet']['resourceId']['videoId']

        try:
            video_df = pandas_get_comments(videoId)
            df = pd.concat([df,video_df])
        except:
            print("ERROR: ",playlist['items'][0]['snippet'])

            
        if u'nextPageToken' not in playlist.keys():
            print("not in keys")
            break
    return df
{% endhighlight %}


{% highlight python %}
channel_id = 'UCwWhs_6x42TyRM4Wstoq8HA'
df = pd.DataFrame()
channel = youtube_api.channel_sections_list_by_id(part='snippet,contentDetails',channelId=channel_id)
for content in channel['items'][:]:
    
    if u'contentDetails' not in content.keys():
        continue

    playlists = content['contentDetails']['playlists']
    
    for playlistId in playlists:
        print(playlistId)
        try:
            playlist_df = pandas_get_playlist(playlistId)
            df = pd.concat([df,playlist_df])
        except:
            print("ERROR: ",content['contentDetails']['playlists'])

{% endhighlight %}


{% highlight python %}
df = df.drop_duplicates()
{% endhighlight %}


{% highlight python %}
len(df.authorDisplayName.unique())
{% endhighlight %}




    47966




{% highlight python %}
df.to_csv("trevor_noah_daily_show_comments.csv",encoding='utf-8')
{% endhighlight %}


{% highlight python %}
for a in df[df.likeCount>1000]['textDisplay']:
    print(a)
    print
{% endhighlight %}

    The daily show is on break so that&#39;s why they are posting old videos. In case anyone wants to know
    
    Because people don&#39;t care about racism if it doesn&#39;t negatively affect them.﻿<br /><br />Edit: Wow, I&#39;m actually appalled by how many white people are making excuses for their apathy and racism in the comments. The fuck is wrong with you guys?﻿
    
    This is crazy, Conservatives would rather claim that these are &quot;Paid Protesters&quot; with out any evidence, rather than do the mature thing and look at the facts
    
    How to spot a paid protester: They are informed the issues they care about. Yup. Sounds about right, a TRUE non-paid protester goes out to protest with no idea whats going on....(oh wait)
    
    Soooooo....every time someone brings up a relevant issue - a legitimate Trump scandal - it must be fake? seems legit...
    
    This title is bait for the right wingers. This is gonna be a fun comment section.
    
    These are real protestors! The only thing that&#39;s fake is Trump&#39;s vomit-inducing tan.
    
    &quot;Is that in this district?&quot; &quot;No&quot; DEAD
    
    &quot;Bringing dildos onto a college campus, it&#39;s very vulgar, it&#39;s very obscene, and I think waving a penis around is quite immature.  I mean, we are talking about college students who probably haven&#39;t matured, yet.&quot;<br />&quot;Should you have a gun if you haven&#39;t fully matured?&quot;<br />&quot;Yes.  Immature people can still be very responsible.&quot;<br />Woooooow. 😳 According to this fellow, there are immature-yet-responsible people in the world.
    
    That inbred redneck contradicts himself with every statement he spew ...
    
    why were the old men protesting at a college they don&#39;t go to?
    
    Four hours to learn... what the fuck.
    
    When even people in Texas are wary of guns, you know your country has a major gun problem.
    
    On behalf of black people I apologize
    
    The same way Muslims don&#39;t want to be connected to terrorists, I, as I Christian don&#39;t want to be connected with these Morons
    
    I think Christians voted for Trump because they knew he&#39;d screw things up so bad, Jesus would have to come back and save them.
    
    As a Muslim even I&#39;m smart enough to know not all Christian are like this wish they could say the same for us 😒﻿ as I said not all
    
    Don&#39;t be stupid. Muslims have better things to do other then trying to sneak to your house. We work, study, and live with morals which teach us not to judge someone by their religion, color or ethnicity.
    
    As A Christian I apologize for these people. Please don&#39;t think they speak for all of us, please.
    
    Did I just read divinity and Donald Trump in the same sentence? Maybe he&#39;s alternative Christian.
    
    My heart dropped when she said &quot;Muslims&quot;<br />Stop making reputation worse for all the good Muslims out there :&#39;(<br />Stop making reputation worse for all the good non-racist white people out there :&#39;(
    
    Watching these folk being interviewed, I&#39;m struck by only one thought: how can people be so fucking dumb?
    
    I love how most of these people don&#39;t even relise they&#39;re being roasted on the spot😂.
    
    I can&#39;t tell if this is funny or sad.
    
    I LOVE how Jordan constantly fucks with people. How he always trips them up with their ridiculous thoughts and beliefs!!
    
    Trump in October: I&#39;m gonna drain the swamp<br />Trump in November/December: THIS IS MY SWAMP!!!
    
    This election has really shown me something about this country. Racism is alive and clearly well. This country is filled with millions of idiots like these people. FACTS literally don&#39;t mean a fucking thing in today&#39;s day and age. People would rather talk about emails than all the many other wrong doings the other candidate has done. Media cares about ratings more than anything else. Republicans would pick party over country any day of the week. How many fake Christians there are in this country. Polls don&#39;t mean shit. We&#39;d rather go backwards rather than forwards. Watching the polls come in is literally making me feel sick. I can&#39;t even laugh at funny videos on social media right now. The world is watching us and they&#39;re seeing this shit. I never thought a day like this would come.
    
    I&#39;ll be honest. I can not belive how we got in this situation.
    
    how many people are scared right now.
    
    Will Hillary stop trump. Will trump build the wall. Will there be a civil war. Find out on the Season finale of MERICA!!!!
    
    waiting for the trevor reaction on the results.
    
    Wait so you&#39;re saying Fox News ISN&#39;T a comdey channel?
    
    Gotta hand it to Fox News, with all the racism against African Americans, Muslims and Hispanics going on the Asians were really beginning to feel left out. Way to be inclusive.
    
    &quot;I&#39;m from Queens&quot;<br />Love her.
    
    This is what the country has come to. A comedy show has to do real reporting because a &quot;News&quot; show thinks it can just make fun of people.
    
    The Daily Show is constantly killing it! This country keeps fucking up and you guys handle it in the best way possible. Thanks!!
    
    On behalf of Americans with more than 2 brain cells, I apologize for this despicable behavior.﻿
    
    Dude, who the f**k even watches Fox News? that sh*t is utter trash.
    
    Holy shit this is so racist and NOT funny. I&#39;m so glad I&#39;m smart enough not to watch Fox News.
    
    Want to know about what Irish people think about Brexit? Do interviews in Boston. Same logic lol
    
    &quot;China town is nothing like China&quot; <br />Louder for the Americans in the back please!
    
    Where was Obama during 9/11?<br />Where was Obama during the Vietnam war?<br />Where was Obama during Pearl Harbor?<br />Where was Obama during the Trial of Tears?<br />Find out next time on Dragon Ball Z.
    
    Where the hell was Barack Obama when Pearl Harbor happened?
    
    Jordan is SO good at making people look stupid<br /><br /><br /><br /><br /><br /><br /><br /><br />Although to be fair, Trump&#39;s supporters do that pretty well on their own. Shit writes itself
    
    I could watch a 2 hour comedy bit completely comprised of Trump supporter&#39;s theories and speculations, featuring Jordan Klepper
    
    You can tell Jordan was REALLY close to snapping on someone.
    
    This is honestly just more proof that Trump supporters don&#39;t know what the fuck they are doing.
    
    I didn&#39;t think it was possible to be this dumb....
    
    &quot;Do you think a woman is fit to be president?&quot;<br />&quot;no, women have lots of hormones so they could start a war in 10 seconds.&quot; <br />&quot;Weren&#39;t all wars started by men?&quot;<br /><br />OHHHHHHHHHHHHHHHHHHHHHH
    
    All of them are uneducated it&#39;s fucking hilarious.
    
    Sometimes, I don&#39;t know whether to laugh or jump off a fucking cliff.
    
    It&#39;s almost as if Trump supporters are dumb people. Imagine that..
    
    I can&#39;t believe Parks and Recreation is real..
    
    It makes me wanna kill myself. And I&#39;m not even American.<br />Listening to these people is honestly quite sickening.
    
    Trump loves the uneducated!
    
    This is why aliens won&#39;t talk to us.
    
    &quot;I love the poorly educated!&quot; - Trump
    
    The guy with the Hillary T-shirt really did not get it...
    
    This actually scares me.
    
    When it was NOT called &quot;AMERICA&quot;. When the Native people of this land lived freely before it was stolen from them.  That is when this land was &#39;great&#39;.
    
    Basically America was great when White folks was in power and doing shit to other groups. - fuck.
    
    make america great again (for white men)
    
    A few Hick ups: &quot;slavery&quot; &quot;the indian thing&quot;
    
    lol @ the guy at the end. Congratulations, u played yourself. 👍🏾
    
    only Trump supporters would describe slavery and native american genocide as &#39;a few hiccups&#39;
    
    &quot;... without breaking a few pieces of China&quot;<br />HAHAHAHAHAHA
    
    <a href="http://www.youtube.com/watch?v=uVQvWwHM5kM&amp;t=0m11s">0:11</a> <br />&quot;Back when women couldn&#39;t vote?&quot;<br />&quot;Yeah, nice&quot;
    
    THAT LAST GUY HAS ME IN HYSTERICS! &quot;That wasn&#39;t an insult btw&quot; WHAT?! WHAT?! WHAT?!
    
    When was America great?<br />When it was founded<br />Except the slavery stuff.<br />Fuck.
    
    Canada is NOT a &quot;melting pot&quot; that is the USA thing. Canada&#39;s metaphor has been a &quot;mosaic.&quot; That difference matters. In Canada the idea is that immigrants should keep their culture and share it with the rest of us. This is supposed to be similar to the way in which a mosaic is made up of different coloured stones. We are not a US &quot;melting pot&quot; where every stone is supposed to melt into the same thing.
    
    Thank you, Canadians for letting my family members in 1956. It&#39;s too bad my country doesn&#39;t remember we used to be refugees too... a Hungarian.
    
    &quot;We don&#39;t blame all Americans for Donald Trump&quot; &quot;You should&quot; :D
    
    Wouldn&#39;t it be unfortunate if one day America goes into a crisis and people needed to leave the country to be safe, but no one would take them in?
    
    Random woman has a good point: Are you seriously going to label an entire group of people as terrorists from the possible actions of one person?
    
    How did that guy get elected?  I thought anyone that made any logical sense or had a conscience was barred for life from any kind of politics.
    
    I&#39;m pretty sure that guy was joking when he said Junebug was a soccer player zzzzz
    
    Gavin was trolling them, the interviewer didn&#39;t get it. Then they just cut it out and tried to make him look retarded.
    
    I would rather share a bathroom with a transgender woman than a child,children are more likely to peek under the stall door,lol
    
    I don&#39;t know about anyone else, but when I use a public bathroom, there&#39;s usually only one thing I need to think about and it&#39;s not who else is in that bathroom
    
    I am a bit curious why some people are afraid that trans women would wave their cocks around in women&#39;s bathrooms. Do they base this on how they themselves behave in the men&#39;s room?
    
    i want more interview with trump supporters!!! its just comedy fucking gold!
    
    The ALERRT guy&#39;s quiet calm gives me the impression that he&#39;s one of those hardcore badasses who been in the shit many a time
    
    If you believe &quot;black lives matter&quot; is a racist statement, then you must also believe &quot;I enjoy tea&quot; is an anti-soda statement.
    
    I&#39;m calling black people I can&#39;t stop laughing 😂😂😂😂
    
    Come on, what is it with these dead audiences lately, they barely laughed. I thought it was hilarious, these two are funnier than I expected haha
    
    Nerdy Suge Knight lol
    
    As a Dutch citizen I can say that Geert Wilders would never be in charge of our next government because other parties refused to work with him beforehand because of his comments about muslims. Still fun to see that a big country like the US notices our little election though :)
    
    bad hair = bad leaders
    
    We as a nation really should be paying attention to other countries and their elections. Isolationism and radicalism have been proven to be terrible for prosperity throughout history.
    
    Well Donald, it&#39;s Wednesday... WE&#39;RE WAITING.
    
    Colbert and Trevor are back to roast Trump and Putin. Life is good.
    
    Bernie beats Trump in double digits. Thanks for not mentioning that. Eye roll
    
    i am HOWLING at the ham/bacon guy
    
    This was Trevor Noah&#39;s most dangerous bit lol
    
    This guy is getting better and better! Never doubted you Trevor Noah!!!
    
    This guy goes through 3 accents in one sentence
    
    Uganda be kidding me
    
    So glad Jon&#39;s successor is funny as well
    
    Trevor stop, you&#39;re breaking the 13th Amendment, you shouldn&#39;t own people like that.
    
    This is a normal Republican. I was beginning to forget what they were like
    
    he seems so sane lmao. poor him.
    
    i would much rather john kasich use the presidency to get free food than trump using it to get free money. <b>cough cough</b> venezuela.
    
    we could have this guy ... but noo we had to try something new... 100 days in I want a refund please
    
    I&#39;m not a republican, but this guy would&#39;ve been a better choice than Trump (then again almost anyone would be better).
    
    Now Obamacare will change to Trumpdoesntcare.
    
    Keegan has the energy of a hamster on coffee.
    
    I got my eye on u pussy grabber😂😂😂😂
    
    &quot;We&#39;re not paying for that fuckken wall&quot; -Mexico
    
    You can&#39;t deny how talented and informed Trevor is. One of the best comedians out there! Dude getting laughs between takes!
    
    49 times? Holy shit. I&#39;m 29 and white and have been driving since I was 16, and I haven&#39;t ever been stopped. I didn’t know black people got pulled over that much. How can people argue that race isn&#39;t an element in this whole thing?
    
    the man had his little baby daughter in the backseat, what danger could a father with his daughter could be? obviously theres a black man problem in this country, people are so scared of black men that even a father with his daughter who is a legal gun owner who has the constitutional right to own a weapon is seen a potential threat. <br /><br />dylan roof who is a terrorist was armed up the wazoo when he shot 9 people and he wasnt shot and killed, a Jeremy Christian was wielding a knife on a train when he killed 2 people and wasnt shot by police, Philando Castile&#39;s only crime was having a broken tail light and he gets shot for reaching for his ID. a normal routine stop turned into a bloody scene for no good reason.
    
    Great, now I&#39;m another white guy shocked that Trevor&#39;s been pulled over 8 times. That&#39;s just insane to me, the problem is so clearly systemic
    
    Love the Between the Scenes videos, Trevor&#39;s personal commentary is so genuine I love hearing his thoughts, he&#39;s such a smart and experienced being.
    
    His behind the scenes are better than the actual show. Damn.
    
    “Our inequality materializes our upper class, vulgarizes our middle class, brutalizes our lower class.” –MATTHEW ARNOLD, ENGLISH ESSAYIST (1822-1888)
    
    If Obama did some shit like this there would be another civil war..
    
    When an intellectual from  South Africa that grew up in Apartheid is literally capable of comparing his homes Dictator to our current President... You know shits gotten really fucking real
    
    Q: What did Trump tell Melania when he couldn&#39;t find his viagra?<br /><br /><br />A: Melania, the erection is rigged
    
    Counting down the days until impeachment
    
    Hilarious as always. This guy is genuinely funny. Nothing is forced. This has got to be my second best channel of all time.
    
    Trevor Noah&#39;s impromptu segments are hysterical!! There should be more of them!
    
    Man this guy has a lot of talent and i feel like it only truly manifests itself when he is doing some freestyling, just pure unscripted stuff. Love it
    
    REPEAL AND REPLACE DONALD TRUMP
    
    Amazing how a bunch of political pundits fall so easily to a mere &quot;change in tone&quot; and all the late show hosts/comedians are the ones who see through the bullshit.
    
    Trevor Noah&#39;s non-scripted between the scenes dialogues give people a good look into just how talented of a story teller and comedian he actually is.
    
    How could you not like this guy?
    
    I love how Trevor absolutely adores and is proud of his home country &amp; when he speaks in his native tongue he&#39;s even hotter!
    



{% highlight python %}
df = pd.read_csv("trevor_noah_daily_show_comments.csv",encoding='utf-8')
{% endhighlight %}


{% highlight python %}
import nltk 
from nltk.corpus import stopwords
from string import punctuation

def one_gram(s):
    stopset = set(stopwords.words('english'))
    tokens = s.replace("b'","").replace("'","").lower().split(" ")
    cleanup = [" ".join(tokens[0+i:1+i]).strip().lower() for i,token in enumerate(tokens)
              if " ".join(tokens[0+i:1+i]).strip()!='' and token.lower() not in stopset and  len(token)>2]
    return cleanup

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)
def remove_rn(s):
    return (",".join([t.rstrip() for t in s.splitlines() ])).replace("\t","")

def remove_spaces(x):
    lenx = len(x)
    count = 0
    spaces = ""
    for i in x:
        if i == " ":
            count+=1
            spaces+=" "
        elif i != " " and count > 0:
            x = x.replace(spaces," ")
            count = 0
            spaces = ""
    return x

def three_gram(s):
    stopset = set(stopwords.words('english'))
    tokens = s.replace("b'","").replace("'","").lower().split(" ")
    cleanup = [" ".join(tokens[0+i:1+i]+tokens[1+i:2+i]+tokens[2+i:3+i]).strip().lower() for i,token in enumerate(tokens)
              if " ".join(tokens[0+i:1+i]+tokens[1+i:2+i]+tokens[2+i:3+i]).strip() != '' and
              len(" ".join(tokens[0+i:1+i]+tokens[1+i:2+i]+tokens[2+i:3+i]).strip().split())==3]
    return cleanup
{% endhighlight %}


{% highlight python %}
corpus = df[['textDisplay','videoId']].groupby(['videoId']).sum()

for a in corpus["textDisplay"][1:2]:
    print()
    x = remove_spaces(remove_rn(strip_punctuation(a)).replace(","," "))
    
{% endhighlight %}




{% highlight python %}
fdist1= nltk.FreqDist(one_gram(x))
dict(fdist1.most_common(100))
{% endhighlight %}




    [('cuba', 27),
     ('peace', 15),
     ('people', 13),
     ('know', 11),
     ('cuban', 11),
     ('world', 11),
     ('american', 10),
     ('one', 10),
     ('israel', 9),
     ('say', 9),
     ('government', 9),
     ('also', 8),
     ('america', 8),
     ('get', 8),
     ('would', 8),
     ('got', 8),
     ('illegal', 7),
     ('good', 7),
     ('local', 7),
     ('like', 7),
     ('embargo', 6),
     ('done', 6),
     ('signed', 6),
     ('years', 6),
     ('it39s', 6),
     ('feel', 6),
     ('russian', 6),
     ('culture', 6),
     ('new', 6),
     ('shit', 6),
     ('you39re', 6),
     ('see', 6),
     ('country', 6),
     ('president', 5),
     ('want', 5),
     ('law', 5),
     ('bern', 5),
     ('come', 5),
     ('naval', 5),
     ('countries', 5),
     ('usa', 5),
     ('order', 5),
     ('didn39t', 5),
     ('many', 5),
     ('libya', 5),
     ('whole', 5),
     ('mean', 5),
     ('even', 5),
     ('don39t', 5),
     ('agreement', 5),
     ('show', 5),
     ('efforts', 5),
     ('use', 5),
     ('democracy', 4),
     ('trump', 4),
     ('far', 4),
     ('venezuela', 4),
     ('accords', 4),
     ('middle', 4),
     ('i39m', 4),
     ('context', 4),
     ('land', 4),
     ('white', 4),
     ('there39s', 4),
     ('talk', 4),
     ('cubans', 4),
     ('gets', 4),
     ('international', 4),
     ('bernie', 4),
     ('first', 4),
     ('cant', 4),
     ('trevor', 4),
     ('simply', 4),
     ('that39s', 4),
     ('camp', 4),
     ('bring', 4),
     ('treaty', 4),
     ('state', 4),
     ('election', 4),
     ('thats', 4),
     ('name', 4),
     ('david', 4),
     ('play', 4),
     ('arab', 3),
     ('family', 3),
     ('right', 3),
     ('rent', 3),
     ('sense', 3),
     ('look', 3),
     ('invade', 3),
     ('soccer', 3),
     ('hand', 3),
     ('911', 3),
     ('part', 3),
     ('another', 3),
     ('development', 3),
     ('baby', 3),
     ('less', 3),
     ('going', 3),
     ('makes', 3)]




{% highlight python %}

vid_df = pd.DataFrame()
for videoId in df.videoId.unique():
    print(videoId)
    x = youtube_api.videos_list_by_id(part='snippet,contentDetails,statistics',
        id=videoId)
    video_details = pd.concat([pd.Series(x['items'][0]['contentDetails']),pd.Series(x['items'][0]['snippet']),pd.Series(x['items'][0]['statistics']),pd.Series({'videoId':x['items'][0]['id']})])

    vid_df = pd.concat([vid_df,pd.DataFrame(video_details).T])
vid_df
{% endhighlight %}

    ZsHwtfcZoY4
    gbKhx_dlwCs
    oANUXY3xXKM
    UbJKhvmshGs
    1YFLBjR-swo
    CcSh2F8e__8
    MIhcVon9ruo
    iAQnXnQQCCI
    SCRY_tOPQ7Q
    7lex_1MLrR8
    rX8jZTN0CdU
    eFQhw3VVToQ
    Y4Zdx97A63s
    uVQvWwHM5kM
    8aeEQw73uDg
    rwfM5LGMmxg
    9gT-vJg-EfM
    wBhlyBrtB90
    x9GqWoy6__s
    PIvCh3EQv1Q
    5NMK88czOug
    eSwbG5V5S-8
    MCI4bUk4vuM
    CqR0HM7Z2gM
    mQsNkt9yuKI
    3QIWolLM9i8
    VW1pdY3sNcA
    4np_7LkqL5M
    bpyferiOOzg
    uNsxCU0glHw
    4tts3c5p6mg
    lm1cuHMCMIg
    -nMyJn54ohk
    tFjih_o2UT8
    9Z4lQ4k_BQ8
    eBhmqoRj_t4
    hv-ZwM4QO_k
    vX-KwOVL-Y8
    GJKoCiI1ZzM
    SefVTIHjtFQ
    c14rLCh61p4
    2FPrJxTvgdQ
    v1POoRsvOXA
    o2S5IjQCQos
    nlddDLeUyXs
    6qKhjjfioC4
    L8w_g5q1RIM
    S1OqtvAyJRE
    ggsVF_MfkU4
    2Pa5VRr_sNQ
    A_-E9eplAT4
    MNigrqY9Cig
    ddtbPZPPUxY
    2MoJnOqVyUM
    80kxBJfajDI
    0AO7iMw7-zM
    apdrky-aLYY
    Hv64Ox76OQQ
    7MxnZPFxXQ0
    1IDk4192seE
    2wsS-NNKuK4
    vUeSqHkrJGI
    -l6DGtGZwnM
    JSBGDC0rKWU
    pdRBL82PgUE
    j2fFF1lFXfk
    UxBbCmRHBME
    zL3w6uIigxk
    BUyfoc611_Y
    G_UKmaY6Gm0
    4YBuxeKi7yc
    z3YzDNGR1Co
    ofL8fsd41mI
    a443i0wlYIM
    ooXFTOybSLc
    8mP7Z817oys
    iVZJCJewZGg
    DT61eBdj_TA
    PcjefMmvFqQ
    0Qw3Lhc_zt8
    QFmLReorTSY
    Z_oUddaYx_w
    F2xv4fba65U
    kXi3dKUgRy0
    Tfw2mq2wJls
    h-H1LddWxo8
    Fcnd4iEJ96U
    JBcI6HoEHQA
    DSc51Z26O9s
    aUtVNXmE-LA
    tPOOXp3S2UI
    wDFMW879WbM
    -L11Bxolo44
    tk60TI2i2ms
    aufMdURbitU
    ON_ZwV2-nOU
    2LUNq03VnEg
    kd9GEQkJLqQ
    JSHkYQoH4EE
    51TRDz-suUc
    n3apu-FHU_0
    KP_YpMBYttM
    pcKNxA8AF4E
    S_iVjOAMD4Y
    gK3ZmAtVIPk
    roddMS3X5Vo
    kbJurMqT4KI
    RcksWUqS1Cg
    oNE-igGbc5A
    bEeFvIWvMLY
    C1vBGz8c930
    XyhHBuS58U4
    MIEvw0Mo0_g
    YW3KEe0S4-s
    JPOy2cqQk58
    yoIMQUMG18U
    _PtoJ6kKl1E
    p50_CmZAFBw
    -ytTnjdXyfQ
    YpAN6e2sTys
    ykPF2Dky7tU
    LZ1VInL5Zh0
    ZQnxjn9Yk40
    DYNuIUXkys8
    -Q4MBdwizzg
    zCmuPDCmBnY
    LvkJm_1Isl4





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>caption</th>
      <th>definition</th>
      <th>dimension</th>
      <th>duration</th>
      <th>licensedContent</th>
      <th>projection</th>
      <th>regionRestriction</th>
      <th>categoryId</th>
      <th>channelId</th>
      <th>channelTitle</th>
      <th>...</th>
      <th>publishedAt</th>
      <th>tags</th>
      <th>thumbnails</th>
      <th>title</th>
      <th>commentCount</th>
      <th>dislikeCount</th>
      <th>favoriteCount</th>
      <th>likeCount</th>
      <th>viewCount</th>
      <th>videoId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT4M11S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'GB', u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2017-04-14T03:30:00.000Z</td>
      <td>[Hasan Minhaj, elections, Donald Trump, candid...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Why Wasn't Donald Trump's Bigotry a Deal-Break...</td>
      <td>1545</td>
      <td>822</td>
      <td>0</td>
      <td>22277</td>
      <td>1322057</td>
      <td>ZsHwtfcZoY4</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT4M30S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU', u'GB']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2017-04-13T03:30:01.000Z</td>
      <td>[Desi Lydic, Town hall, Congress, protests, Re...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>How to Spot a Paid Liberal Protester: The Dail...</td>
      <td>1221</td>
      <td>535</td>
      <td>0</td>
      <td>16537</td>
      <td>970295</td>
      <td>gbKhx_dlwCs</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT4M29S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'GB', u'CA', u'AU']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2017-04-12T03:30:00.000Z</td>
      <td>[Today’s Future Now, Ronny Chieng, technology,...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Today’s Future Now - Smart Technology: The Dai...</td>
      <td>337</td>
      <td>376</td>
      <td>0</td>
      <td>13600</td>
      <td>800535</td>
      <td>oANUXY3xXKM</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT5M37S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'GB', u'CA', u'AU']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2017-04-11T03:30:00.000Z</td>
      <td>[the daily show, Roy Wood Jr, Texas, guns, wea...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Texas Students Opt For C**cks Not Glocks: The ...</td>
      <td>2105</td>
      <td>514</td>
      <td>0</td>
      <td>19520</td>
      <td>908460</td>
      <td>UbJKhvmshGs</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT5M5S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU', u'GB']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2017-04-07T03:58:02.000Z</td>
      <td>[jordan klepper, donald trump, religion, fans,...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>The Divinity of Donald Trump: The Daily Show</td>
      <td>4203</td>
      <td>452</td>
      <td>0</td>
      <td>29893</td>
      <td>1482701</td>
      <td>1YFLBjR-swo</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT5M39S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'GB', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2017-04-04T03:37:06.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Black Eye on America - What Is Black Twitter?:...</td>
      <td>1717</td>
      <td>1125</td>
      <td>0</td>
      <td>14601</td>
      <td>935017</td>
      <td>CcSh2F8e__8</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT4M16S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'GB', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-12-20T17:17:22.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>The Gift of Reproductive Rights: The Daily Show</td>
      <td>2046</td>
      <td>648</td>
      <td>0</td>
      <td>9142</td>
      <td>543312</td>
      <td>MIhcVon9ruo</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT4M36S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'GB', u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-12-06T20:30:00.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Jordan Klepper Fingers the Pulse - President-E...</td>
      <td>1504</td>
      <td>325</td>
      <td>0</td>
      <td>12270</td>
      <td>865266</td>
      <td>iAQnXnQQCCI</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT4M29S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-11-09T02:32:12.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Make America Hate Again - Uncensored: The Dail...</td>
      <td>3845</td>
      <td>3483</td>
      <td>0</td>
      <td>16928</td>
      <td>1506103</td>
      <td>SCRY_tOPQ7Q</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT4M9S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'GB', u'CA', u'AU']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-11-09T16:03:50.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Jordan Klepper Fingers the Pulse - Clinton and...</td>
      <td>1357</td>
      <td>287</td>
      <td>0</td>
      <td>11164</td>
      <td>875089</td>
      <td>7lex_1MLrR8</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT5M37S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'GB', u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-10-07T20:30:01.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>"The O'Reilly Factor" Gets Racist in Chinatown...</td>
      <td>4357</td>
      <td>1377</td>
      <td>0</td>
      <td>50204</td>
      <td>2039249</td>
      <td>rX8jZTN0CdU</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT5M25S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU', u'GB']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-09-21T19:33:23.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Jordan Klepper Fingers the Pulse - Conspiracy ...</td>
      <td>5322</td>
      <td>698</td>
      <td>0</td>
      <td>31495</td>
      <td>2489287</td>
      <td>eFQhw3VVToQ</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT4M33S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'GB', u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-08-19T19:38:07.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Putting Donald Trump Supporters Through an Ide...</td>
      <td>9646</td>
      <td>2962</td>
      <td>0</td>
      <td>57961</td>
      <td>4003035</td>
      <td>Y4Zdx97A63s</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT2M14S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU', u'GB']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-07-23T02:35:25.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>When Was America Great?: The Daily Show</td>
      <td>3082</td>
      <td>771</td>
      <td>0</td>
      <td>24798</td>
      <td>1904339</td>
      <td>uVQvWwHM5kM</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT5M48S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU', u'GB']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-07-20T00:06:36.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Preparing for Anti-Press Hostility at the RNC ...</td>
      <td>504</td>
      <td>289</td>
      <td>0</td>
      <td>7605</td>
      <td>560548</td>
      <td>8aeEQw73uDg</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT9M2S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU', u'GB']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-07-01T19:30:01.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Jessica Williams Questions Sanders-to-Trump Su...</td>
      <td>3947</td>
      <td>2756</td>
      <td>0</td>
      <td>9668</td>
      <td>676958</td>
      <td>rwfM5LGMmxg</td>
    </tr>
    <tr>
      <th>0</th>
      <td>false</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT11M16S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'GB', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-05-19T18:42:35.000Z</td>
      <td>[comedy central, stand up comedy, comedians, c...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Prime Minister Justin Trudeau Welcomes Syrian ...</td>
      <td>4403</td>
      <td>2822</td>
      <td>0</td>
      <td>37715</td>
      <td>2340963</td>
      <td>9gT-vJg-EfM</td>
    </tr>
    <tr>
      <th>0</th>
      <td>false</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT5M37S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'GB', u'AU']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-05-13T19:28:18.000Z</td>
      <td>[Jordan Klepper, Donald Trump, elections, cand...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Donald Trump to Thank for Increase in Latino C...</td>
      <td>745</td>
      <td>534</td>
      <td>0</td>
      <td>10877</td>
      <td>1427838</td>
      <td>wBhlyBrtB90</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT6M19S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'CA', u'GB']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-05-05T19:30:00.000Z</td>
      <td>[Hasan Minhaj, soccer, sports, men/women, disc...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>American Soccer's Gender Wage Gap: The Daily Show</td>
      <td>7108</td>
      <td>14017</td>
      <td>0</td>
      <td>9452</td>
      <td>550079</td>
      <td>x9GqWoy6__s</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT8M11S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'GB', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-04-07T20:12:52.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>The Trans Panic Epidemic: The Daily Show</td>
      <td>2141</td>
      <td>1509</td>
      <td>0</td>
      <td>16955</td>
      <td>1039343</td>
      <td>PIvCh3EQv1Q</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT5M4S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'GB', u'CA', u'AU']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-02-12T20:48:20.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Donald Trump - The Greatest Show on Earth: The...</td>
      <td>566</td>
      <td>407</td>
      <td>0</td>
      <td>6267</td>
      <td>719456</td>
      <td>5NMK88czOug</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT6M53S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'GB', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-01-22T20:30:00.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Wrestling with History in Whitesboro, NY: The ...</td>
      <td>813</td>
      <td>708</td>
      <td>0</td>
      <td>7709</td>
      <td>539454</td>
      <td>eSwbG5V5S-8</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT10M38S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'GB', u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2015-12-11T20:30:01.000Z</td>
      <td>[comedy central, stand up comedy, comedians, c...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Jordan Klepper: Good Guy with a Gun: The Daily...</td>
      <td>1425</td>
      <td>1665</td>
      <td>0</td>
      <td>17120</td>
      <td>1417276</td>
      <td>MCI4bUk4vuM</td>
    </tr>
    <tr>
      <th>0</th>
      <td>false</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT1M29S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU', u'GB']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2015-11-10T23:40:40.000Z</td>
      <td>[comedy central, stand up comedy, comedians, c...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Serial Killer Tourism in Nebraska: The Daily Show</td>
      <td>29</td>
      <td>56</td>
      <td>0</td>
      <td>640</td>
      <td>125974</td>
      <td>CqR0HM7Z2gM</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT5M33S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'GB', u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2015-11-10T23:27:52.000Z</td>
      <td>[comedy central, stand up comedy, comedians, c...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>America's Voting Machines Are F**ked: The Dail...</td>
      <td>634</td>
      <td>199</td>
      <td>0</td>
      <td>10121</td>
      <td>1049688</td>
      <td>mQsNkt9yuKI</td>
    </tr>
    <tr>
      <th>0</th>
      <td>false</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT10M1S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU', u'GB']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2015-10-01T16:00:01.000Z</td>
      <td>[The Daily Show, Daily Show videos, comedy cen...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Are All Cops Racist?: The Daily Show</td>
      <td>2163</td>
      <td>1843</td>
      <td>0</td>
      <td>25097</td>
      <td>3336613</td>
      <td>3QIWolLM9i8</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT5M30S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'GB', u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2017-03-17T03:28:40.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>The Dutch Ditch Their Own Crazy-Haired Populis...</td>
      <td>6988</td>
      <td>3541</td>
      <td>0</td>
      <td>40235</td>
      <td>2615508</td>
      <td>VW1pdY3sNcA</td>
    </tr>
    <tr>
      <th>0</th>
      <td>false</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT4M23S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'GB', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2017-01-04T20:00:00.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Russia Reacts to President Obama's Sanctions: ...</td>
      <td>2307</td>
      <td>1722</td>
      <td>0</td>
      <td>21248</td>
      <td>1517469</td>
      <td>4np_7LkqL5M</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT7M41S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU', u'GB']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-12-07T04:33:38.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>President-Elect Trump Talks to Taiwan: The Dai...</td>
      <td>2912</td>
      <td>1589</td>
      <td>0</td>
      <td>17865</td>
      <td>1641814</td>
      <td>bpyferiOOzg</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT6M9S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'GB', u'CA', u'AU']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-06-28T19:30:00.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Brett Breakdown: The Daily Show</td>
      <td>5721</td>
      <td>4718</td>
      <td>0</td>
      <td>26330</td>
      <td>2163023</td>
      <td>uNsxCU0glHw</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT2M20S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2017-05-12T04:46:29.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - Trump's Dictator Tendenci...</td>
      <td>448</td>
      <td>168</td>
      <td>0</td>
      <td>14349</td>
      <td>823725</td>
      <td>kd9GEQkJLqQ</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT2M4S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2017-04-24T15:15:38.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - Running Out of Spanish: T...</td>
      <td>842</td>
      <td>137</td>
      <td>0</td>
      <td>18744</td>
      <td>658910</td>
      <td>JSHkYQoH4EE</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT1M26S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2017-03-28T03:57:40.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - The "Hidden Gun" of Healt...</td>
      <td>304</td>
      <td>132</td>
      <td>0</td>
      <td>13186</td>
      <td>599404</td>
      <td>51TRDz-suUc</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT1M19S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2017-03-17T17:23:17.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - The One Exciting Thing Ab...</td>
      <td>455</td>
      <td>121</td>
      <td>0</td>
      <td>12352</td>
      <td>626841</td>
      <td>n3apu-FHU_0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT1M37S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2017-03-17T05:05:59.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - Playa-Hatin' Republicans:...</td>
      <td>190</td>
      <td>74</td>
      <td>0</td>
      <td>9710</td>
      <td>440696</td>
      <td>KP_YpMBYttM</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT1M22S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2017-03-17T05:06:51.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - Health Care: It's Not Jus...</td>
      <td>312</td>
      <td>84</td>
      <td>0</td>
      <td>9472</td>
      <td>385473</td>
      <td>pcKNxA8AF4E</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT51S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2017-03-02T21:30:00.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>The Melania Trump Litmus Test: The Daily Show ...</td>
      <td>239</td>
      <td>159</td>
      <td>0</td>
      <td>9489</td>
      <td>608662</td>
      <td>S_iVjOAMD4Y</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT2M21S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2017-03-02T21:30:00.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Donald Trump Dresses Up His Act: The Daily Sho...</td>
      <td>777</td>
      <td>281</td>
      <td>0</td>
      <td>16127</td>
      <td>955217</td>
      <td>gK3ZmAtVIPk</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT1M21S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2017-01-25T05:18:49.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - Feminism in South Africa:...</td>
      <td>805</td>
      <td>206</td>
      <td>0</td>
      <td>14756</td>
      <td>612345</td>
      <td>roddMS3X5Vo</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT1M8S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2017-01-14T20:30:00.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - The Manila Folder Preside...</td>
      <td>401</td>
      <td>158</td>
      <td>0</td>
      <td>7586</td>
      <td>385595</td>
      <td>kbJurMqT4KI</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT1M41S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-12-15T20:06:50.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - Trump's Foreign Policy Kn...</td>
      <td>331</td>
      <td>120</td>
      <td>0</td>
      <td>8372</td>
      <td>391131</td>
      <td>RcksWUqS1Cg</td>
    </tr>
    <tr>
      <th>0</th>
      <td>false</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT54S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-12-05T23:20:19.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - Donald Trump vs. Intellig...</td>
      <td>681</td>
      <td>251</td>
      <td>0</td>
      <td>9111</td>
      <td>504980</td>
      <td>oNE-igGbc5A</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT1M43S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-10-28T21:44:51.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - Obamacare Confusion: The ...</td>
      <td>203</td>
      <td>37</td>
      <td>0</td>
      <td>5449</td>
      <td>274119</td>
      <td>bEeFvIWvMLY</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT1M17S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-10-28T21:45:14.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - Obamacare vs. the Afforda...</td>
      <td>241</td>
      <td>55</td>
      <td>0</td>
      <td>5191</td>
      <td>306404</td>
      <td>C1vBGz8c930</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT1M40S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-10-20T02:43:29.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - The Evolution of Pussygat...</td>
      <td>329</td>
      <td>88</td>
      <td>0</td>
      <td>9260</td>
      <td>424892</td>
      <td>XyhHBuS58U4</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT54S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-09-13T07:09:54.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - Basket of Deplorables: Th...</td>
      <td>234</td>
      <td>62</td>
      <td>0</td>
      <td>2871</td>
      <td>203819</td>
      <td>MIEvw0Mo0_g</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT1M4S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-08-19T16:02:28.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - Censoring Gratuitous Movi...</td>
      <td>138</td>
      <td>66</td>
      <td>0</td>
      <td>4648</td>
      <td>302146</td>
      <td>YW3KEe0S4-s</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT1M28S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-08-19T16:02:32.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - The Rio Games: The Daily ...</td>
      <td>132</td>
      <td>70</td>
      <td>0</td>
      <td>5869</td>
      <td>404772</td>
      <td>JPOy2cqQk58</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT2M4S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-07-12T15:30:19.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - Trevor's Dream Match-up: ...</td>
      <td>164</td>
      <td>24</td>
      <td>0</td>
      <td>3238</td>
      <td>217849</td>
      <td>yoIMQUMG18U</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT5M12S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-07-12T15:30:35.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - Going Deep with Lindsey G...</td>
      <td>564</td>
      <td>95</td>
      <td>0</td>
      <td>5661</td>
      <td>412194</td>
      <td>_PtoJ6kKl1E</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT1M29S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-07-12T15:30:49.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - Trinidadian Accent: The D...</td>
      <td>148</td>
      <td>32</td>
      <td>0</td>
      <td>4472</td>
      <td>265481</td>
      <td>p50_CmZAFBw</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT1M2S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-07-11T14:59:58.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - The NAACP or the NCAA?: T...</td>
      <td>31</td>
      <td>6</td>
      <td>0</td>
      <td>1081</td>
      <td>82137</td>
      <td>-ytTnjdXyfQ</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT53S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-07-12T15:31:44.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - Donald Trump: Worse Than ...</td>
      <td>86</td>
      <td>49</td>
      <td>0</td>
      <td>1863</td>
      <td>142642</td>
      <td>YpAN6e2sTys</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT1M2S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-07-12T15:32:00.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - Desi Lydic's Pregnancy: T...</td>
      <td>29</td>
      <td>11</td>
      <td>0</td>
      <td>1076</td>
      <td>102805</td>
      <td>ykPF2Dky7tU</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT1M5S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-07-12T15:32:15.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - The Ben and Ken Carson Wo...</td>
      <td>64</td>
      <td>20</td>
      <td>0</td>
      <td>1881</td>
      <td>129979</td>
      <td>LZ1VInL5Zh0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT1M2S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-07-12T15:32:33.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - Donald Trump or Ted Cruz:...</td>
      <td>47</td>
      <td>33</td>
      <td>0</td>
      <td>1847</td>
      <td>138729</td>
      <td>ZQnxjn9Yk40</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT55S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-07-11T16:47:54.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - The Global Edition: The D...</td>
      <td>91</td>
      <td>86</td>
      <td>0</td>
      <td>1857</td>
      <td>167769</td>
      <td>DYNuIUXkys8</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT2M46S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'CA', u'AU']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-07-12T15:33:02.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - B.O.B's Flat Earth Twitte...</td>
      <td>666</td>
      <td>147</td>
      <td>0</td>
      <td>8257</td>
      <td>522043</td>
      <td>-Q4MBdwizzg</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT1M19S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-07-12T15:33:19.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - The Truth About Ronny Chi...</td>
      <td>101</td>
      <td>24</td>
      <td>0</td>
      <td>2979</td>
      <td>224649</td>
      <td>zCmuPDCmBnY</td>
    </tr>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>hd</td>
      <td>2d</td>
      <td>PT1M12S</td>
      <td>True</td>
      <td>rectangular</td>
      <td>{u'blocked': [u'AU', u'CA']}</td>
      <td>23</td>
      <td>UCwWhs_6x42TyRM4Wstoq8HA</td>
      <td>The Daily Show with Trevor Noah</td>
      <td>...</td>
      <td>2016-07-12T15:33:32.000Z</td>
      <td>[the daily show, trevor noah, daily show with ...</td>
      <td>{u'default': {u'url': u'https://i.ytimg.com/vi...</td>
      <td>Between the Scenes - The Trouble with Pandas: ...</td>
      <td>79</td>
      <td>31</td>
      <td>0</td>
      <td>3072</td>
      <td>205239</td>
      <td>LvkJm_1Isl4</td>
    </tr>
  </tbody>
</table>
<p>127 rows × 24 columns</p>
</div>




{% highlight python %}
for title in vid_df.title:
    print(title)
    print
{% endhighlight %}

    Why Wasn't Donald Trump's Bigotry a Deal-Breaker?: The Daily Show
    
    How to Spot a Paid Liberal Protester: The Daily Show
    
    Today’s Future Now - Smart Technology: The Daily Show
    
    Texas Students Opt For C**cks Not Glocks: The Daily Show
    
    The Divinity of Donald Trump: The Daily Show
    
    Black Eye on America - What Is Black Twitter?: The Daily Show
    
    The Gift of Reproductive Rights: The Daily Show
    
    Jordan Klepper Fingers the Pulse - President-Elect Trump's Victory/Thank You Tour: The Daily Show
    
    Make America Hate Again - Uncensored: The Daily Show
    
    Jordan Klepper Fingers the Pulse - Clinton and Trump Supporters Find Common Ground: The Daily Show
    
    "The O'Reilly Factor" Gets Racist in Chinatown: The Daily Show
    
    Jordan Klepper Fingers the Pulse - Conspiracy Theories Thrive at a Trump Rally: The Daily Show
    
    Putting Donald Trump Supporters Through an Ideology Test: The Daily Show
    
    When Was America Great?: The Daily Show
    
    Preparing for Anti-Press Hostility at the RNC - Uncensored: The Daily Show
    
    Jessica Williams Questions Sanders-to-Trump Supporters & Says Goodbye: The Daily Show
    
    Prime Minister Justin Trudeau Welcomes Syrian Refugees to Canada: The Daily Show
    
    Donald Trump to Thank for Increase in Latino Citizenship: The Daily Show
    
    American Soccer's Gender Wage Gap: The Daily Show
    
    The Trans Panic Epidemic: The Daily Show
    
    Donald Trump - The Greatest Show on Earth: The Daily Show
    
    Wrestling with History in Whitesboro, NY: The Daily Show
    
    Jordan Klepper: Good Guy with a Gun: The Daily Show
    
    Serial Killer Tourism in Nebraska: The Daily Show
    
    America's Voting Machines Are F**ked: The Daily Show
    
    Are All Cops Racist?: The Daily Show
    
    The Dutch Ditch Their Own Crazy-Haired Populist: The Daily Show
    
    Russia Reacts to President Obama's Sanctions: The Daily Show
    
    President-Elect Trump Talks to Taiwan: The Daily Show
    
    Brett Breakdown: The Daily Show
    
    President Obama's Visit to Vietnam & Hillary Clinton's Likability Problem: The Daily Show
    
    South African President Jacob Zuma & The Panama Papers: The Daily Show
    
    The Americanization of Cuba: The Daily Show
    
    Back in Black - Osama bin Laden's Last Wishes: The Daily Show
    
    Uganda - Even Worse at Elections Than America: The Daily Show
    
    The Fight Against ISIS: The Daily Show
    
    Tragedy in Paris - The Three Stages of Political Grief: The Daily Show
    
    The Myanmar Daily Show: The Daily Show
    
    China Ditches Its One-Child Policy: The Daily Show
    
    Benghazi - The Never-Ending Scandal: The Daily Show
    
    Canada's Hot New Prime Minister: The Daily Show
    
    Donald Trump - America's African President: The Daily Show
    
    Exclusive - Jason Isbell - "If We Were Vampires": The Daily Show
    
    Alabama Week - Prejudice & Pigskin: The Daily Show
    
    LEAKED: Fox News' 8 p.m. Anchor Audition Tape: The Daily Show
    
    Please, Just Like… Don't: How to Protest: The Daily Show
    
    Exclusive - Zara Larsson - "So Good": The Daily Show
    
    Brain Doctors MD: The Daily Show
    
    The Daily Show's Gift Guide: Uncensored
    
    Exclusive - Hasan Minhaj Says Goodbye to 2016: The Daily Show
    
    Donald Trump's Christmas (NOT HOLIDAY) Yule Log: The Daily Show
    
    R.I.P. Facts: The Daily Show
    
    "U Name It" Challenge for a Stressful Thanksgiving: The Daily Show
    
    Exclusive - Election Day 2016: Feel the Rush: The Daily Show
    
    Exclusive - Trump Gym: The Daily Show
    
    Eric the Eel - Uncensored: The Daily Show
    
    Ronny Chieng's Philly Food Tour - Exclusive: The Daily Show
    
    Behind the Scenes - There Will Be Mud: The Daily Show
    
    Exclusive - Preparing for the Conventions: The Daily Show
    
    Imagining Donald Trump's Cabinet: The Daily Show
    
    Thank You, Jessica Williams: The Daily Show
    
    Jordan Klepper's Happy Endings - Solving Illinois's Budget Gridlock - Uncensored: The Daily Show
    
    Exclusive - Wishing Donald Trump a Happy Birthday - Uncensored: The Daily Show
    
    "They Love Me" Music Video - Black Trump (ft. Jordan Klepper): The Daily Show
    
    Trevor's Thanksgiving Thankstacular Round-Up: The Daily Show
    
    Keys to Success with DJ Khaled and Hasan Minhaj: The Daily Show
    
    New Theme Song - "Dog On Fire": The Daily Show
    
    The Affirmative Actions Porn Series: The Daily Show
    
    Trump Supporters Speak Out: The Daily Show
    
    Podium Pandemonium at the New Hampshire Primary: The Daily Show
    
    Behind the Scenes at the New Hampshire Primary: The Daily Show
    
    Exclusive - The Daily Show vs. Justin Trudeau: Sorry Not Sorry
    
    Donald Trump Speaks to the Washington Post: A Dramatic Reenactment: The Daily Show
    
    Exclusive - The Daily Show vs. Justin Trudeau: Get Ready to Look Like S**t
    
    Exclusive - The Daily Show vs. Justin Trudeau: Read My Spec Script
    
    Exclusive - In the Green Room with Joe Morton: The Daily Show
    
    Exclusive - Hasan and Ellie Positivity-Off: The Daily Show
    
    Exclusive - Breaking Down the Ban the Box Campaign: The Daily Show
    
    The Battle for Less Bias: The Daily Show
    
    John Kasich - Uniting America in "Two Paths" - Extended Interview: The Daily Show
    
    Keegan-Michael Key - A Final Address from Obama's Anger Translator: The Daily Show
    
    Cecile Richards - The High Cost of Defunding Planned Parenthood: The Daily Show
    
    Tomi Lahren - Giving a Voice to Conservative America on "Tomi": The Daily Show
    
    Wesley Lowery - Delving Deeper Into Police Violence with “They Can’t Kill Us All": The Daily Show
    
    George Packer Extended Interview: Donald Trump’s Path to Victory: The Daily Show
    
    Bill Clinton - Hillary Clinton and the Changing Political Landscape: The Daily Show
    
    Hannah Hart Extended Interview - Coming of Age in "Dirty 30": The Daily Show
    
    John Lewis Extended Interview - Getting Into Trouble to Fight Injustice: The Daily Show
    
    The Extended GOP Debate - Singles Night with Rand Paul: The Daily Show
    
    Dr. Ben Carson vs. Dr. Ken Carson: The Doctors Debate: The Daily Show
    
    Lindsey Graham - The Senator Picks His Poison: Ted Cruz vs. Donald Trump: The Daily Show
    
    Lilly Singh - Taking Fans on "A Trip to Unicorn Island": The Daily Show
    
    Jon Stewart Returns to Shame Congress: The Daily Show
    
    Kevin Hart - Extended Interview: The Daily Show
    
    Between the Scenes - Philando Castile & the Black Experience in America: The Daily Show
    
    Between the Scenes - The White House's Messy Lie: The Daily Show
    
    Between the Scenes - Donald Trump: America's Penis-Shaped Asteroid: The Daily Show
    
    Between the Scenes - Trump's Dictator Tendencies: The Daily Show
    
    Between the Scenes - Running Out of Spanish: The Daily Show
    
    Between the Scenes - The "Hidden Gun" of Health Care: The Daily Show
    
    Between the Scenes - The One Exciting Thing About Donald Trump: The Daily Show - Uncensored
    
    Between the Scenes - Playa-Hatin' Republicans: The Daily Show
    
    Between the Scenes - Health Care: It's Not Just for Healthy Young People: The Daily Show
    
    The Melania Trump Litmus Test: The Daily Show - Between the Scenes
    
    Donald Trump Dresses Up His Act: The Daily Show - Between the Scenes
    
    Between the Scenes - Feminism in South Africa: The Daily Show
    
    Between the Scenes - The Manila Folder Presidency: The Daily Show
    
    Between the Scenes - Trump's Foreign Policy Knowhow: The Daily Show
    
    Between the Scenes - Donald Trump vs. Intelligence: The Daily Show
    
    Between the Scenes - Obamacare Confusion: The Daily Show
    
    Between the Scenes - Obamacare vs. the Affordable Care Act: The Daily Show
    
    Between the Scenes - The Evolution of Pussygate: The Daily Show
    
    Between the Scenes - Basket of Deplorables: The Daily Show
    
    Between the Scenes - Censoring Gratuitous Movies: The Daily Show
    
    Between the Scenes - The Rio Games: The Daily Show
    
    Between the Scenes - Trevor's Dream Match-up: The Daily Show
    
    Between the Scenes - Going Deep with Lindsey Graham: The Daily Show
    
    Between the Scenes - Trinidadian Accent: The Daily Show
    
    Between the Scenes - The NAACP or the NCAA?: The Daily Show
    
    Between the Scenes - Donald Trump: Worse Than an African Dictator: The Daily Show
    
    Between the Scenes - Desi Lydic's Pregnancy: The Daily Show
    
    Between the Scenes - The Ben and Ken Carson Word-Off: The Daily Show
    
    Between the Scenes - Donald Trump or Ted Cruz: The Daily Show
    
    Between the Scenes - The Global Edition: The Daily Show
    
    Between the Scenes - B.O.B's Flat Earth Twitter Rant: The Daily Show
    
    Between the Scenes - The Truth About Ronny Chieng’s Accent: The Daily Show
    
    Between the Scenes - The Trouble with Pandas: The Daily Show
    



{% highlight python %}
vid_df=pd.read_csv("trevor_noah_daily_show_videos.csv",encoding='utf-8')
{% endhighlight %}


{% highlight python %}
import ast
import json
for videoId in vid_df.videoId:
    tags = vid_df[vid_df.videoId == videoId].tags.reset_index(drop=True)[0]
    print(tags)
    tag_val = {}
    for tag in ast.literal_eval(tags):
        tag_val[tag]=1
    print(tag_val)
    print

{% endhighlight %}

    [u'Hasan Minhaj', u'elections', u'Donald Trump', u'candidates', u'campaigns', u'voting', u'religion', u'discrimination', u'racism', u'immigration', u'rage', u'rants', u'flying', u'Barack Obama', u'parents', u'family', u'daily show', u'the daily show', u'daily show with trevor noah', u'comedy central', u'late night talk show hosts', u'comedy central comedians', u'comedian', u'comedy', u'funny', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'family': 1, u'rants': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'Donald Trump': 1, u'funny': 1, u'the daily show': 1, u'discrimination': 1, u'flying': 1, u'racism': 1, u'religion': 1, u'parents': 1, u'stand up videos': 1, u'candidates': 1, u'comedy': 1, u'hilarious videos': 1, u'daily show': 1, u'immigration': 1, u'funny video': 1, u'comedy central comedians': 1, u'campaigns': 1, u'Barack Obama': 1, u'comedy central': 1, u'funny jokes': 1, u'hilarious clips': 1, u'rage': 1, u'elections': 1, u'comedy videos': 1, u'comedian': 1, u'Hasan Minhaj': 1, u'voting': 1, u'daily show with trevor noah': 1}
    
    [u'Desi Lydic', u'Town hall', u'Congress', u'protests', u'Republicans', u'man on the street', u'Alabama', u'Adolf Hitler', u'daily show', u'the daily show', u'daily show with trevor noah', u'comedy central', u'late night talk show hosts', u'comedy central comedians', u'comedian', u'comedy', u'funny', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'protests': 1, u'comedian': 1, u'funny clips': 1, u'funny': 1, u'Congress': 1, u'the daily show': 1, u'Republicans': 1, u'Town hall': 1, u'daily show': 1, u'stand up videos': 1, u'comedy': 1, u'hilarious videos': 1, u'Desi Lydic': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'Adolf Hitler': 1, u'Alabama': 1, u'man on the street': 1, u'funny jokes': 1, u'hilarious clips': 1, u'comedy videos': 1, u'late night talk show hosts': 1, u'daily show with trevor noah': 1}
    
    [u'Today\u2019s Future Now', u'Ronny Chieng', u'technology', u'millennials', u'internet', u'rage', u'rants', u'kids', u'porn', u'security', u'spying', u'daily show', u'the daily show', u'daily show with trevor noah', u'comedy central', u'late night talk show hosts', u'comedy central comedians', u'comedian', u'comedy', u'funny', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'porn': 1, u'spying': 1, u'rants': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'technology': 1, u'Today\u2019s Future Now': 1, u'funny': 1, u'the daily show': 1, u'stand up videos': 1, u'internet': 1, u'comedy': 1, u'hilarious videos': 1, u'daily show': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'funny jokes': 1, u'Ronny Chieng': 1, u'kids': 1, u'hilarious clips': 1, u'rage': 1, u'millennials': 1, u'comedy videos': 1, u'comedian': 1, u'security': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'Roy Wood Jr', u'Texas', u'guns', u'weapons', u'college', u'school', u'Olive Garden', u'protests', u'sex toys', u'censorship', u'daily show with trevor noah', u'comedy central', u'late night talk show hosts', u'comedy central comedians', u'comedian', u'comedy', u'funny', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'protests': 1, u'college': 1, u'funny clips': 1, u'Texas': 1, u'funny': 1, u'the daily show': 1, u'weapons': 1, u'stand up videos': 1, u'comedian': 1, u'comedy': 1, u'guns': 1, u'hilarious videos': 1, u'Olive Garden': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'censorship': 1, u'Roy Wood Jr': 1, u'school': 1, u'funny jokes': 1, u'hilarious clips': 1, u'comedy videos': 1, u'late night talk show hosts': 1, u'sex toys': 1, u'daily show with trevor noah': 1}
    
    [u'jordan klepper', u'donald trump', u'religion', u'fans', u'Christianity', u'RNC', u'Republicans', u'Republican National Conventions', u'marriage', u'divorce', u'money', u'conventions', u'Bible', u'Hillary Clinton', u'Impressions', u'adultery', u'Bill Clinton', u'songs', u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah', u'comedy central politics', u'late night talk show hosts', u'comedy central', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'money': 1, u'Christianity': 1, u'jordan klepper': 1, u'late night talk show hosts': 1, u'adultery': 1, u'funny clips': 1, u'funny': 1, u'the daily show': 1, u'divorce': 1, u'Republican National Conventions': 1, u'Republicans': 1, u'religion': 1, u'fans': 1, u'stand up videos': 1, u'comedian': 1, u'comedy': 1, u'hilarious videos': 1, u'Bible': 1, u'Bill Clinton': 1, u'funny video': 1, u'donald trump': 1, u'conventions': 1, u'comedy central politics': 1, u'comedy central': 1, u'funny jokes': 1, u'hilarious clips': 1, u'RNC': 1, u'new trevor noah': 1, u'Hillary Clinton': 1, u'comedy videos': 1, u'marriage': 1, u'Impressions': 1, u'songs': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Black Eye on America - What is Black Twitter?', u'Roy Wood Jr.', u'African American/black', u'stereotypes', u'Twitter', u'Beyonce', u'racism', u'discrimination', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'Beyonce': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'Roy Wood Jr.': 1, u'funny': 1, u'new trevor noah show': 1, u'the daily show': 1, u'discrimination': 1, u'racism': 1, u'comedians': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'African American/black': 1, u'stereotypes': 1, u'comedy central politics': 1, u'comedy': 1, u'hilarious clips': 1, u'Twitter': 1, u'comedy videos': 1, u'Black Eye on America - What is Black Twitter?': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Donald Trump', u'Trump administration', u'men/women', u'birth control', u'Mike Pence', u'Desi Lydic', u'doctors', u'advertising', u'Ronny Chieng', u'Jordan Klepper', u'Eliza Cossio', u'Christmas', u'gifts & presents', u'The Best F#@king News Team Ever', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny video', u'comedy videos']
    {u'trevor noah': 1, u'Jordan Klepper': 1, u'late night talk show hosts': 1, u'birth control': 1, u'Eliza Cossio': 1, u'new trevor noah show': 1, u'Christmas': 1, u'gifts & presents': 1, u'men/women': 1, u'the daily show': 1, u'comedians': 1, u'Trump administration': 1, u'stand up comedy': 1, u'Desi Lydic': 1, u'Mike Pence': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'doctors': 1, u'Ronny Chieng': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'comedy': 1, u'The Best F#@king News Team Ever': 1, u'comedy videos': 1, u'advertising': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Donald Trump', u'Trump administration', u'Twitter', u'behaving badly', u'lying', u'Mike Pence', u'George Stephanopoulos', u'awards', u'Jake Tapper', u'Bill Clinton', u'scandals', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips']
    {u'trevor noah': 1, u'George Stephanopoulos': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'scandals': 1, u'behaving badly': 1, u'the daily show': 1, u'comedians': 1, u'Trump administration': 1, u'stand up comedy': 1, u'stand up videos': 1, u'funny': 1, u'Bill Clinton': 1, u'Mike Pence': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'awards': 1, u'Jake Tapper': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'comedy': 1, u'Twitter': 1, u'comedy videos': 1, u'comedian': 1, u'lying': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'exclusives', u'Ronny Chieng', u'elections', u'candidates', u'campaigns', u'Hillary Clinton', u'Donald Trump', u'uncensored', u'Bill Cosby', u'voting', u'Adolf Hitler', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes']
    {u'trevor noah': 1, u'uncensored': 1, u'Hillary Clinton': 1, u'late night talk show hosts': 1, u'Bill Cosby': 1, u'new trevor noah show': 1, u'Donald Trump': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'candidates': 1, u'stand up videos': 1, u'stand up comedy': 1, u'exclusives': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'Adolf Hitler': 1, u'Ronny Chieng': 1, u'comedy central politics': 1, u'campaigns': 1, u'comedy': 1, u'elections': 1, u'comedy videos': 1, u'comedian': 1, u'voting': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Jordan Klepper', u'Jordan Klepper Fingers the Pulse', u'Hillary Clinton', u'Donald Trump', u'fans', u'man on the street', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'Jordan Klepper': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Jordan Klepper Fingers the Pulse': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'fans': 1, u'stand up videos': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'man on the street': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'comedy': 1, u'hilarious clips': 1, u'Hillary Clinton': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Ronny Chieng', u'China', u'debates', u'stereotypes', u"The O'Reilly Factor", u'Asian American', u'discrimination', u'racism', u'martial arts', u'douchebags', u'behaving badly', u'insults', u'elections', u'candidates', u'campaigns', u'translations', u'man on the streetlate night talk show hosts', u'comedy central', u'comedy', u'funny', u'comedian', u'funny video', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'comedian': 1, u'douchebags': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'behaving badly': 1, u'Asian American': 1, u'the daily show': 1, u'discrimination': 1, u'racism': 1, u'candidates': 1, u'China': 1, u'comedy': 1, u'martial arts': 1, u'hilarious videos': 1, u'translations': 1, u'funny video': 1, u'insults': 1, u'campaigns': 1, u'the daily show episodes': 1, u'Ronny Chieng': 1, u'stereotypes': 1, u'comedy central politics': 1, u'comedy central': 1, u'funny jokes': 1, u'hilarious clips': 1, u'debates': 1, u'man on the streetlate night talk show hosts': 1, u'elections': 1, u"The O'Reilly Factor": 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Jordan Klepper', u'Hillary Clinton', u'rallies', u'Donald Trump', u'AIDS', u'Bill Clinton', u'Magic Johnson', u'conspiracies', u'Barack Obama', u'night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'Magic Johnson': 1, u'Jordan Klepper': 1, u'comedian': 1, u'AIDS': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'stand up videos': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'conspiracies': 1, u'Bill Clinton': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'Barack Obama': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'comedy': 1, u'hilarious clips': 1, u'night talk show hosts': 1, u'Hillary Clinton': 1, u'comedy videos': 1, u'rallies': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Jordan Klepper', u'Donald Trump', u'elections', u'candidates', u'Islam', u'religion', u'immigration', u'discrimination', u'sexism', u'Monica Lewinsky', u'LGBT', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'Jordan Klepper': 1, u'late night talk show hosts': 1, u'sexism': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'discrimination': 1, u'comedians': 1, u'religion': 1, u'candidates': 1, u'stand up videos': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'immigration': 1, u'funny video': 1, u'comedy central comedians': 1, u'comedy central': 1, u'LGBT': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'Monica Lewinsky': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'comedy': 1, u'hilarious clips': 1, u'elections': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1, u'Islam': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Roy Wood Jr.', u'man on the street', u'Republican National Convention', u'slavery', u'Jordan Klepper', u'Ronny Chieng', u'history', u'Trump campaign', u'Make America Great Again', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos']
    {u'Trump campaign': 1, u'trevor noah': 1, u'Jordan Klepper': 1, u'late night talk show hosts': 1, u'Roy Wood Jr.': 1, u'funny': 1, u'new trevor noah show': 1, u'the daily show': 1, u'comedians': 1, u'stand up comedy': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'Ronny Chieng': 1, u'Republican National Convention': 1, u'man on the street': 1, u'comedy central politics': 1, u'comedy': 1, u'Make America Great Again': 1, u'slavery': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1, u'history': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Hasan Minhaj', u'Roy Wood Jr.', u'Ronny Chieng', u'elections', u'Republican National Convention', u'RNC', u'Politico', u'Donald Trump', u'media', u'safety', u'uncensored', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips']
    {u'trevor noah': 1, u'uncensored': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'Roy Wood Jr.': 1, u'funny': 1, u'new trevor noah show': 1, u'the daily show': 1, u'media': 1, u'comedians': 1, u'safety': 1, u'stand up videos': 1, u'stand up comedy': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'Ronny Chieng': 1, u'Republican National Convention': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'comedy': 1, u'Politico': 1, u'RNC': 1, u'elections': 1, u'comedy videos': 1, u'comedian': 1, u'Hasan Minhaj': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Jessica Williams', u'Donald Trump', u'elections', u'candidates', u'campaigns', u'Hillary Clinton', u'Bernie Sanders', u'The Best F#@king News Team Ever', u'fans late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips']
    {u'trevor noah': 1, u'Hillary Clinton': 1, u'comedian': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'Jessica Williams': 1, u'the daily show': 1, u'comedians': 1, u'candidates': 1, u'stand up videos': 1, u'Bernie Sanders': 1, u'stand up comedy': 1, u'comedy central': 1, u'comedy central comedians': 1, u'funny video': 1, u'Donald Trump': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'campaigns': 1, u'comedy': 1, u'The Best F#@king News Team Ever': 1, u'fans late night talk show hosts': 1, u'elections': 1, u'comedy videos': 1, u'daily show with trevor noah': 1}
    
    [u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips', u'exclusives', u'Canada', u'Hasan Minhaj', u'terrorism', u'Islam', u'religion', u'refugees', u'Syria', u'ISIS', u'mongering', u'Justin Trudeau', u'stereotypes', u'Game of Thrones', u'borders', u'America', u'violence', u'murder', u'death', u'family', u'literary references', u'safety', u'security', u'statistics', u'racism', u'partying', u'Stanley Cup', u'crying', u'patriotism', u'on location']
    {u'Canada': 1, u'partying': 1, u'security': 1, u'family': 1, u'crying': 1, u'Justin Trudeau': 1, u'on location': 1, u'comedian': 1, u'patriotism': 1, u'funny clips': 1, u'funny': 1, u'death': 1, u'ISIS': 1, u'racism': 1, u'comedians': 1, u'Game of Thrones': 1, u'religion': 1, u'safety': 1, u'America': 1, u'stand up videos': 1, u'terrorism': 1, u'stand up comedy': 1, u'comedy': 1, u'hilarious videos': 1, u'murder': 1, u'literary references': 1, u'comedy central comedians': 1, u'funny video': 1, u'Stanley Cup': 1, u'comedy central': 1, u'statistics': 1, u'mongering': 1, u'Syria': 1, u'stereotypes': 1, u'borders': 1, u'exclusives': 1, u'funny jokes': 1, u'hilarious clips': 1, u'violence': 1, u'comedy videos': 1, u'Hasan Minhaj': 1, u'refugees': 1, u'Islam': 1}
    
    [u'Jordan Klepper', u'Donald Trump', u'elections', u'candidates', u'campaigns', u'immigration', u'Mexico', u'insults', u'racism', u'discrimination', u'citizenship', u'Latino/Hispanic', u'statistics', u'crime', u'voting', u'America', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'citizenship': 1, u'Jordan Klepper': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'Donald Trump': 1, u'Latino/Hispanic': 1, u'funny': 1, u'statistics': 1, u'discrimination': 1, u'comedy central comedians': 1, u'racism': 1, u'comedians': 1, u'crime': 1, u'candidates': 1, u'America': 1, u'stand up videos': 1, u'comedy': 1, u'hilarious videos': 1, u'immigration': 1, u'funny video': 1, u'insults': 1, u'campaigns': 1, u'funny jokes': 1, u'comedy central': 1, u'stand up comedy': 1, u'hilarious clips': 1, u'Mexico': 1, u'elections': 1, u'comedy videos': 1, u'comedian': 1, u'voting': 1}
    
    [u'Hasan Minhaj', u'soccer', u'sports', u'men/women', u'discrimination', u'sexism', u'lawsuits', u'World Cup', u'Olympics', u'Fox News', u'tennis', u'advertising', u'feminism', u'on location', u'Becky Sauerbrunn', u'Ali Krieger', u'Hope Solo', u'Gavin McInnes', u'Billie Jean King']
    {u'World Cup': 1, u'men/women': 1, u'Gavin McInnes': 1, u'Hope Solo': 1, u'discrimination': 1, u'lawsuits': 1, u'tennis': 1, u'on location': 1, u'sports': 1, u'Olympics': 1, u'advertising': 1, u'Billie Jean King': 1, u'sexism': 1, u'Hasan Minhaj': 1, u'soccer': 1, u'Fox News': 1, u'Ali Krieger': 1, u'feminism': 1, u'Becky Sauerbrunn': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Jessica Williams', u'LGBT', u'discrimination', u'Iowa', u'African American', u'crime', u'laws', u'religion', u'safety', u'sexual assault', u'statistics', u'men/women', u'insults', u'Bible', u'church and state', u'movies', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'new trevor noah show': 1, u'funny': 1, u'men/women': 1, u'statistics': 1, u'Jessica Williams': 1, u'the daily show': 1, u'church and state': 1, u'comedians': 1, u'crime': 1, u'religion': 1, u'safety': 1, u'stand up videos': 1, u'stand up comedy': 1, u'Bible': 1, u'comedy central comedians': 1, u'funny video': 1, u'insults': 1, u'comedy central': 1, u'LGBT': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'sexual assault': 1, u'Iowa': 1, u'African American': 1, u'comedy central politics': 1, u'comedy': 1, u'movies': 1, u'comedy videos': 1, u'discrimination': 1, u'comedian': 1, u'daily show with trevor noah': 1, u'laws': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Jordan Klepper', u'Donald Trump', u'elections', u'candidates', u'campaigns', u'circuses', u'safety', u'polls', u'rallies', u'fans', u'China', u'Mexico', u'ISIS', u'terrorism', u'Dwight D. Eisenhower', u'discrimination', u'night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos']
    {u'trevor noah': 1, u'night talk show hosts': 1, u'Jordan Klepper': 1, u'comedian': 1, u'new trevor noah show': 1, u'funny': 1, u'ISIS': 1, u'the daily show': 1, u'discrimination': 1, u'comedians': 1, u'fans': 1, u'safety': 1, u'candidates': 1, u'stand up videos': 1, u'China': 1, u'terrorism': 1, u'stand up comedy': 1, u'comedy central': 1, u'comedy central comedians': 1, u'Dwight D. Eisenhower': 1, u'Donald Trump': 1, u'the daily show episodes': 1, u'funny video': 1, u'comedy central politics': 1, u'campaigns': 1, u'comedy': 1, u'Mexico': 1, u'polls': 1, u'elections': 1, u'comedy videos': 1, u'rallies': 1, u'circuses': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Jessica Williams', u'Native American/American Indian', u'NFL', u'New York', u'white people', u'history', u'wrestling', u'fights', u'Revolutionary War', u'racism', u'African American', u'discrimination', u'controversies', u'late night talk show hosts', u'comedy central', u'comedy central comedians', u'comedy', u'funny', u'funny video', u'comedy videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'fights': 1, u'wrestling': 1, u'late night talk show hosts': 1, u'controversies': 1, u'Revolutionary War': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'Jessica Williams': 1, u'the daily show': 1, u'discrimination': 1, u'racism': 1, u'Native American/American Indian': 1, u'comedy': 1, u'hilarious videos': 1, u'NFL': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'African American': 1, u'comedy central politics': 1, u'funny jokes': 1, u'hilarious clips': 1, u'white people': 1, u'comedy videos': 1, u'New York': 1, u'daily show with trevor noah': 1, u'history': 1}
    
    [u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'funny': 1, u'stand up comedy': 1, u'hilarious clips': 1, u'comedy central comedians': 1, u'comedians': 1, u'comedy central': 1, u'comedy videos': 1, u'funny jokes': 1, u'comedian': 1, u'funny video': 1, u'stand up videos': 1, u'funny clips': 1, u'comedy': 1, u'hilarious videos': 1}
    
    [u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'funny': 1, u'stand up comedy': 1, u'hilarious clips': 1, u'comedy central comedians': 1, u'comedians': 1, u'comedy central': 1, u'comedy videos': 1, u'funny jokes': 1, u'comedian': 1, u'funny video': 1, u'stand up videos': 1, u'funny clips': 1, u'comedy': 1, u'hilarious videos': 1}
    
    [u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'funny': 1, u'stand up comedy': 1, u'hilarious clips': 1, u'comedy central comedians': 1, u'comedians': 1, u'comedy central': 1, u'comedy videos': 1, u'funny jokes': 1, u'comedian': 1, u'funny video': 1, u'stand up videos': 1, u'funny clips': 1, u'comedy': 1, u'hilarious videos': 1}
    
    [u'The Daily Show', u'Daily Show videos', u'comedy central politics', u'the daily show episodes', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips', u'Trevor Noah']
    {u'comedy central politics': 1, u'funny': 1, u'comedy': 1, u'hilarious clips': 1, u'The Daily Show': 1, u'comedy videos': 1, u'Daily Show videos': 1, u'comedy central comedians': 1, u'comedians': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedian': 1, u'funny video': 1, u'stand up videos': 1, u'funny clips': 1, u'stand up comedy': 1, u'Trevor Noah': 1, u'hilarious videos': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'The Dutch Ditch Their Own Crazy-Haired Populist', u'Netherlands', u'candidates', u'elections', u'extremism', u'Parliament', u'Islamic', u'racism', u"lookin' good", u'super villains', u'Donald Trump', u'comedy central', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'funny jokes', u'funny clips', u'hilarious videos']
    {u'trevor noah': 1, u'extremism': 1, u'comedian': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'Parliament': 1, u'the daily show': 1, u'racism': 1, u'comedians': 1, u'Islamic': 1, u'candidates': 1, u'comedy': 1, u'hilarious videos': 1, u'Netherlands': 1, u'super villains': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u"lookin' good": 1, u'The Dutch Ditch Their Own Crazy-Haired Populist': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'funny jokes': 1, u'elections': 1, u'comedy videos': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Obama administration', u'Barack Obama', u'Russia', u'security', u'Vladimir Putin', u'Christmas', u'holidays', u'elections', u'candidates', u'Donald Trump', u'Twitter', u'emails', u'Internet', u'Don King', u'technology', u'kids', u'spying', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos']
    {u'trevor noah': 1, u'spying': 1, u'late night talk show hosts': 1, u'Vladimir Putin': 1, u'Internet': 1, u'new trevor noah show': 1, u'technology': 1, u'Christmas': 1, u'funny': 1, u'the daily show': 1, u'Obama administration': 1, u'comedians': 1, u'Russia': 1, u'candidates': 1, u'stand up videos': 1, u'stand up comedy': 1, u'comedy': 1, u'comedy central comedians': 1, u'funny video': 1, u'holidays': 1, u'Donald Trump': 1, u'the daily show episodes': 1, u'Barack Obama': 1, u'Don King': 1, u'emails': 1, u'comedy central politics': 1, u'comedy central': 1, u'kids': 1, u'Twitter': 1, u'elections': 1, u'comedy videos': 1, u'comedian': 1, u'security': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Donald Trump', u'Trump administration', u'international affairs', u'China', u'behaving badly', u"lookin' good", u'Taiwan', u'phones', u'communism', u'democracy', u'censorship', u'Richard Nixon', u'war', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos']
    {u'trevor noah': 1, u'censorship': 1, u'phones': 1, u'international affairs': 1, u'late night talk show hosts': 1, u'new trevor noah show': 1, u'funny': 1, u'behaving badly': 1, u'communism': 1, u'democracy': 1, u'the daily show': 1, u'comedians': 1, u'China': 1, u'Trump administration': 1, u'stand up comedy': 1, u'war': 1, u'Richard Nixon': 1, u'funny video': 1, u'comedy central comedians': 1, u'comedy central': 1, u'the daily show episodes': 1, u"lookin' good": 1, u'comedy central politics': 1, u'Donald Trump': 1, u'comedy': 1, u'comedy videos': 1, u'comedian': 1, u'Taiwan': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'U.K.', u'European Union', u'elections', u'money', u'economy', u'immigration', u'voting', u'history', u'translations', u'rage', u'Donald Trump', u'campaigns', u'candidates', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'U.K.': 1, u'daily show with trevor noah': 1, u'money': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'candidates': 1, u'stand up videos': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'economy': 1, u'comedy central': 1, u'translations': 1, u'immigration': 1, u'funny video': 1, u'comedy central comedians': 1, u'campaigns': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'European Union': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'comedy': 1, u'hilarious clips': 1, u'rage': 1, u'elections': 1, u'comedy videos': 1, u'comedian': 1, u'voting': 1, u'history': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Tales From the Trump Archive', u'Donald Trump', u'Marla Maples', u"Trump's wife", u'Donald Trump women', u'Donald Trump interview', u'Donald Trump 1994', u'late night talk show hosts', u'comedy central', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'Donald Trump interview': 1, u'Donald Trump women': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u"Trump's wife": 1, u'Tales From the Trump Archive': 1, u'the daily show': 1, u'Donald Trump 1994': 1, u'comedians': 1, u'Marla Maples': 1, u'comedy': 1, u'hilarious videos': 1, u'funny': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'funny jokes': 1, u'hilarious clips': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Panama Papers', u'Panama law firm', u'Jackie Chan', u'Vladimir Putin', u'Jacob Zuma', u'Russian president', u'South African president', u'global corruption', u'late night talk show hosts', u'comedy central', u'comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'global corruption': 1, u'comedian': 1, u'comedians': 1, u'Vladimir Putin': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Panama Papers': 1, u'funny': 1, u'the daily show': 1, u'Jackie Chan': 1, u'Panama law firm': 1, u'stand up videos': 1, u'Russian president': 1, u'comedy': 1, u'hilarious videos': 1, u'South African president': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'Jacob Zuma': 1, u'comedy central politics': 1, u'funny jokes': 1, u'hilarious clips': 1, u'comedy videos': 1, u'late night talk show hosts': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Cuba', u'history', u'friends', u'Barack Obama', u'Obama administration', u'business', u'travel', u'hotels/motels', u'Desi Lydic', u'coffee', u'Starbucks', u'restaurants', u'housing', u'Donald Trump', u'refugees', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes']
    {u'trevor noah': 1, u'daily show with trevor noah': 1, u'late night talk show hosts': 1, u'new trevor noah show': 1, u'funny': 1, u'hotels/motels': 1, u'Obama administration': 1, u'travel': 1, u'comedians': 1, u'Starbucks': 1, u'stand up videos': 1, u'stand up comedy': 1, u'coffee': 1, u'business': 1, u'Desi Lydic': 1, u'Cuba': 1, u'funny video': 1, u'comedy central comedians': 1, u'Donald Trump': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'Barack Obama': 1, u'friends': 1, u'comedy central politics': 1, u'comedy central': 1, u'comedy': 1, u'the daily show': 1, u'comedy videos': 1, u'comedian': 1, u'history': 1, u'refugees': 1, u'housing': 1, u'restaurants': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Lewis Black', u'Back in Black', u'rants', u'rage', u'CIA', u'Osama bin Lade', u'death', u'books', u'terrorism', u'money', u'Sudan', u'religion', u'Islam', u'family', u'Al Qaeda', u'work/office', u'global warming', u'friends', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos']
    {u'trevor noah': 1, u'CIA': 1, u'family': 1, u'Sudan': 1, u'rants': 1, u'books': 1, u'late night talk show hosts': 1, u'comedians': 1, u'new trevor noah show': 1, u'funny': 1, u'death': 1, u'the daily show': 1, u'work/office': 1, u'Lewis Black': 1, u'religion': 1, u'stand up videos': 1, u'money': 1, u'terrorism': 1, u'stand up comedy': 1, u'Osama bin Lade': 1, u'comedy central comedians': 1, u'global warming': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny video': 1, u'Back in Black': 1, u'friends': 1, u'comedy central politics': 1, u'comedy': 1, u'rage': 1, u'Al Qaeda': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1, u'Islam': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Uganda', u'elections', u'candidates', u'Back to the Future', u'Africa', u'dictators', u'corruption', u'Yoweri Museveni', u'voting', u'Donald Trump', u'man on the street', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'candidates': 1, u'stand up videos': 1, u'stand up comedy': 1, u'corruption': 1, u'Back to the Future': 1, u'comedy central comedians': 1, u'dictators': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'funny video': 1, u'Yoweri Museveni': 1, u'man on the street': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'comedy': 1, u'Africa': 1, u'elections': 1, u'comedy videos': 1, u'Uganda': 1, u'comedian': 1, u'voting': 1, u'daily show with trevor noah': 1}
    
    [u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'funny': 1, u'stand up comedy': 1, u'hilarious clips': 1, u'comedy central comedians': 1, u'comedians': 1, u'comedy central': 1, u'comedy videos': 1, u'funny jokes': 1, u'comedian': 1, u'funny video': 1, u'stand up videos': 1, u'funny clips': 1, u'comedy': 1, u'hilarious videos': 1}
    
    [u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'funny': 1, u'stand up comedy': 1, u'hilarious clips': 1, u'comedy central comedians': 1, u'comedians': 1, u'comedy central': 1, u'comedy videos': 1, u'funny jokes': 1, u'comedian': 1, u'funny video': 1, u'stand up videos': 1, u'funny clips': 1, u'comedy': 1, u'hilarious videos': 1}
    
    [u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'funny': 1, u'stand up comedy': 1, u'hilarious clips': 1, u'comedy central comedians': 1, u'comedians': 1, u'comedy central': 1, u'comedy videos': 1, u'funny jokes': 1, u'comedian': 1, u'funny video': 1, u'stand up videos': 1, u'funny clips': 1, u'comedy': 1, u'hilarious videos': 1}
    
    [u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'funny': 1, u'stand up comedy': 1, u'hilarious clips': 1, u'comedy central comedians': 1, u'comedians': 1, u'comedy central': 1, u'comedy videos': 1, u'funny jokes': 1, u'comedian': 1, u'funny video': 1, u'stand up videos': 1, u'funny clips': 1, u'comedy': 1, u'hilarious videos': 1}
    
    [u'The Daily Show', u'Daily Show videos', u'the daily show', u'jon stewart', u'john stewart', u'comedy central politics', u'the daily show episodes', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'comedy central politics': 1, u'funny': 1, u'stand up comedy': 1, u'hilarious clips': 1, u'the daily show': 1, u'comedy videos': 1, u'jon stewart': 1, u'Daily Show videos': 1, u'comedy central comedians': 1, u'comedians': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedian': 1, u'funny video': 1, u'stand up videos': 1, u'The Daily Show': 1, u'funny clips': 1, u'john stewart': 1, u'comedy': 1, u'hilarious videos': 1}
    
    [u'The Daily Show', u'Daily Show videos', u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Canada', u'Drake', u'music', u'music videos', u'dancing', u'hip hop', u"lookin' good", u'hockey', u'sports', u'elections', u'candidates', u'liberal', u'debt', u'money', u'songs', u'Stephen Harper', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos']
    {u'Canada': 1, u'trevor noah': 1, u'liberal': 1, u'money': 1, u'Daily Show videos': 1, u'late night talk show hosts': 1, u'Stephen Harper': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'sports': 1, u'music': 1, u'hockey': 1, u'music videos': 1, u'candidates': 1, u'stand up comedy': 1, u'The Daily Show': 1, u'comedy central comedians': 1, u'Drake': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny video': 1, u"lookin' good": 1, u'debt': 1, u'comedy central politics': 1, u'comedy': 1, u'dancing': 1, u'elections': 1, u'comedy videos': 1, u'hip hop': 1, u'comedian': 1, u'songs': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Donald Trump president', u'African presidents', u'Donald Trump', u'Africa', u'South Africa', u'Zimbabwe', u'Idi Amin', u'Robert Mugabe', u'Muammar al-Gaddafi', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos']
    {u'trevor noah': 1, u'Robert Mugabe': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Donald Trump': 1, u'funny': 1, u'Donald Trump president': 1, u'the daily show': 1, u'comedians': 1, u'Zimbabwe': 1, u'stand up videos': 1, u'comedian': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'Muammar al-Gaddafi': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'African presidents': 1, u'comedy': 1, u'South Africa': 1, u'Africa': 1, u'comedy videos': 1, u'late night talk show hosts': 1, u'Idi Amin': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Jason Isbell', u'performances', u'exclusives', u'songs', u'music', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'Exclusive', u'If We Were Vampires', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'Jason Isbell': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'music': 1, u'stand up videos': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'exclusives': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'Exclusive': 1, u'comedy': 1, u'hilarious clips': 1, u'performances': 1, u'If We Were Vampires': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1, u'songs': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Alabama Week - Prejudice & Pigskin', u'Alabama', u'football', u'Jordan Klepper', u'college', u'arguments', u'exclusives', u'Roll Tide', u'Auburn', u'call-in show', u'late night talk show hosts', u'comedy central', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'Jordan Klepper': 1, u'college': 1, u'Alabama Week - Prejudice & Pigskin': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Roll Tide': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'arguments': 1, u'comedian': 1, u'comedy': 1, u'hilarious videos': 1, u'football': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'Alabama': 1, u'comedy central politics': 1, u'exclusives': 1, u'funny jokes': 1, u'hilarious clips': 1, u'Auburn': 1, u'comedy videos': 1, u'call-in show': 1, u'late night talk show hosts': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u"LEAKED: Fox News' 8 p.m. Anchor Audition Tape", u'Fox News', u'exclusives', u"Bill O'Reilly", u"The O'Reilly Factor", u'TV', u'sexism', u'scandals', u'discrimination', u'men/women', u'sexual advances', u'sex', u'montages late night talk show hosts', u'comedy central', u'comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'sex': 1, u'funny': 1, u'comedian': 1, u'sexism': 1, u'funny clips': 1, u'new trevor noah show': 1, u'scandals': 1, u'men/women': 1, u'the daily show': 1, u'TV': 1, u'discrimination': 1, u'comedians': 1, u'comedy': 1, u'Fox News': 1, u'hilarious videos': 1, u"Bill O'Reilly": 1, u'funny video': 1, u'montages late night talk show hosts': 1, u'comedy central': 1, u'the daily show episodes': 1, u"LEAKED: Fox News' 8 p.m. Anchor Audition Tape": 1, u'comedy central politics': 1, u'exclusives': 1, u'funny jokes': 1, u'hilarious clips': 1, u'sexual advances': 1, u"The O'Reilly Factor": 1, u'comedy videos': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Eliza Cossio', u'Please', u"Just Like\u2026 Don't: How to Protest", u'protests', u'media', u'African American', u'Muslim', u'Nancy Pelosi', u'Madonna', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'protests': 1, u'Please': 1, u'late night talk show hosts': 1, u'Eliza Cossio': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'media': 1, u'Muslim': 1, u'comedians': 1, u'stand up videos': 1, u'comedian': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'Nancy Pelosi': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'African American': 1, u'comedy central politics': 1, u'comedy': 1, u'hilarious clips': 1, u'comedy videos': 1, u'Madonna': 1, u'daily show with trevor noah': 1, u"Just Like\u2026 Don't: How to Protest": 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Zara Larsson', u'exclusives', u'performances', u'songs', u'music', u'So Good', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'So Good': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'music': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'hilarious clips': 1, u'performances': 1, u'Zara Larsson': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1, u'songs': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'exclusives', u'Ben Carson', u'Jesse Williams', u"Grey's Anatomy", u'TV', u'parody', u'impressions', u'slavery', u'surgery', u'doctors', u'Brain Doctors MD', u'brain surgery', u'late night talk show hosts', u'comedy central', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'Jesse Williams': 1, u'late night talk show hosts': 1, u'impressions': 1, u'surgery': 1, u'new trevor noah show': 1, u'funny clips': 1, u'funny': 1, u'the daily show': 1, u'TV': 1, u"Grey's Anatomy": 1, u'comedians': 1, u'comedy': 1, u'hilarious videos': 1, u'brain surgery': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'doctors': 1, u'comedy central politics': 1, u'exclusives': 1, u'funny jokes': 1, u'hilarious clips': 1, u'slavery': 1, u'Brain Doctors MD': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1, u'Ben Carson': 1, u'parody': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'internet troll doll', u'holiday gift guide', u'bern the toast', u'exclusives', u'Ronny Chieng', u'gifts & presents', u'holidays', u'Donald Trump', u'Star Wars', u'movies', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'holiday gift guide': 1, u'internet troll doll': 1, u'comedian': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Donald Trump': 1, u'gifts & presents': 1, u'the daily show': 1, u'comedians': 1, u'stand up videos': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'Star Wars': 1, u'funny': 1, u'comedy central comedians': 1, u'funny video': 1, u'holidays': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'Ronny Chieng': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'hilarious clips': 1, u'bern the toast': 1, u'movies': 1, u'comedy videos': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'orange president', u'Zika', u'couch body', u'fake news', u'garbage year', u'Hasan Minhaj', u'elections', u'Donald Trump', u'health', u'Facebook', u'media', u'New Year', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'couch body': 1, u'funny': 1, u'New Year': 1, u'the daily show': 1, u'media': 1, u'comedians': 1, u'health': 1, u'stand up videos': 1, u'Facebook': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'Zika': 1, u'fake news': 1, u'orange president': 1, u'comedy central comedians': 1, u'funny video': 1, u'Donald Trump': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'comedy central': 1, u'comedy': 1, u'garbage year': 1, u'elections': 1, u'comedy videos': 1, u'comedian': 1, u'Hasan Minhaj': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Yule log', u'burning Constitution', u'Trump quotes', u'bing bing bong bong', u'exclusives', u'holidays', u'Christmas', u'Donald Trump', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'funny clips': 1, u'late night talk show hosts': 1, u'burning Constitution': 1, u'Trump quotes': 1, u'Christmas': 1, u'bing bing bong bong': 1, u'funny': 1, u'new trevor noah show': 1, u'the daily show': 1, u'comedians': 1, u'stand up videos': 1, u'comedy': 1, u'hilarious videos': 1, u'comedy central': 1, u'comedy central comedians': 1, u'funny video': 1, u'holidays': 1, u'Donald Trump': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'exclusives': 1, u'stand up comedy': 1, u'hilarious clips': 1, u'Yule log': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Donald Trump', u'Trump administration', u'Ted Cruz', u'Rudy Giuliani', u'racism', u'Roy Wood Jr.', u'Paul Ryan', u'Mitch McConnell', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'Roy Wood Jr.': 1, u'Donald Trump': 1, u'Ted Cruz': 1, u'funny': 1, u'new trevor noah show': 1, u'the daily show': 1, u'racism': 1, u'comedians': 1, u'Trump administration': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'stand up videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'Paul Ryan': 1, u'Mitch McConnell': 1, u'comedy central politics': 1, u'Rudy Giuliani': 1, u'comedy': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Roy Wood Jr.', u'exclusives', u'Thanksgiving', u'holidays', u'songs', u'music', u'hip hop', u'Donald Trump', u'elections', u'family', u'dancing', u'food', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'family': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'Roy Wood Jr.': 1, u'funny': 1, u'new trevor noah show': 1, u'the daily show': 1, u'comedy central comedians': 1, u'comedians': 1, u'music': 1, u'stand up videos': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'exclusives': 1, u'food': 1, u'Thanksgiving': 1, u'funny video': 1, u'holidays': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'comedy': 1, u'hilarious clips': 1, u'dancing': 1, u'elections': 1, u'comedy videos': 1, u'hip hop': 1, u'comedian': 1, u'daily show with trevor noah': 1, u'songs': 1}
    
    [u'The Daily Show with Trevor Noah', u'Trevor Noah Daily Show', u'exclusives', u'Desi Lydic', u'elections', u'voting', u'Hasan Minhaj', u'partying', u'Donald Trump', u'Hillary Clinton comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'partying': 1, u'comedian': 1, u'funny clips': 1, u'funny': 1, u'The Daily Show with Trevor Noah': 1, u'comedians': 1, u'stand up videos': 1, u'stand up comedy': 1, u'comedy': 1, u'hilarious videos': 1, u'Desi Lydic': 1, u'comedy central comedians': 1, u'funny video': 1, u'Donald Trump': 1, u'Hillary Clinton comedy central': 1, u'Hasan Minhaj': 1, u'exclusives': 1, u'funny jokes': 1, u'hilarious clips': 1, u'elections': 1, u'comedy videos': 1, u'Trevor Noah Daily Show': 1, u'voting': 1}
    
    [u'the daily show', u'jon stewart', u'john stewart', u'comedy central politics', u'the daily show episodes', u'exclusives', u'Donald Trump', u'exercise', u'fitness', u'sexism', u'sexual assault', u"grab 'em by the pussy", u'tic tacs', u'small hands', u'violence', u'working out', u'bros', u'assholes', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'assholes': 1, u'bros': 1, u'late night talk show hosts': 1, u'sexism': 1, u'funny clips': 1, u'john stewart': 1, u'Donald Trump': 1, u'funny': 1, u'the daily show': 1, u'small hands': 1, u'sexual assault': 1, u'working out': 1, u'comedians': 1, u'stand up videos': 1, u'stand up comedy': 1, u'comedy': 1, u'exercise': 1, u'jon stewart': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'exclusives': 1, u'tic tacs': 1, u'hilarious clips': 1, u'violence': 1, u'comedy videos': 1, u'hilarious videos': 1, u'fitness': 1, u'comedian': 1, u"grab 'em by the pussy": 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Eric the Eel', u'Olympic parody', u'Olympics', u'sports', u'Roy Wood Jr.', u'Africa', u'swimming', u'awards', u'Daily Show writers', u'uncensored', u'Equatorial Guinea', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos']
    {u'trevor noah': 1, u'uncensored': 1, u'late night talk show hosts': 1, u'Roy Wood Jr.': 1, u'funny': 1, u'new trevor noah show': 1, u'the daily show': 1, u'Olympic parody': 1, u'comedians': 1, u'sports': 1, u'comedian': 1, u'stand up comedy': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'awards': 1, u'Daily Show writers': 1, u'Eric the Eel': 1, u'comedy central politics': 1, u'comedy': 1, u'Equatorial Guinea': 1, u'Africa': 1, u'comedy videos': 1, u'Olympics': 1, u'daily show with trevor noah': 1, u'swimming': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Ronny Chieng', u'exclusives', u'Philadelphia', u'DNC', u'Democratic National Convention', u'conventions', u'shopping', u'food', u'gross-out', u'health', u'poop & pee', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos']
    {u'trevor noah': 1, u'Philadelphia': 1, u'conventions': 1, u'late night talk show hosts': 1, u'comedians': 1, u'new trevor noah show': 1, u'poop & pee': 1, u'funny': 1, u'the daily show': 1, u'gross-out': 1, u'health': 1, u'stand up videos': 1, u'stand up comedy': 1, u'shopping': 1, u'DNC': 1, u'food': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'Ronny Chieng': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'comedy videos': 1, u'comedian': 1, u'Democratic National Convention': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'exclusives', u'elections', u'Hillary Clinton', u'Donald Trump', u'Roy Wood Jr.', u'Desi Lydic', u'Jordan Klepper', u'Hasan Minhaj', u'competitions', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'Ronny Chieng']
    {u'trevor noah': 1, u'Hillary Clinton': 1, u'Jordan Klepper': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'Roy Wood Jr.': 1, u'funny': 1, u'new trevor noah show': 1, u'the daily show': 1, u'comedians': 1, u'competitions': 1, u'stand up videos': 1, u'stand up comedy': 1, u'exclusives': 1, u'Desi Lydic': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'Ronny Chieng': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'comedy': 1, u'elections': 1, u'comedy videos': 1, u'comedian': 1, u'Hasan Minhaj': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'exclusives', u'elections', u'Desi Lydic', u'LGBT', u'Barack Obama', u'speeches', u'Mitt Romney', u'history', u'dancing', u'Democratic National Convention', u'Republican National Convention', u'Obama', u'Vietnam War', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'stand up comedy': 1, u'comedy': 1, u'Obama': 1, u'Vietnam War': 1, u'Desi Lydic': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'LGBT': 1, u'the daily show episodes': 1, u'Barack Obama': 1, u'Republican National Convention': 1, u'comedy central politics': 1, u'exclusives': 1, u'speeches': 1, u'Mitt Romney': 1, u'dancing': 1, u'elections': 1, u'comedian': 1, u'daily show with trevor noah': 1, u'Democratic National Convention': 1, u'history': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Donald Trump', u'Donald Trump campaign', u'Donald Trump cabinet', u'Donald Trump quotes', u'Trump cabinet', u'Donald Trump flip flip', u'election 2016', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes']
    {u'trevor noah': 1, u'Donald Trump quotes': 1, u'late night talk show hosts': 1, u'new trevor noah show': 1, u'Donald Trump campaign': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'election 2016': 1, u'stand up videos': 1, u'comedian': 1, u'stand up comedy': 1, u'Trump cabinet': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'comedy': 1, u'comedy videos': 1, u'Donald Trump cabinet': 1, u'Donald Trump flip flip': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Jessica Williams', u'racism', u'guns', u'YouTube', u'dancing', u'white people', u'mashups', u'African American', u'Beyonce', u'LGBT', u'PSAs', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'Beyonce': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'Jessica Williams': 1, u'the daily show': 1, u'YouTube': 1, u'racism': 1, u'comedians': 1, u'stand up videos': 1, u'stand up comedy': 1, u'guns': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'PSAs': 1, u'comedy central': 1, u'LGBT': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'African American': 1, u'comedy central politics': 1, u'comedy': 1, u'hilarious clips': 1, u'white people': 1, u'dancing': 1, u'mashups': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Jordan Klepper', u'exclusives', u'uncensored', u'Illinois', u'money', u'economy', u'government', u'Congress', u'Democrats', u'Republicans', u'arguments', u'animals', u'Adam Sandler', u'movies', u'toys & games', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos']
    {u'trevor noah': 1, u'uncensored': 1, u'money': 1, u'Illinois': 1, u'Jordan Klepper': 1, u'late night talk show hosts': 1, u'new trevor noah show': 1, u'funny': 1, u'Congress': 1, u'the daily show': 1, u'toys & games': 1, u'Republicans': 1, u'comedians': 1, u'arguments': 1, u'stand up videos': 1, u'stand up comedy': 1, u'economy': 1, u'government': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'comedy central politics': 1, u'exclusives': 1, u'Adam Sandler': 1, u'comedy': 1, u'animals': 1, u'movies': 1, u'comedy videos': 1, u'Democrats': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Desi Lydic', u'exclusives', u'uncensored', u'Donald Trump', u'birthdays', u'New York City', u'man on the street', u'singing', u'songs', u'middle finger', u'insults', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video']
    {u'trevor noah': 1, u'uncensored': 1, u'late night talk show hosts': 1, u'new trevor noah show': 1, u'singing': 1, u'funny': 1, u'middle finger': 1, u'New York City': 1, u'birthdays': 1, u'comedians': 1, u'stand up comedy': 1, u'exclusives': 1, u'Desi Lydic': 1, u'comedy central comedians': 1, u'funny video': 1, u'insults': 1, u'comedy central': 1, u'the daily show episodes': 1, u'man on the street': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'comedy': 1, u'the daily show': 1, u'comedian': 1, u'daily show with trevor noah': 1, u'songs': 1}
    
    [u'he daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'rap video', u'make America great again', u'very intelligent', u'Trump rap', u'Roy Wood Jr.', u'Jordan Klepper', u'hip hop', u'songs', u'music videos', u'Donald Trump', u'elections', u'exclusives', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy']
    {u'trevor noah': 1, u'very intelligent': 1, u'Jordan Klepper': 1, u'late night talk show hosts': 1, u'Roy Wood Jr.': 1, u'Trump rap': 1, u'new trevor noah show': 1, u'comedians': 1, u'rap video': 1, u'music videos': 1, u'stand up comedy': 1, u'exclusives': 1, u'make America great again': 1, u'comedy central comedians': 1, u'comedy central': 1, u'the daily show episodes': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'comedy': 1, u'he daily show': 1, u'elections': 1, u'hip hop': 1, u'daily show with trevor noah': 1, u'songs': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'thanksgiving', u'turkey jokes', u'thanksgiving jokes', u'embarrassing moments', u'turkey time', u'talk turkey', u'talk show hosts', u'pundits', u'fox & friends', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos']
    {u'trevor noah': 1, u'thanksgiving jokes': 1, u'pundits': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'stand up videos': 1, u'comedian': 1, u'stand up comedy': 1, u'turkey jokes': 1, u'embarrassing moments': 1, u'talk turkey': 1, u'comedy central comedians': 1, u'turkey time': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny video': 1, u'talk show hosts': 1, u'fox & friends': 1, u'comedy central politics': 1, u'comedy': 1, u'thanksgiving': 1, u'comedy videos': 1, u'late night talk show hosts': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'DJ Khaled', u'major talk', u'keys to success', u'hotline bling', u'sneaker collection', u'Hasan Minhaj', u'fashion', u'exclusives', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'fashion': 1, u'major talk': 1, u'comedian': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'stand up videos': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'DJ Khaled': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'sneaker collection': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'hotline bling': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'hilarious clips': 1, u'keys to success': 1, u'comedy videos': 1, u'late night talk show hosts': 1, u'Hasan Minhaj': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Hasan Minhaj', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips', u'music', u'theme song', u'Timbaland', u'They Might Be Giants']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'music': 1, u'stand up videos': 1, u'Timbaland': 1, u'stand up comedy': 1, u'They Might Be Giants': 1, u'comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'comedy central politics': 1, u'theme song': 1, u'funny jokes': 1, u'hilarious clips': 1, u'comedy videos': 1, u'comedian': 1, u'Hasan Minhaj': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Roy Wood Jr.', u'exclusives', u'porn', u'costumes', u'African American', u'stereotypes', u'Ron Jeremy', u'sex', u'impressions', u'Jordan Klepper', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'porn': 1, u'sex': 1, u'Jordan Klepper': 1, u'late night talk show hosts': 1, u'impressions': 1, u'funny clips': 1, u'Roy Wood Jr.': 1, u'funny': 1, u'new trevor noah show': 1, u'the daily show': 1, u'comedians': 1, u'stand up videos': 1, u'comedy': 1, u'hilarious videos': 1, u'Ron Jeremy': 1, u'comedy central comedians': 1, u'funny video': 1, u'costumes': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'African American': 1, u'stereotypes': 1, u'comedy central politics': 1, u'exclusives': 1, u'stand up comedy': 1, u'hilarious clips': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Jordan klepper Donald trump', u'Donald trump supporters', u'Donald trump rally', u'Jordan klepper interview', u'2016 election', u'Donald Trump fan', u'trump fans', u'late night talk show hosts', u'comedy central', u'comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'Donald trump rally': 1, u'Donald Trump fan': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Jordan klepper Donald trump': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'comedy': 1, u'hilarious videos': 1, u'Donald trump supporters': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'2016 election': 1, u'hilarious clips': 1, u'trump fans': 1, u'comedy central politics': 1, u'funny jokes': 1, u'Jordan klepper interview': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'New Hampshire', u'exclusives', u'Michael Steele', u'Howard Dean', u'Ryan Lizza', u'Alicia Menendez', u'Ronny Chieng', u'Roy Wood Jr.', u'Jordan Klepper', u'Hasan Minhaj', u'late night talk show hosts', u'comedy central', u'comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'Michael Steele': 1, u'Jordan Klepper': 1, u'comedian': 1, u'funny clips': 1, u'Roy Wood Jr.': 1, u'funny': 1, u'new trevor noah show': 1, u'the daily show': 1, u'Ryan Lizza': 1, u'comedy videos': 1, u'comedians': 1, u'Howard Dean': 1, u'New Hampshire': 1, u'comedy': 1, u'hilarious videos': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'Ronny Chieng': 1, u'comedy central politics': 1, u'exclusives': 1, u'funny jokes': 1, u'hilarious clips': 1, u'Alicia Menendez': 1, u'late night talk show hosts': 1, u'Hasan Minhaj': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Roy Wood Jr.', u'Jessica Williams', u'Ronny Chieng', u'Jordan Klepper', u'Hasan Minhaj', u'New Hampshire', u'elections', u'media', u'drugs', u'exclusives', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos']
    {u'trevor noah': 1, u'Jordan Klepper': 1, u'late night talk show hosts': 1, u'Roy Wood Jr.': 1, u'funny': 1, u'new trevor noah show': 1, u'Jessica Williams': 1, u'the daily show': 1, u'media': 1, u'comedians': 1, u'New Hampshire': 1, u'stand up videos': 1, u'stand up comedy': 1, u'drugs': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'Ronny Chieng': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'elections': 1, u'comedy videos': 1, u'comedian': 1, u'Hasan Minhaj': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Drake', u'facial hair', u'Hasan Minhaj', u'Justin Trudeau', u'Canada', u'interviews', u'exclusives', u'apologies', u'Degrassi', u'goatee', u'Movember', u'comedy central', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'Canada': 1, u'trevor noah': 1, u'Justin Trudeau': 1, u'facial hair': 1, u'comedian': 1, u'apologies': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'Degrassi': 1, u'the daily show': 1, u'interviews': 1, u'comedians': 1, u'stand up videos': 1, u'comedy': 1, u'hilarious videos': 1, u'goatee': 1, u'comedy central comedians': 1, u'Drake': 1, u'comedy central': 1, u'the daily show episodes': 1, u'Movember': 1, u'funny video': 1, u'comedy central politics': 1, u'exclusives': 1, u'funny jokes': 1, u'hilarious clips': 1, u'comedy videos': 1, u'Hasan Minhaj': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Washington Post interview', u'private parts', u'reenactment', u'Roy Wood Jr.', u'exclusives', u'Donald Trump', u'reenactments', u'Washington Post', u'Marco Rubio', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos']
    {u'Washington Post interview': 1, u'trevor noah': 1, u'late night talk show hosts': 1, u'Washington Post': 1, u'funny clips': 1, u'Roy Wood Jr.': 1, u'funny': 1, u'new trevor noah show': 1, u'the daily show': 1, u'comedians': 1, u'stand up videos': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'exclusives': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'Marco Rubio': 1, u'funny jokes': 1, u'reenactment': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'comedy': 1, u'reenactments': 1, u'private parts': 1, u'the daily show episodes': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'black Richard Gere', u'interviews', u'Barack Obama', u'aging', u'Hasan Minhaj', u'Justin Trudeau', u'Canada', u'Bernie Sanders', u'exclusives', u"lookin' good", u'comedy central', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'Canada': 1, u'trevor noah': 1, u'Justin Trudeau': 1, u'comedian': 1, u'funny clips': 1, u'new trevor noah show': 1, u'aging': 1, u'the daily show': 1, u'interviews': 1, u'comedians': 1, u'stand up videos': 1, u'Bernie Sanders': 1, u'comedy': 1, u'hilarious videos': 1, u'funny': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'Barack Obama': 1, u"lookin' good": 1, u'comedy central politics': 1, u'exclusives': 1, u'funny jokes': 1, u'hilarious clips': 1, u'black Richard Gere': 1, u'comedy videos': 1, u'Hasan Minhaj': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'interviews', u'Justin Trudeau', u'Canada', u'Hasan Minhaj', u'dating', u'movies', u'men/women', u'boxing', u'exclusives', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'Canada': 1, u'trevor noah': 1, u'Justin Trudeau': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'men/women': 1, u'the daily show': 1, u'interviews': 1, u'boxing': 1, u'comedians': 1, u'stand up videos': 1, u'dating': 1, u'comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'exclusives': 1, u'stand up comedy': 1, u'hilarious clips': 1, u'movies': 1, u'comedy videos': 1, u'comedian': 1, u'Hasan Minhaj': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Joe Morton', u'start the show', u'Scandal', u'papa Pope', u'Rowan Pope', u'exclusives', u'parody', u'TV', u'Jordan Klepper', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'papa Pope': 1, u'start the show': 1, u'Rowan Pope': 1, u'Jordan Klepper': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'TV': 1, u'comedians': 1, u'Joe Morton': 1, u'stand up videos': 1, u'comedian': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'hilarious clips': 1, u'comedy videos': 1, u'Scandal': 1, u'daily show with trevor noah': 1, u'parody': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'compliment competition', u'Sandrew', u'Hasan Minhaj', u'Ellie Kemper', u'Unbreakable Kimmy Schmidt', u'mental health', u'pets', u'competitions', u'growing up', u'poverty', u'exclusives', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'Unbreakable Kimmy Schmidt': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'mental health': 1, u'the daily show': 1, u'comedians': 1, u'Sandrew': 1, u'competitions': 1, u'pets': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'stand up videos': 1, u'Ellie Kemper': 1, u'comedy central comedians': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'poverty': 1, u'growing up': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'hilarious clips': 1, u'compliment competition': 1, u'comedy videos': 1, u'Hasan Minhaj': 1, u'daily show with trevor noah': 1}
    
    [u'The Daily Show', u'Daily Show videos', u'exclusives', u'Hasan Minhaj', u'work/office', u'crime', u"McDonald's", u'target', u'laws', u'Law & Order', u'civil rights', u'restaurants', u'Trevor Noah']
    {u'exclusives': 1, u'target': 1, u'The Daily Show': 1, u'work/office': 1, u'civil rights': 1, u'Daily Show videos': 1, u"McDonald's": 1, u'crime': 1, u'Law & Order': 1, u'Hasan Minhaj': 1, u'restaurants': 1, u'Trevor Noah': 1, u'laws': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'police sensitivity training', u'racial bias', u'police bias', u'african americans', u'racism', u'police racism', u'racist cops', u'Jordan Klepper', u'roy wood jr.', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos']
    {u'trevor noah': 1, u'Jordan Klepper': 1, u'late night talk show hosts': 1, u'police racism': 1, u'new trevor noah show': 1, u'funny': 1, u'police bias': 1, u'the daily show': 1, u'racism': 1, u'comedians': 1, u'roy wood jr.': 1, u'african americans': 1, u'comedy': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'racist cops': 1, u'comedy central politics': 1, u'stand up comedy': 1, u'police sensitivity training': 1, u'racial bias': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'John Kasich - Uniting America in', u'The Daily Show interview', u'interviews', u'extended interviews', u'exclusives', u'Ohio', u'books', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'books': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'John Kasich - Uniting America in': 1, u'interviews': 1, u'comedians': 1, u'stand up videos': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'extended interviews': 1, u'Ohio': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'hilarious clips': 1, u'The Daily Show interview': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Keegan-Michael Key', u'interviews', u'TV', u'Key & Peele', u'Jordan Peele', u'elections', u'Donald Trump', u'Barack Obama', u'Obama administration', u'rage', u'rants', u'Luther', u"Obama's Anger Translator", u'Trump administration', u'discrimination', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian']
    {u'trevor noah': 1, u'Key & Peele': 1, u'rants': 1, u'late night talk show hosts': 1, u'Keegan-Michael Key': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'TV': 1, u'discrimination': 1, u'interviews': 1, u'comedians': 1, u'Trump administration': 1, u'stand up comedy': 1, u'Luther': 1, u'comedy central': 1, u"Obama's Anger Translator": 1, u'comedy central comedians': 1, u'Donald Trump': 1, u'the daily show episodes': 1, u'Barack Obama': 1, u'comedy central politics': 1, u'Jordan Peele': 1, u'comedy': 1, u'rage': 1, u'elections': 1, u'Obama administration': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Cecile Richards', u'Cecile Richards interview', u'planned parenthood', u'defunding planned parenthood', u'obamacare', u'paul ryan', u'trump administration', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos']
    {u'trevor noah': 1, u'Cecile Richards': 1, u'late night talk show hosts': 1, u'paul ryan': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'obamacare': 1, u'comedians': 1, u'stand up comedy': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'trump administration': 1, u'planned parenthood': 1, u'comedy central politics': 1, u'comedy': 1, u'defunding planned parenthood': 1, u'Cecile Richards interview': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Tomi Lahren', u'Tomi Lahren interview', u'Tomi', u'TheBlaze', u'Black Lives Matter', u'Colin Kaepernick', u'Tomi Lahren show', u'Tomi Lahren Trevor Noah', u'late night talk show hosts', u'comedy central', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'Tomi Lahren Trevor Noah': 1, u'Tomi Lahren interview': 1, u'TheBlaze': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'comedy videos': 1, u'comedians': 1, u'Colin Kaepernick': 1, u'comedy': 1, u'Black Lives Matter': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'Tomi': 1, u'the daily show episodes': 1, u'comedy central politics': 1, u'funny jokes': 1, u'hilarious clips': 1, u'Tomi Lahren': 1, u'Tomi Lahren show': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'police brutality', u'Wesley Lowery', u'police shootings', u'Ferguson', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'Wesley Lowery': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'police shootings': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'Ferguson': 1, u'stand up videos': 1, u'stand up comedy': 1, u'comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'police brutality': 1, u'comedy central politics': 1, u'funny jokes': 1, u'hilarious clips': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'George Packer', u'white working class', u'Donald Trump', u'Bill Clinton', u'globalization', u'populism', u'fake news', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'white working class': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'globalization': 1, u'stand up videos': 1, u'stand up comedy': 1, u'George Packer': 1, u'fake news': 1, u'Bill Clinton': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'comedy': 1, u'hilarious clips': 1, u'comedy videos': 1, u'hilarious videos': 1, u'populism': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Bill Clinton', u'interviews', u'Hillary Clinton', u'elections', u'campaigns', u'candidates', u'insults', u'discrimination', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'elections': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'discrimination': 1, u'interviews': 1, u'comedians': 1, u'candidates': 1, u'stand up videos': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'Bill Clinton': 1, u'comedy central comedians': 1, u'funny video': 1, u'insults': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'campaigns': 1, u'comedy': 1, u'hilarious clips': 1, u'Hillary Clinton': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Hannah Hart', u'interviews', u'extended interviews', u'exclusives', u'movies', u'alcohol', u'cooking', u'drunk', u'Uber', u'books', u'YouTube', u'aging', u'Mamrie Hart', u'puns/wordplay', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips']
    {u'trevor noah': 1, u'drunk': 1, u'puns/wordplay': 1, u'books': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'Mamrie Hart': 1, u'aging': 1, u'new trevor noah show': 1, u'alcohol': 1, u'the daily show': 1, u'interviews': 1, u'cooking': 1, u'YouTube': 1, u'comedians': 1, u'funny video': 1, u'stand up videos': 1, u'stand up comedy': 1, u'Uber': 1, u'funny': 1, u'comedy central comedians': 1, u'Hannah Hart': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'extended interviews': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'movies': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'John Lewis', u'interviews', u'extended interviews', u'exclusives', u'books', u'Black Lives Matter', u'Martin Luther King Jr.', u'protests', u'African American', u'death', u'murder', u'violence', u'police business', u'racism', u'discrimination', u'voting', u'laws', u'civil rights', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian']
    {u'trevor noah': 1, u'protests': 1, u'funny': 1, u'books': 1, u'late night talk show hosts': 1, u'police business': 1, u'new trevor noah show': 1, u'John Lewis': 1, u'death': 1, u'the daily show': 1, u'discrimination': 1, u'interviews': 1, u'racism': 1, u'comedians': 1, u'stand up comedy': 1, u'Black Lives Matter': 1, u'Martin Luther King Jr.': 1, u'murder': 1, u'laws': 1, u'civil rights': 1, u'comedy central comedians': 1, u'extended interviews': 1, u'comedy central': 1, u'the daily show episodes': 1, u'African American': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'violence': 1, u'comedian': 1, u'voting': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Rand Paul', u'exclusive', u'debates', u'Republicans', u'elections', u'candidates', u'drinking games', u'alcohol', u'Kentucky', u'Middle East', u'wars', u'ISIS', u'terrorism', u'violence', u'bombs', u'weapons', u'impressions', u'economy', u'work/office', u'taxes', u'government', u'wealth', u'money', u'Libertarians', u'drugs', u'late night talk show hosts', u'comedy central', u'comedy', u'funny', u'funny video', u'comedy videos', u'funny clips', u'hilarious videos']
    {u'trevor noah': 1, u'exclusive': 1, u'money': 1, u'late night talk show hosts': 1, u'Middle East': 1, u'impressions': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'ISIS': 1, u'alcohol': 1, u'the daily show': 1, u'work/office': 1, u'Republicans': 1, u'drinking games': 1, u'weapons': 1, u'candidates': 1, u'terrorism': 1, u'comedy': 1, u'hilarious videos': 1, u'economy': 1, u'Rand Paul': 1, u'government': 1, u'drugs': 1, u'wars': 1, u'Kentucky': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny video': 1, u'wealth': 1, u'comedy central politics': 1, u'debates': 1, u'Libertarians': 1, u'violence': 1, u'elections': 1, u'taxes': 1, u'comedy videos': 1, u'bombs': 1, u'daily show with trevor noah': 1}
    
    [u'Ben Carson', u'interviews', u'doctors', u'surgery', u'siblings', u'debates', u'elections', u'candidates', u'impressions', u'Desi Lydic', u'Donald Trump', u'sleep', u'Affordable Care Act', u'slavery', u'history', u'guns', u'safety', u'weapons', u'Second Amendment', u'Constitution', u'parents', u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes']
    {u'trevor noah': 1, u'daily show with trevor noah': 1, u'Constitution': 1, u'slavery': 1, u'sleep': 1, u'impressions': 1, u'surgery': 1, u'new trevor noah show': 1, u'the daily show': 1, u'interviews': 1, u'weapons': 1, u'parents': 1, u'siblings': 1, u'Affordable Care Act': 1, u'safety': 1, u'guns': 1, u'Desi Lydic': 1, u'Second Amendment': 1, u'Donald Trump': 1, u'the daily show episodes': 1, u'doctors': 1, u'comedy central politics': 1, u'debates': 1, u'elections': 1, u'history': 1, u'Ben Carson': 1, u'candidates': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Donald Trump', u'Republicans', u'elections', u'candidates', u'Lindsey Graham', u'Ted Cruz', u'interviews', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Ted Cruz': 1, u'funny': 1, u'the daily show': 1, u'interviews': 1, u'Republicans': 1, u'comedians': 1, u'candidates': 1, u'stand up videos': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'comedy': 1, u'hilarious clips': 1, u'Lindsey Graham': 1, u'elections': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'he daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Lilly Singh', u'interviews', u'YouTube', u'movie', u'documentary', u"lookin' good", u'fans', u'mental health', u'school', u'college', u'parents', u'impressions', u'Dwayne', u'TV', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'impressions': 1, u'new trevor noah show': 1, u'funny': 1, u'mental health': 1, u'TV': 1, u'movie': 1, u'interviews': 1, u'YouTube': 1, u'comedians': 1, u'fans': 1, u'parents': 1, u'stand up videos': 1, u'comedian': 1, u'stand up comedy': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'Lilly Singh': 1, u"lookin' good": 1, u'comedy central politics': 1, u'school': 1, u'comedy': 1, u'he daily show': 1, u'documentary': 1, u'the daily show episodes': 1, u'comedy videos': 1, u'college': 1, u'Dwayne': 1, u'daily show with trevor noah': 1}
    
    [u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'funny': 1, u'stand up comedy': 1, u'hilarious clips': 1, u'comedy central comedians': 1, u'comedians': 1, u'comedy central': 1, u'comedy videos': 1, u'funny jokes': 1, u'comedian': 1, u'funny video': 1, u'stand up videos': 1, u'funny clips': 1, u'comedy': 1, u'hilarious videos': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Kevin Hart', u'interviews', u'Ride along 2', u'what now', u'real husbands of Hollywood', u'male bitch', u'mitch', u'exercise', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'mitch': 1, u'the daily show': 1, u'interviews': 1, u'comedians': 1, u'stand up videos': 1, u'Kevin Hart': 1, u'stand up comedy': 1, u'exercise': 1, u'male bitch': 1, u'what now': 1, u'comedy central comedians': 1, u'funny video': 1, u'Ride along 2': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'real husbands of Hollywood': 1, u'comedy central politics': 1, u'comedy': 1, u'hilarious clips': 1, u'comedy videos': 1, u'hilarious videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Between the Scenes - Philando Castile & the Black Experience in America', u'behind the scenes', u'audience interaction', u'exclusives', u'police business', u'comedy central', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'comedian': 1, u'police business': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'audience interaction': 1, u'comedians': 1, u'comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'behind the scenes': 1, u'the daily show episodes': 1, u'Between the Scenes - Philando Castile & the Black Experience in America': 1, u'comedy central politics': 1, u'exclusives': 1, u'funny jokes': 1, u'hilarious clips': 1, u'comedy videos': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u"Between the Scenes - The White House's Messy Lie", u'bullshit Beetlejuice', u'Donald Trump', u'Between the Scenes', u'Kellyanne Conway', u'exclusives', u'late night talk show hosts', u'comedy central', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u"Between the Scenes - The White House's Messy Lie": 1, u'Kellyanne Conway': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Between the Scenes': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'stand up videos': 1, u'comedy': 1, u'bullshit Beetlejuice': 1, u'exclusives': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'funny jokes': 1, u'hilarious clips': 1, u'comedy videos': 1, u'hilarious videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u"Between the Scenes - Donald Trump: America's Penis-Shaped Asteroid", u'night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'comedian': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'stand up videos': 1, u'stand up comedy': 1, u'comedy': 1, u'hilarious videos': 1, u"Between the Scenes - Donald Trump: America's Penis-Shaped Asteroid": 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'comedy central politics': 1, u'funny jokes': 1, u'hilarious clips': 1, u'night talk show hosts': 1, u'comedy videos': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u"Between the Scenes - Trump's Dictator Tendencies", u'Between the Scenes', u'Donald Trump', u'South Africa', u'Jacob Zuma', u'James Comey', u'FBI', u'animals', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'comedian': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Between the Scenes': 1, u'funny': 1, u'the daily show': 1, u"Between the Scenes - Trump's Dictator Tendencies": 1, u'comedians': 1, u'stand up videos': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'Donald Trump': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'Jacob Zuma': 1, u'hilarious clips': 1, u'James Comey': 1, u'comedy central politics': 1, u'comedy central': 1, u'comedy': 1, u'animals': 1, u'South Africa': 1, u'comedy videos': 1, u'FBI': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Running Out of Spanish', u'Between the Scenes', u'exclusives', u'Spain', u'travel', u'giving directions', u'Spanish', u'no habla', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'Running Out of Spanish': 1, u'late night talk show hosts': 1, u'no habla': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Between the Scenes': 1, u'giving directions': 1, u'the daily show': 1, u'travel': 1, u'comedians': 1, u'stand up videos': 1, u'stand up comedy': 1, u'Spain': 1, u'funny': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'hilarious clips': 1, u'comedy videos': 1, u'hilarious videos': 1, u'Spanish': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u"Between the Scenes - Paul Ryan's", u'behind the scenes', u'exclusives', u'movies', u'guns', u'Affordable Care Act', u'health care', u'night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'health care': 1, u'comedian': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'Affordable Care Act': 1, u'stand up comedy': 1, u'guns': 1, u'hilarious videos': 1, u'stand up videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'behind the scenes': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'hilarious clips': 1, u'night talk show hosts': 1, u'movies': 1, u'comedy videos': 1, u'daily show with trevor noah': 1, u"Between the Scenes - Paul Ryan's": 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Between the Scenes - Thanks for the Civics Lesson', u'President Trump', u'between the scenes', u'behind the scenes', u'exclusives', u'education', u'laws', u'America', u'Barack Obama', u'government comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'comedian': 1, u'funny clips': 1, u'education': 1, u'between the scenes': 1, u'funny': 1, u'new trevor noah show': 1, u'the daily show': 1, u'comedians': 1, u'stand up comedy': 1, u'comedy': 1, u'hilarious videos': 1, u'President Trump': 1, u'comedy central comedians': 1, u'funny video': 1, u'behind the scenes': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'Barack Obama': 1, u'comedy central politics': 1, u'exclusives': 1, u'America': 1, u'hilarious clips': 1, u'Between the Scenes - Thanks for the Civics Lesson': 1, u'comedy videos': 1, u'government comedy central': 1, u'daily show with trevor noah': 1, u'laws': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u"Between the Scenes - Playa-Hatin' Republicans", u'behind the scenes', u'exclusives', u'audience interaction', u'health care', u'Republicans', u'Affordable Care Act', u'music', u'sexual advances comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'health care': 1, u'comedian': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'audience interaction': 1, u'Republicans': 1, u'comedians': 1, u'sexual advances comedy central': 1, u'music': 1, u'Affordable Care Act': 1, u"Between the Scenes - Playa-Hatin' Republicans": 1, u'comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'behind the scenes': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'exclusives': 1, u'stand up comedy': 1, u'hilarious clips': 1, u'comedy videos': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Between the Scenes - Health Care: Not Just for Healthy Young People Anymore', u'behind the scenes', u'exclusives', u'audience interaction', u'Republicans', u'health insurance', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'Between the Scenes - Health Care: Not Just for Healthy Young People Anymore': 1, u'health insurance': 1, u'comedian': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'audience interaction': 1, u'Republicans': 1, u'comedians': 1, u'stand up comedy': 1, u'comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'behind the scenes': 1, u'the daily show episodes': 1, u'comedy central politics': 1, u'exclusives': 1, u'funny jokes': 1, u'hilarious clips': 1, u'comedy videos': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'The Melania Trump Litmus Test', u'exclusives', u'Between the Scenes', u'Melania Trump', u'Donald Trump', u'Melania Trump eyes', u'Melania Trump reaction', u'late night talk show hosts', u'comedy central', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'The Melania Trump Litmus Test': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Between the Scenes': 1, u'funny': 1, u'Melania Trump': 1, u'the daily show': 1, u'comedians': 1, u'Melania Trump reaction': 1, u'comedy': 1, u'hilarious videos': 1, u'Melania Trump eyes': 1, u'exclusives': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'funny jokes': 1, u'hilarious clips': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'President Trump Dresses Up His Act', u'immigration policy', u'immigrant crime', u'speech to Congress', u'Between the Scenes', u'Donald Trump', u'immigration', u'late night talk show hosts', u'comedy central', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Between the Scenes': 1, u'funny': 1, u'speech to Congress': 1, u'the daily show': 1, u'comedians': 1, u'President Trump Dresses Up His Act': 1, u'comedy': 1, u'hilarious videos': 1, u'immigration': 1, u'funny video': 1, u'comedy central comedians': 1, u'comedy central': 1, u'the daily show episodes': 1, u'immigrant crime': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'funny jokes': 1, u'hilarious clips': 1, u'immigration policy': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'TK', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'TK': 1, u'stand up videos': 1, u'stand up comedy': 1, u'comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'comedy central politics': 1, u'funny jokes': 1, u'hilarious clips': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Donald Trump', u'all show', u'Carrier deal', u'jobs numbers', u'Obama', u'unemployment', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes']
    {u'trevor noah': 1, u'unemployment': 1, u'late night talk show hosts': 1, u'new trevor noah show': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'stand up videos': 1, u'stand up comedy': 1, u'Obama': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'Carrier deal': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'comedy': 1, u'all show': 1, u'comedy videos': 1, u'comedian': 1, u'jobs numbers': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Kazakhstan', u'intelligence briefings', u'exclusives', u'Between the Scenes', u'Donald Trump', u'Mike Pence', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'intelligence briefings': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Between the Scenes': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'stand up videos': 1, u'stand up comedy': 1, u'Kazakhstan': 1, u'hilarious videos': 1, u'comedy central': 1, u'Mike Pence': 1, u'comedy central comedians': 1, u'funny video': 1, u'Donald Trump': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'hilarious clips': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Donald trump', u'president-elect', u'Donald trump intelligence', u'between the scenes', u'behind the scenes', u'intelligence briefing', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'intelligence briefing': 1, u'funny': 1, u'the daily show': 1, u'Donald trump intelligence': 1, u'comedians': 1, u'Donald trump': 1, u'between the scenes': 1, u'stand up videos': 1, u'hilarious': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'behind the scenes': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'comedy': 1, u'comedy videos': 1, u'comedian': 1, u'president-elect': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Obamacare', u'hospital costs', u'appendectomy', u'exclusives', u'behind the scenes', u'Between the Scenes', u'health insurance', u'Affordable Care Act', u'late night talk show hosts', u'comedy central', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'Obamacare': 1, u'health insurance': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Between the Scenes': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'hospital costs': 1, u'Affordable Care Act': 1, u'comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'behind the scenes': 1, u'the daily show episodes': 1, u'comedy central politics': 1, u'exclusives': 1, u'funny jokes': 1, u'hilarious clips': 1, u'comedy videos': 1, u'comedian': 1, u'appendectomy': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Obamacare name', u'Obamacare poll', u'exclusives', u'behind the scenes', u'Between the Scenes', u'health insurance', u'Affordable Care Act', u'polls', u'Barack Obama', u'late night talk show hosts', u'comedy central', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'health insurance': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Between the Scenes': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'Affordable Care Act': 1, u'comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'Obamacare poll': 1, u'comedy central': 1, u'behind the scenes': 1, u'the daily show episodes': 1, u'Barack Obama': 1, u'funny video': 1, u'comedy central politics': 1, u'exclusives': 1, u'funny jokes': 1, u'hilarious clips': 1, u'polls': 1, u'comedy videos': 1, u'Obamacare name': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'TK', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'comedy central politics': 1, u'trevor noah': 1, u'stand up comedy': 1, u'hilarious clips': 1, u'the daily show': 1, u'funny video': 1, u'comedy videos': 1, u'comedy central comedians': 1, u'comedians': 1, u'comedy central': 1, u'funny': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedian': 1, u'TK': 1, u'stand up videos': 1, u'funny clips': 1, u'new trevor noah show': 1, u'comedy': 1, u'hilarious videos': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Between the Scenes', u'behind the scenes', u'exclusives', u'audience interactions', u'Hillary Clinton', u'campaigns', u'candidates', u'elections', u'insults', u'Donald Trump', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos']
    {u'trevor noah': 1, u'elections': 1, u'late night talk show hosts': 1, u'new trevor noah show': 1, u'Donald Trump': 1, u'Between the Scenes': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'candidates': 1, u'stand up comedy': 1, u'exclusives': 1, u'comedy central comedians': 1, u'funny video': 1, u'insults': 1, u'comedy central': 1, u'behind the scenes': 1, u'the daily show episodes': 1, u'audience interactions': 1, u'comedy central politics': 1, u'campaigns': 1, u'comedy': 1, u'Hillary Clinton': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Between the Scenes', u'behind the scenes', u'Africa', u'South Africa', u'growing up', u'movies', u'food', u'parents', u'watch movies', u'food fights', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Between the Scenes': 1, u'funny': 1, u'food fights': 1, u'the daily show': 1, u'comedians': 1, u'parents': 1, u'stand up videos': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'food': 1, u'watch movies': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'behind the scenes': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'growing up': 1, u'comedy central politics': 1, u'comedy': 1, u'hilarious clips': 1, u'South Africa': 1, u'Africa': 1, u'movies': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Between the Scenes', u'exclusives', u'behind the scenes', u'Olympics', u'sports', u'gymnastics', u'awards', u'trevor watching Olympics', u'rio games', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Between the Scenes': 1, u'funny': 1, u'gymnastics': 1, u'the daily show': 1, u'comedians': 1, u'sports': 1, u'stand up videos': 1, u'comedian': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'behind the scenes': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'awards': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'hilarious clips': 1, u'trevor watching Olympics': 1, u'rio games': 1, u'comedy videos': 1, u'Olympics': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'exclusives', u'behind the scenes', u'Donald Trump', u'racism', u'organized crime', u'Between the Scenes', u'birth certificate', u'Bernie Sanders', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'birth certificate': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Between the Scenes': 1, u'organized crime': 1, u'funny': 1, u'the daily show': 1, u'racism': 1, u'comedians': 1, u'stand up videos': 1, u'Bernie Sanders': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'exclusives': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'behind the scenes': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'comedy': 1, u'hilarious clips': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'comedy central politics', u'the daily show episodes', u'Between the Scenes', u'audience', u'audience interaction', u'Lindsey Graham', u'Donald Trump', u'Republicans', u'discrimination', u'conservative', u'gambling', u'Democrats', u'Ted Cruz', u'Hillary Clinton', u'wars', u'terrorism', u'international affairs', u'Middle East', u'Iraq War', u'military', u'extremism', u'ISIS', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedy', u'funny', u'comedy videos', u'stand up videos', u'funny jokes']
    {u'trevor noah': 1, u'extremism': 1, u'international affairs': 1, u'late night talk show hosts': 1, u'Middle East': 1, u'Iraq War': 1, u'Between the Scenes': 1, u'Ted Cruz': 1, u'funny': 1, u'ISIS': 1, u'the daily show': 1, u'discrimination': 1, u'audience interaction': 1, u'Republicans': 1, u'comedy videos': 1, u'conservative': 1, u'stand up videos': 1, u'gambling': 1, u'terrorism': 1, u'stand up comedy': 1, u'wars': 1, u'Donald Trump': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'comedy central': 1, u'comedy': 1, u'Lindsey Graham': 1, u'Hillary Clinton': 1, u'audience': 1, u'Democrats': 1, u'military': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'island accent', u'accents', u'Trinidadian Liam Neeson', u'exclusives', u'Between the Scenes', u'behind the scenes', u'audience interaction', u'impressions', u'Trinidad and Tobago', u'Liam Neeson', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video']
    {u'trevor noah': 1, u'accents': 1, u'late night talk show hosts': 1, u'impressions': 1, u'Liam Neeson': 1, u'new trevor noah show': 1, u'Between the Scenes': 1, u'funny': 1, u'the daily show': 1, u'audience interaction': 1, u'comedians': 1, u'Trinidadian Liam Neeson': 1, u'stand up comedy': 1, u'island accent': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'Trinidad and Tobago': 1, u'the daily show episodes': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'behind the scenes': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'civil rights and basketball', u'Between the Scenes', u'exclusives', u'audience interaction', u'NAACP', u'NCAA', u'basketball', u'sports', u'African American', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'new trevor noah show': 1, u'Between the Scenes': 1, u'funny': 1, u'basketball': 1, u'the daily show': 1, u'audience interaction': 1, u'NCAA': 1, u'comedians': 1, u'sports': 1, u'stand up videos': 1, u'stand up comedy': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'civil rights and basketball': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'African American': 1, u'NAACP': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Africa', u'dictators', u'behaving badly', u'impressions', u'murder', u'Between the Scenes', u'audience interaction', u'exclusives', u'rude dictators', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'impressions': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Between the Scenes': 1, u'funny': 1, u'behaving badly': 1, u'the daily show': 1, u'audience interaction': 1, u'comedians': 1, u'stand up videos': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'murder': 1, u'comedy central comedians': 1, u'dictators': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'funny video': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'hilarious clips': 1, u'Africa': 1, u'rude dictators': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'exclusives', u'behind the scenes', u'Desi Lydic', u'pregnancy', u'audience interaction', u'Between the Scenes', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'pregnancy': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Between the Scenes': 1, u'funny': 1, u'the daily show': 1, u'audience interaction': 1, u'comedians': 1, u'stand up videos': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'Desi Lydic': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'behind the scenes': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'hilarious clips': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'rapping Ben Carson', u'Ben Carson', u'exclusives', u'Between the Scenes', u'Desi Lydic', u'competitions', u'doctors', u'Bush impression', u'impressions', u'Wu-Tang Clan', u'music', u'hip hop', u'late night talk show hosts', u'comedy central', u'comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'impressions': 1, u'new trevor noah show': 1, u'Bush impression': 1, u'funny': 1, u'the daily show': 1, u'rapping Ben Carson': 1, u'comedians': 1, u'competitions': 1, u'music': 1, u'Between the Scenes': 1, u'stand up videos': 1, u'Wu-Tang Clan': 1, u'hilarious videos': 1, u'Desi Lydic': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'doctors': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'hilarious clips': 1, u'comedy videos': 1, u'hip hop': 1, u'comedian': 1, u'Ben Carson': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Dexter knock knock joke', u'Donald Trump', u'Ted Cruz', u'elections', u'parents', u'Between the Scenes', u'audience interaction', u'exclusives', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Between the Scenes': 1, u'Ted Cruz': 1, u'funny': 1, u'the daily show': 1, u'audience interaction': 1, u'comedians': 1, u'parents': 1, u'stand up videos': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'exclusives': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'Dexter knock knock joke': 1, u'comedy central politics': 1, u'Donald Trump': 1, u'comedy': 1, u'hilarious clips': 1, u'elections': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'exclusives', u'behind the scenes', u'audience interaction', u'Between the Scenes', u'American applause', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'American applause': 1, u'funny': 1, u'the daily show': 1, u'audience interaction': 1, u'comedians': 1, u'Between the Scenes': 1, u'stand up videos': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'behind the scenes': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'hilarious clips': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Between the Scenes', u'exclusives', u'Donald Trump', u'elections', u'candidates', u'Africa', u'music', u'songs', u'rants', u'Twitter', u'science', u'Neil deGrasse Tyson', u'conspiracies', u'white people', u'African American', u'lying', u'rage', u'fights', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'hilarious clips']
    {u'trevor noah': 1, u'rants': 1, u'fights': 1, u'late night talk show hosts': 1, u'Neil deGrasse Tyson': 1, u'new trevor noah show': 1, u'Between the Scenes': 1, u'funny': 1, u'the daily show': 1, u'comedians': 1, u'candidates': 1, u'music': 1, u'stand up comedy': 1, u'comedy central': 1, u'conspiracies': 1, u'comedy central comedians': 1, u'funny video': 1, u'Donald Trump': 1, u'the daily show episodes': 1, u'African American': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'hilarious clips': 1, u'rage': 1, u'white people': 1, u'science': 1, u'Twitter': 1, u'Africa': 1, u'elections': 1, u'comedian': 1, u'daily show with trevor noah': 1, u'lying': 1, u'songs': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'Ronny Chieng', u'China', u'exclusives', u'Between the Scenes', u'Asian American', u'audience interaction', u'fake accent', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'new trevor noah show': 1, u'Between the Scenes': 1, u'funny': 1, u'Asian American': 1, u'the daily show': 1, u'audience interaction': 1, u'comedians': 1, u'fake accent': 1, u'China': 1, u'stand up videos': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'Ronny Chieng': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'hilarious clips': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    
    [u'the daily show', u'trevor noah', u'daily show with trevor noah', u'new trevor noah show', u'comedy central politics', u'the daily show episodes', u'animals', u'ISIS', u'exclusives', u'Between the Scenes', u'terrorism', u'murder', u'audience interaction', u'killer pandas', u'too soon', u'late night talk show hosts', u'comedy central', u'stand up comedy', u'comedians', u'comedy central comedians', u'comedy', u'funny', u'comedian', u'funny video', u'comedy videos', u'stand up videos', u'funny jokes', u'funny clips', u'hilarious videos', u'hilarious clips']
    {u'trevor noah': 1, u'late night talk show hosts': 1, u'funny clips': 1, u'too soon': 1, u'Between the Scenes': 1, u'funny': 1, u'new trevor noah show': 1, u'ISIS': 1, u'the daily show': 1, u'audience interaction': 1, u'comedians': 1, u'killer pandas': 1, u'stand up videos': 1, u'terrorism': 1, u'stand up comedy': 1, u'hilarious videos': 1, u'murder': 1, u'comedy central comedians': 1, u'funny video': 1, u'comedy central': 1, u'the daily show episodes': 1, u'funny jokes': 1, u'hilarious clips': 1, u'comedy central politics': 1, u'exclusives': 1, u'comedy': 1, u'animals': 1, u'comedy videos': 1, u'comedian': 1, u'daily show with trevor noah': 1}
    



{% highlight python %}
pd.Series(x['items'][0]['statistics'])
{% endhighlight %}




    commentCount        1545
    dislikeCount         822
    favoriteCount          0
    likeCount          22269
    viewCount        1321050
    dtype: object




{% highlight python %}
pd.Series(x['items'][0]['snippet'])
{% endhighlight %}




    categoryId                                                             23
    channelId                                        UCwWhs_6x42TyRM4Wstoq8HA
    channelTitle                              The Daily Show with Trevor Noah
    defaultAudioLanguage                                                   en
    description             Hasan Minhaj gives his shell-shocked take on D...
    liveBroadcastContent                                                 none
    localized               {u'description': u'Hasan Minhaj gives his shel...
    publishedAt                                      2017-04-14T03:30:00.000Z
    tags                    [Hasan Minhaj, elections, Donald Trump, candid...
    thumbnails              {u'default': {u'url': u'https://i.ytimg.com/vi...
    title                   Why Wasn't Donald Trump's Bigotry a Deal-Break...
    dtype: object




{% highlight python %}
pd.Series(x['items'][0]['contentDetails'])
{% endhighlight %}




    caption                                             true
    definition                                            hd
    dimension                                             2d
    duration                                         PT4M11S
    licensedContent                                     True
    projection                                   rectangular
    regionRestriction    {u'blocked': [u'GB', u'AU', u'CA']}
    dtype: object




{% highlight python %}
df = pd.read_csv("trevor_noah_daily_show_comments.csv",encoding='utf-8',dtype=str)
{% endhighlight %}


{% highlight python %}

for videoId in df['videoId'].unique():
    corpus = df[df['videoId']==videoId]['textDisplay'].sum().replace("39","'")
    x = remove_spaces(remove_rn(strip_punctuation(corpus)).replace(","," "))
    fdist1= nltk.FreqDist(one_gram(x))
    print(dict(fdist1.most_common(100)))
    break
{% endhighlight %}

    {'thats': 45, 'something': 24, 'come': 19, 'need': 24, 'fucking': 23, 'world': 19, 'better': 23, 'ban': 33, 'ideology': 19, 'think': 46, 'since': 20, 'minhaj': 21, 'america': 46, 'guy': 21, 'actually': 20, 'every': 22, 'first': 25, 'trevor': 21, 'cant': 38, 'wasnt': 20, 'deal': 27, 'vote': 24, 'see': 34, 'funny': 18, 'stupid': 20, 'said': 22, 'really': 36, 'bigotry': 39, 'feel': 24, 'one': 54, 'mean': 20, 'even': 32, 'trump': 149, 'isnt': 46, 'made': 24, 'know': 33, 'country': 36, 'fuck': 28, 'didnt': 24, 'show': 47, 'never': 32, 'right': 45, 'way': 19, 'saying': 20, 'still': 46, 'god': 18, 'talk': 19, 'make': 23, 'people': 179, 'trumps': 26, 'much': 31, 'get': 44, 'believe': 27, 'bad': 18, 'good': 34, 'things': 24, 'muslim': 71, 'many': 34, 'racist': 71, 'thing': 21, 'real': 19, 'want': 44, 'black': 30, 'white': 73, 'voted': 33, 'well': 18, 'done': 18, 'could': 24, 'hes': 30, 'dont': 95, 'doesnt': 40, 'also': 32, 'race': 69, 'daily': 24, 'islam': 77, 'shit': 24, 'love': 30, 'someone': 20, 'back': 25, 'thought': 26, 'racism': 56, 'hasan': 42, 'would': 60, 'like': 98, 'word': 18, 'person': 21, 'man': 21, 'take': 24, 'anyone': 24, 'youre': 26, 'president': 32, 'say': 24, 'fact': 22, 'care': 30, 'religion': 52, 'hate': 31, 'muslims': 65, 'hillary': 26, 'going': 23, 'middle': 20}



