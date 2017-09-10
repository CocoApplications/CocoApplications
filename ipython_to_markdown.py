import sys

script = open(sys.argv[1]).read()

script = script.replace(".png)",".png' /></p>").replace("![png]","<p><img src='").replace("```python","{% highlight python %}").replace("```","{% endhighlight %}")

script = script.replace("(output_","/signal_images/output_")

print(script)