#Neural Network Boundary Plot Example 
#Adapted From: https://stats.stackexchange.com/questions/253217/neural-network-decision-boundary/254325

#Note:
#Each neuron splits the input into one of 2 classes (see: http://hagan.okstate.edu/NNDesign.pdf, Chapter 11)
#The first neuron splits the upper left blue input from the rest
#The second neuron splits the lower right blue input from the rest
#The output neuron splits the result into red area or blue area

import matplotlib.pyplot as plt
import numpy as np

boundary1 = np.linspace(-.2,1.2,100)
plt.plot(boundary1,boundary1+0.4,c='black')
plt.plot(boundary1,boundary1-0.6,c='black')
plt.scatter(1,1,c='red',s=100)
plt.scatter(0,0,c='red',s=100)
plt.scatter(0,1,s=100)
plt.scatter(1,0,s=100)
plt.fill_between(x=boundary1,y1=boundary1+.4,y2=boundary1+3,alpha=.2,color='blue')
plt.fill_between(x=boundary1,y1=boundary1-.6,y2=boundary1-2,alpha=.2,color='blue')
plt.fill_between(x=boundary1,y1=boundary1+.4,y2=boundary1-0.6,alpha=.2,color='red')
plt.xlim(-.2,1.2)
plt.ylim(-.1,1.1)
plt.show()