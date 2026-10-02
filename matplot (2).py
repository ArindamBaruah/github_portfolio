

#understanding the matplot

import matplotlib.pyplot as pt
pt.plot([1,2,3,4],[3,8,10,25 ])
pt.show()


'''
import matplotlib.pyplot as pt
pt.plot([1,2,3,4],[3,8,10,25 ])
pt.title('rain in december')
pt.xlabel('days in december')
pt.ylabel('inches in rain')
pt.show()

'''

#change the line with the dot or any other symbol
'''
import matplotlib.pyplot as plt
#plt.plot([1,2,3,4],[1,4,9,16])
#plt.plot([1,2,3,4],[1,4,9,16],'ro')
#plt.plot([1,2,3,4],[1,4,9,16],'r--')
#plt.plot([1,2,3,4],[1,4,9,16],'bs')
plt.plot([1,2,3,4],[1,4,9,16],'g^')
plt.ylabel('y- AXIS')
plt.show()
'''


#--------------------------------------------------
#understanding subplots
'''
import matplotlib.pyplot as pt

fig = pt.figure()
rect = fig.patch
rect.set_facecolor('green')
x=[3,7,8,12]
y=[5,13,2,8]
graph1=fig.add_subplot(1,1,1, facecolor='black')
graph1.plot(x,y,'red',linewidth=4.0)
pt.show()
'''

#coloring of graph


import matplotlib.pyplot as pt

fig = pt.figure()
    
rect = fig.patch
rect.set_facecolor('green')

x=[3,7,8,12]
y=[5,13,2,8]
graph1=fig.add_subplot(1,1,1, axisbg='black')
graph1.plot(x,y,'red',linewidth=4.0)

graph1.tick_params(axis='x', color='white')
graph1.tick_params(axis='y', color='white')

graph1.spines['top'].set_color('w')
graph1.spines['left'].set_color('w')
graph1.spines['right'].set_color('w')
graph1.spines['bottom'].set_color('w')

graph1.set_title('random graph', color='white')
graph1.set_xlabel('this is the x axis', color='white')
graph1.set_ylabel('this is the y axis')

pt.show()


#add multiple lines and multiple graph
#draw multiple graph using .plot
'''
import matplotlib.pyplot as pt

fig = pt.figure()

rect = fig.patch
rect.set_facecolor('green')

#x=[3,7,8,12]
#y=[5,13,2,8]

#x2=[0,4,7,10]
#y2=[3,7,1,12]

#x3=[2,4,6,8]
#y3=[13,5,8,2]

x=[0,7,8,12]
y=[5,13,2,8]

x2=[0,4,7,12]
y2=[3,7,1,12]

x3=[0,4,6,12]
y3=[13,5,8,2]

graph1=fig.add_subplot(1,1,1, axisbg='black')
graph1.plot(x,y,'red',linewidth=4.0)

graph1.plot(x2,y2,'yellow',linewidth=2.0)

graph1.plot(x3,y3,'orange',linewidth=6.0)

graph1.tick_params(axis='x', color='white')
graph1.tick_params(axis='y', color='white')

graph1.spines['top'].set_color('w')
graph1.spines['left'].set_color('w')
graph1.spines['right'].set_color('w')
graph1.spines['bottom'].set_color('w')

graph1.set_title('random graph', color='white')
graph1.set_xlabel('this is the x axis', color='white')
graph1.set_ylabel('this is the y axis')

pt.show()
'''


#multiple graph in a figure

'''
import matplotlib.pyplot as pt

fig = pt.figure()

rect = fig.patch
rect.set_facecolor('green')

x=[0,7,8,12]
y=[5,13,2,8]

x2=[0,4,7,12]
y2=[3,7,1,12]

x3=[0,4,6,12]
y3=[13,5,8,2]

fig.add_subplot(3,1,1)
pt.plot(x,y,'r--')
fig.add_subplot(3,1,2)
pt.plot(x2,y2,'r--')
fig.add_subplot(3,1,3)
pt.plot(x3,y3,'r--')

pt.show()
'''
'''
import matplotlib.pyplot as pt


fig = pt.figure()

rect = fig.patch
rect.set_facecolor('green')

x=[0,7,8,12]
y=[5,13,2,8]

x2=[0,4,7,12]
y2=[3,7,1,12]

x3=[0,4,6,12]
y3=[13,5,8,2]

fig.add_subplot(3,1,1)
pt.plot(x,y,'r--')
fig.add_subplot(3,1,2)
pt.plot(x2,y2,'r--')
fig.add_subplot(3,1,3)
pt.plot(x3,y3,'r--')

pt.show()
'''


#bar graph
'''
import matplotlib.pyplot as pt
import numpy as np

pos = np.arange(6) + 0.5

#arrange function create an array
#6 bars and 0.5 is the difference b/w bar graph

pt.barh(pos,(4,8,12,3, 17,6), align = 'center', color='red')
pt.show()
'''
 
 #modify bar graph
'''
import matplotlib.pyplot as pt
import numpy as np

pos = np.arange(6) + 0.5

#arrange function create an array
#6 bars and 0.5 is the difference b/w bar graph

names=['ram', 'sham', 'tom', 'dom', 'jeery', 'merry' ]

pt.barh(pos,(4,8,12,3, 17,6), align = 'center', color='red')
#help(pt)
pt.xlabel('height in inches', color='red')
pt.ylabel('student', color='red')
pt.title('height of student in inches', color='blue')

pt.tick_params(axis='x', color='white')
pt.tick_params(axis='y', color='white')

pt.yticks(pos,names)

pt.show()



#pie chart


import matplotlib.pyplot as pt
sizes=[50,23,7,15,5]

pt.pie(sizes)
pt.show()
'''

#modify 
'''
import matplotlib.pyplot as pt
sizes=[50,23,17,5,5]

colors=['yellow','orange','cyan','magenta','red']

#pt.pie(sizes,colors=colors)
pt.pie(sizes,colors=colors, startangle=90)

pt.axis('equal')
pt.show()
'''
#labels

'''
import matplotlib.pyplot as pt
sizes=[50,23,17,5,5]
labels='android', 'apple','window','blackberry','Xiaomi'
colors=['yellow','orange','cyan','magenta','red']

#pt.pie(sizes,colors=colors)
pt.pie(sizes,labels=labels,colors=colors, startangle=90)
pt.title('pie chart')
#pt.legend(title='legend',loc='lower left')
pt.axis('equal')
pt.legend(loc='lower left')
pt.show()
'''

#scatter plot
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pt
import numpy as np
fig=pt.figure()
ax1 = fig.add_subplot(111, projection='3d')
xpose=[1,2,3,4,5,6,7,8,9,10]
ypose=[2,3,4,5,1,6,2,1,7,2]
zpose=[5,8,7,3,2,1,5,1,2,3]
dx = np.ones(10)
dy=np.ones(10)
dz = [1,2,3,4,5,6,7,8,9,10]
ax1.scatter(xpose, ypose, zpose, color='red', marker='o')
pt.show()

'''

