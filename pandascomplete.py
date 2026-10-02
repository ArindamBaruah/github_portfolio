'''
# one dimensional using Series
import pandas as p
import numpy as n
data= n.array([1,2,3,4])
s=p.Series(data,index=['a','b','c','d'])
print(s)

d= n.array(["Ram", "Sham",20])
s1=p.Series(d,index=[1000,1001,1002])
print(s1)

z=p.Series([10,"lpu"])
print(z)

import pandas as p
d={'jalandhar':800 ,'Delhi':500 ,'Amritsar':900}
cities=p.Series(d)
cities['delhi']=250
print(cities)

# 2D using DataFrame
import pandas as p
data=[['jatin', 10],['Rupinder',95],['Sreehari',93],['Somesh',97]]
d=p.DataFrame(data, columns=['Name', 'Attendance'], index=['J','R','S','S'])
print(d)

# missing : NaN
import pandas as p
data=[['jatin', 10],['Rupinder'],['Sreehari',93],['Somesh',97]]
d=p.DataFrame(data, columns=['Name', 'Attendance'], index=['J','R','S','S'])
print(d)

# dataframe from dict of series (Key will be column name)
import pandas as p
d={'First':p.Series([1,2,3],index=['a','b','c']) ,
   'Second':p.Series(["Ram","Sham","Lpu"]) }
df=p.DataFrame(d)
print(df)

# column Selection
import pandas as p
d={'First':p.Series([1, 'Ram', 'Punjab']),
   'Second':p.Series([11,'Sham', 'Delhi'])}
s=p.DataFrame(d)
print(s['First'])

# column addition
import pandas as p
data={'Student':['Lpu','Jal','Punjab','Good'],
      'Maths':[80,70,50,90],
      'Computer':[90,60,40,80]}
s=p.DataFrame(data)
print(s)
# add new column from existing columns
print("===================")
s['Total']=s['Maths'] + s['Computer']
print(s)

#Column Addition :add new column by passing as Series
import pandas as p
d={'First':p.Series([1, 'Ram', 'Punjab']),
   'Second':p.Series([11,'Sham', 'Delhi'])}
s=p.DataFrame(d)
print(s)
print("===================")
s['Third']=p.Series([100,200,300])
print(s)

# column Deletion (del, pop )
import pandas as p
data={'Student':['numpy','pandas','matplotlib'],
      'Maths':[40,80,70],
      'English':[60,70,75],
      'Computer':[60,90,95]}
s=p.DataFrame(data)
print(s)
s['Total']=s['Maths']+s['English']+s['Computer']
print(s)
#using del
#del s['Maths']
# using pop
s.pop('Maths')
print(s)

# Row selection (loc and iloc function)
# row can be selected by passing row label to loc function
import pandas as p
d={'one':p.Series([1,2,3],index=['a','b','c']),
   'two':p.Series([4,5,6],index=['a','b','c'])}
df=p.DataFrame(d)
print(df)
print("====================")
print(df.loc['b'])

# row can be selected by passing the integer location to iloc function
import pandas as p
d={'one':p.Series([1,2,3],index=['a','b','c']),
   'two':p.Series([4,5,6],index=['a','b','c'])}
df=p.DataFrame(d)
print(df)
print("====================")
print(df.iloc[2])

# Multiple rows can be selected using : operator
import pandas as p
d={'one':p.Series([1,2,3,10,20,30],index=['a','b','c','d','f','g']),
   'two':p.Series([4,5,6,21,31,71],index=['a','b','c','d','f','g'])}
df=p.DataFrame(d)
print(df)
print("====================")
print(df[1:6:2])

# append
import pandas as p
d1={'one':p.Series([1,2,3,10,20,30],index=['a','b','c','d','f','g'])}
d2={'one':p.Series([4,5,6,21,31,71],index=['a','b','c','d','f','g'])}
df1=p.DataFrame(d1)
df2=p.DataFrame(d2)
df1=df2.append(df1)
print(df1)

#Row deletion using drop (label value)
import pandas as p
df=p.DataFrame([[1,2],[3,4]], columns=['a','b'])
df2=p.DataFrame([[5,6],[7,8]], columns=['a','b'])
df=df2.append(df)
print(df)
# drop row with label
df=df.drop(0)
print(df)

#Row deletion using drop (label name)
import pandas as p
df=p.DataFrame([[1,2],[3,4]], columns=['a','b'], index=['d1','d2'])
df2=p.DataFrame([[5,6],[7,8]], columns=['a','b'], index=['d3','d4'])
df=df2.append(df)
print(df)
# drop row with label
df=df.drop('d3')
print(df)

import pandas as p
d={'A':100, 'B':200, 'C':300, 'D':250, 'E':500, 'G':280, 'H':800}
count=p.Series(d)
print(count)
print("=======================")
count[count<300]=10
print(count)
#print("=======================")
#print(count[count<300])
#print(count['B'])
print('D' in count)

# read csv filr
import pandas as p
x=p.read_csv("G:/rh.csv")
print(x)

# header names 
import pandas as p
x=p.read_csv("G:/rh.csv", names=['Roll','Marks','Name'], header=0)
print(x)

#usecols
import pandas as p
x=p.read_csv("G:/rh.csv",names=['Roll','Marks','Name'],header=0, usecols=[0,2])
print(x)

# write data to csv file

import pandas as p
data=[['Ram',11,'Punjab'],['Sham',12,'Delhi'],
      ['John',13,'Jal'],['Rohit',14,'Amritsar']]
df=p.DataFrame(data, columns=['Name','Roll', 'Address'])
print(df)
df.to_csv("G:/rh.csv")

# skiprows
import pandas as p
x=p.read_csv("G:/rh.csv",names=['Student Name','Student Roll','Student Add'], skiprows=1)
print(x)
'''
# g=F, from first 10 record
import pandas as p
users=p.read_csv("G:/ml-100k/u.user", sep='|',
                 names=['UserId','Age','Gender','Occupation','EmpCode'])
#data=users['Gender'].head(10)
# multiple rows
#x=['Gender','Occupation']
#data=users[x].head(10)
#data=users[100:150:5]
data=users[(users.Age<30) & (users.Gender=='F')].head(10)

print(data)




























































































































































      
      










































































'''
POLL 1
import numpy as np
x=np.array([[1,2,3],[100,200,300]])
print(x.itemsize)

A. 8
B. 6
C. 4
D. 1

'''

'''
POLL 2
import numpy as np
x=np.array([[1,2,3],[100,200,300],[10,20,30]])
print(np.ptp(x, axis=0))

A. [99, 198, 297]
B. [99 198 297]
C. [[99 198 297]]
'''



















