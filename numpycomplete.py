# Numpy File
#1D
import numpy as np
a=np.array([[1,2,3]])
print(a)
print(type(a))

#2D
import numpy as np
a=np.array([[1,2],[3,4]])
print(a)

# number of elements: size

import numpy as np
a=np.array([[1,2],[30,4],[15,10]])
print(a)
print(type(a))
print(a.size)

# number of rows and column: shape

import numpy as np
a=np.array([[1,2,30,4],[1,5,10,15]])
print(a)
print(type(a))
print(a.size)
print(a.shape)

#change in number of rows and column: reshape
import numpy as np
a=np.array([[1,2,3],[4,5,6]])
print(a)
print(a.shape)
x=a.reshape(3,3)
print(x)
print(type(x))
print(x.size)
print(x.shape)

# resize
import numpy as np
a=np.array([[1,2],[3,4]])
b=np.resize(a,(3,4))
print(b)
print(type(b))
print(b.size)

# arange
import numpy as np
x=np.arange(10)
print(x)
print(type(x))
print(x.size)

#arange with reshape 
import numpy as np
x=np.arange(10)
print(x)
print(type(x))
print(x.size)
b=x.reshape(2,5)
print(b)
print(type(b))

#arange with resize
import numpy as np
x=np.arange(10)
print(x)
print(x.shape)
print(type(x))
print(x.size)
print("===================")
b=np.resize(x, (3,4))
print(b)
print(type(b))
print(x.shape)

# resize : n no. of matrix 
import numpy as np
x=np.arange(20)
print(x)
print(type(x))
print(x.size)
print("===================")
b=np.resize(x,(3,3,4)) # two no. of matrix, 3 rows, 4 columns
print(b)
print(type(b))
print(b.shape)
print(b.size)

# ndim : retun number for N dimensional array 

import numpy as np
x=np.arange(24)
print(x)
print(type(x))
print(x.ndim)

b=x.reshape(2,3,2,2)
print(b)
print(b.ndim)

# empty
import numpy as np
x=np.empty([3,2], dtype=int)
print(x)
print(type(x))


# numpy.zeros
import numpy as np
x=np.zeros(7, dtype=np.int)
print(x)
print(type(x))

# numpy.ones
import numpy as np
x=np.ones([5,2], dtype=np.int)
print(x)
print(type(x))

# itemsize :length of each element
import numpy as np
x1=np.array([1,5,7], dtype=np.int8)
x2=np.array([10,50,70], dtype=np.int16)
x3=np.array([11,105,17], dtype=np.int32)
x4=np.array([15,55,74], dtype=np.int64)
x5=np.array([100,585,75], dtype=np.float32)
print("=======INT8=============")
print(x1.itemsize)

print("=======INT16=============")
print(x2.itemsize)

print("=======INT32=============")
print(x3.itemsize)

print("=======INT64=============")
print(x4.itemsize)

print("=======Float32=============")
print(x5.itemsize)

# program to convert list into array4
import numpy as np
x=[1,2,3,4]
print(x)
print(type(x))
a=np.asarray(x)
print(a)
print(type(a))

#asarray ( tuple to array )
import numpy as np
x=(10,20,40)
print(x)
print(type(x))
a=np.asarray(x)
print(a)
print(type(a))

# linspace
import numpy as np
x=np.linspace(0,10,3)
print(x)

#arithmetic operations
# axis=0 for column
# axis =1 for row
import numpy as np
a=[[1,2],[10,20]]
print(a)
print(np.sum(a))
print(np.sum(a, axis=1))

# 0 1 2      10 11 12      10+0   11+1    12+2
# 3 4 5      00 00 00      10+3  20+4     30+5 

#arithmetic
# axis =0 /1 0 for column ,1 for row
import numpy as np
a=[[0,1],[2,3]]
print(np.sum(a))
print(np.sum(a, axis=0))
print(np.sum(a, axis=1))

#matrix operation (add, subtract, multiply,divide,dot)
import numpy as np
a=np.array([[1,2],[3,4]])
b=np.array([[5,10],[15,20]])
#print(np.add(a,b))
#print(np.subtract(a,b))
#print(np.multiply(a,b))
#print(np.divide(b,a))
#print(np.dot([1,2],[5,3]))

#amin,amax
import numpy as np
a=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a)
print(np.amin(a)) #1
print(np.amin(a, axis=0)) #[1 2 3] 
print(np.amin(a, axis=1)) #[1 4 7]
print(np.amax(a)) #9
print(np.amax(a, axis=0)) #[7 8 9]
print(np.amax(a, axis=1)) #[3 6 9]

# ptp (maximum -minimum)
import numpy as np
a=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a)
print(np.mean(a))
print(np.mean(a, axis=0))  
print(np.mean(a, axis=1))

print(np.median(a))
print(np.median(a, axis=0))  
print(np.median(a, axis=1))

print(np.ptp(a))
print(np.ptp(a, axis=0))  
print(np.ptp(a, axis=1))

import numpy as np
a=np.array([1,2,3,4])
print(np.average(a))

w=np.array([4,3,2,1])
print(np.average(a, weights=w))

import numpy as np
a=np.array([1,2,3,4])
print(np.std(a))
print(np.var(a))

# broadcasting
import numpy as np
a=np.array([[1,2],[3,4]])
b=np.array([5,10])
#print(np.add(a,b))















































































































































































