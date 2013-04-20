from __future__ import print_function
import sys
import pprint
import numpy as np

import xdtest
print(xdtest.__file__)

import xdtest.xdstlc as x
print(x.__doc__)
pprint.pprint(dir(x))

#print(np.array([1.0, 42.0], dtype=x.xd_double)
#print(np.array([1.0, 42.0], dtype=x.xd_double)[0]
#a = np.array([1.0, 42.0], dtype=x.xd_double)
#print(str(a)

print(np.array(["hello", "world"], dtype=x.xd_str))
print(np.array(["hello", "world"], dtype=x.xd_str)[0])
#a = np.array(["hello"], dtype=x.xd_str)
#a = np.array(["hello", "world"], dtype=x.xd_str)
#a = np.array(["hello", "momma", 'love', "me"], dtype=x.xd_str)
b = ["hello", "momma", 'love', "me"]
a = np.array(b, dtype=x.xd_str)
#print("allocated"
print(a)
#print(a[1:3]
v = a[0]
print("HERE ---->", v, type(v))
#del a
#a = np.array(b, dtype=x.xd_str)
#a = np.array(["hello", "momma", 'love', "me"], dtype=x.xd_str)
#print(a
#print(map(lambda x: str(x), a)
#print(repr(a[1])
#x = a[1]
#print(type(x), x 
#print(a[1:-1]
#print(a

import gc
n = 0

#while True:
#    a = np.arange(100)
#    a = np.array(["hello", "momma", 'love', "me"], dtype=x.xd_str)
#    del a
#    a[0] = "wakka"
#    gc.collect()

#    n += 1
#    if n%100 == 0:
#        gc.collect()
#        n = 0



#a = np.array([1, 65], dtype=x.xd_int)
#print(np.array([1, 65], dtype=x.xd_int)[0]
#print(a

print("Test desc")
print(xdtest.xdstlc.xd_str.fields)
#print(xdtest.xdstlc.xd_str.names 
