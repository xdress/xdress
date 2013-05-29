from __future__ import print_function
import xdtest.pydevice as d
import sys

x = d.TwoNums(5, 10)
print("a and b = ", x.a, x.b)

f = lambda a_, b_: a_ + b_
print("f's refcount before assignment: ", sys.getrefcount(f))
x.op = f
print("f's refcount after assignment: ", sys.getrefcount(f))
print("x.op = ", x.op)

# Test that function pointer in C is getting set properly
x._op = None
print("x._op is None:", x._op is None)
print("f's refcount after None assignment: ", sys.getrefcount(f))
print("x.op = ", x.op)

v = x.op(14, 16)
print("result of x.op(14, 16) = ", v)
v = x.call_from_c(14, 16)
print("result of x.call_from_c(14, 16) = ", v)

print("-"*40)

x.op = lambda a_, b_: 2*a_ + b_**2
print("x.op = ", x.op)
print("f's refcount after op re-assignment: ", sys.getrefcount(f))

v = x.op(14, 16)
print("result of x.op(14, 16) = ", v)
v = x.call_from_c(14, 16)
print("result of x.call_from_c(14, 16) = ", v)

print("-"*40)

# test two instances
y = d.TwoNums(5, 10)
x.op = lambda a_, b_: a_ + b_
print("x.op = ", x.op)
y.op = lambda a_, b_: 2*a_ + b_**2
print("y.op = ", y.op)
v = x.op(14, 16)
print("result of x.op(14, 16) = ", v)
v = x.call_from_c(14, 16)
print("result of x.call_from_c(14, 16) = ", v)
