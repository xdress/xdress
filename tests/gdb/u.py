import xdtest.pydevice as d

x = d.TwoNums(5, 10)
print x.a, x.b

f = lambda a_, b_: a_ + b_
x.op = f
print x.op

# Test that function pointer in C is getting set properly
x._op = None
print "x._op is None", x._op is None
print x.op 

v = x.op(14, 16)
print v

