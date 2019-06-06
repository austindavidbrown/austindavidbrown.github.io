from sympy import *

t = symbols("t", real = True)
x = symbols("x", real = True)
plot(ln(gamma(t)), (t, 1, 10))