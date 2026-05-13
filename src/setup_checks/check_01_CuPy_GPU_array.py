import cupy as cp

x = cp.arange(10)
y = x * 2

print("CuPy works:", y)
print("Device:", cp.cuda.Device())
