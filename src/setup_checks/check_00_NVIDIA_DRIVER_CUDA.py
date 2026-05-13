import cupy as cp

print("CUDA runtime version:", cp.cuda.runtime.runtimeGetVersion())
print("Driver version:", cp.cuda.runtime.driverGetVersion())
print("Device count:", cp.cuda.runtime.getDeviceCount())
