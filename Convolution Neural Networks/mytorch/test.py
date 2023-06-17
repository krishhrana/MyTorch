import resampling
import Conv1d
import numpy as np
import flatten

'''up = resampling.Upsample1d(2)
a = np.arange(1, 25).reshape((2,3,4))

a_back = np.random.randn(1, 10).reshape((1,1,10))
up_a = up.forward(a)
print(up_a)

print('\n')

back_a = up.backward(up_a)
print(back_a)


down = resampling.Downsample1d(3)
a = np.arange(1, 43).reshape((2,3,7))
print(a)
a_forward = down.forward(a)
print(a_forward)

print('\n')
a_back = down.backward(a_forward)
print(a_back)


up_2d = resampling.Upsample2d(2)
a = np.arange(1, 97).reshape((2,3,4,4))

print(a)

a_forward = up_2d.forward(a)

print(a_forward)


a_back = up_2d.backward(a_forward)
print(a_back)'''


'''down = resampling.Downsample2d(3)
a = np.arange(1, 97).reshape((2, 3, 4, 4))
print(a)
a_forward = down.forward(a)
print(a_forward)

print('\n')
a_back = down.backward(a_forward)
print(a_back)'''


'''conv = Conv1d.Conv1d_stride1(in_channels=3, out_channels=3, kernel_size=3)
#print("Weights: ", conv.W)
a = np.arange(0, 24).reshape((2, 3, 4))
a_forward = conv.forward(a)
print(a_forward)


a_back = conv.backward((a_forward))
print(a_back)'''




A = np.arange(0, 18).reshape(2, 3, 3)
flat = flatten.Flatten()
flat_A = flat.forward(A)

back_A = flat.backward(flat_A)
print(back_A)


