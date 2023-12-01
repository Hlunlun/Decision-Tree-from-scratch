''' plot contour, conturf '''
import numpy as np 
import matplotlib.pyplot as plt 

x = range(4)
y = range(4)

# coordinate
x, y = np.meshgrid(x, y)


# depth
z = [
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
]

fig = plt.figure(figsize=(14,6))

fig.add_subplot(121)
plt.contour(x, y, z)
# plt.show()
fig.add_subplot(122)
plt.contourf(x, y, z)
# plt.show()


''' plot z as the point '''
fig = plt.figure(figsize=(14,6))
z = np.array(z)
for idx, color in [
    (z == 0, '#000'),
    (z == 1, '#F00')
]: plt.plot(x[idx], y[idx], '*', color=color)

plt.show()