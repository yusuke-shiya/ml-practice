import numpy as np
import matplotlib.pyplot as plt

# x = np.linspace(-2, 2)

# y_2 = 2**x
# y_3 = 3**x

# plt.plot(x, y_2, label="2^x")
# plt.plot(x, y_3, label="3^x")
# plt.legend()

# plt.xlabel("x", size=14)
# plt.ylabel("y", size=14)
# plt.grid()
# plt.show()

# x = np.linspace(-2, 2)

# e = np.e

# y_2 = 2**x
# y_e = e**x
# y_3 = 3**x

# plt.plot(x, y_2, label="2^x")
# plt.plot(x, y_e, label="e^x")
# plt.plot(x, y_3, label="3^x")
# plt.legend()

# plt.xlabel("x", size=14)
# plt.ylabel("y", size=14)
# plt.grid()
# plt.show()

x = np.linspace(-2, 2)
dx = 0.001
e = np.e

# y_e = e**x
# y_de = (e ** (x + dx) - e**x) / dx
y_e = 2**x
y_de = (2 ** (x + dx) - 2**x) / dx

plt.plot(x, y_e, label="2^x")
plt.plot(x, y_de, label="d2")
plt.legend()

plt.xlabel("x", size=14)
plt.ylabel("y", size=14)
plt.grid()
plt.show()
