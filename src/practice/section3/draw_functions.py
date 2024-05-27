import numpy as np  # NumPyの導入。以降npという名前でNumPyを使用できる。
import matplotlib.pyplot as plt  # Matplotlibの導入。以降pltという名前でMatplotlibを使用できる。

a = [0, 1, 2, 3, 4, 5]
b = np.array(a)  # リストからNumPyの配列を作る
print(b)

aa = [0, 1, 2, 3, 4, 5]
bb = np.array(aa)
print(bb)

x = np.linspace(-5, 5)

print(x)
print(len(x))

# y = 2 * x + 1
# plt.plot(x, y)
# plt.show()

# y_1 = 1.5 * x
# y_2 = -2 * x + 1
# plt.plot(x, y_1, label="y = 1.5x")
# plt.plot(x, y_2, label="y = -2x + 1", linestyle="dashed")
# plt.title("Graph of y = 1.5x", size=20, color="green")

y_1 = 2 * x + 1
y_2 = x**2 + 1
y_3 = 0.5 * x**3 - 6 * x

plt.xlabel("X value", size=14, color="red")
plt.ylabel("Y value", size=14, color="blue")
plt.plot(x, y_1, label="y = 2x + 1")
plt.plot(x, y_2, label="y = x^2 + 1")
plt.plot(x, y_3, label="y = 0.5x^3 - 6x")
plt.title("Graph of y = 2x + 1", size=20, color="green")

plt.grid()
plt.legend()

plt.show()
