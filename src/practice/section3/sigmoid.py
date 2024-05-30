import numpy as np
import matplotlib.pyplot as plt

e = np.e  # ネイピア数


def sigmoid(x):
    s = 1 / (1 + e**-x)  # シグモイド関数
    return s


# x = np.linspace(-5, 5)
# y_sig = sigmoid(x)

# plt.plot(x, y_sig)

# dx = 0.1
# x = np.linspace(-8, 8)
# y_sig = sigmoid(x)
# y_d = (sigmoid(x + dx) - sigmoid(x)) / dx  # シグモイド関数の傾き

# plt.plot(x, y_sig, label="sigmoid")
# plt.plot(x, y_d, label="d_sigmoid")
# plt.legend()


def df_sigmoid(x):
    d = sigmoid(x) * (1 - sigmoid(x))  # シグモイド関数を微分
    return d


dx = 0.1
x = np.linspace(-8, 8)

y_sig = sigmoid(x)
y_d = (sigmoid(x + dx) - sigmoid(x)) / dx
y_df = df_sigmoid(x)

plt.plot(x, y_sig, label="sigmoid")
plt.plot(x, y_d, label="d")
plt.plot(x, y_df, label="df")
plt.legend()

plt.xlabel("x", size=14)
plt.ylabel("y", size=14)
plt.grid()
plt.show()
