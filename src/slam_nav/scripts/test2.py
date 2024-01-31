import numpy as np
import matplotlib.pyplot as plt

start_x = 19.967400327488775
start_y = -9.98308612539298

end_x = 22.20813466624915
end_y = -9.9657718701554

x = np.linspace(-0.9, 0.9, 10)
y = np.arctanh(x)
print(y)

y *= 10
y /= np.abs(np.min(y))

print(y)

plt.scatter(x, y)
plt.show()
