# Bài 3.28

import numpy as np

w = np.array([-2, 1, 0])
x = np.array([2, 3, 1])
y = 1

# Tính w^T x
wx = np.dot(w, x)

print("Trước cập nhật:")
print("w =", w)
print("w^T x =", wx)

# Dự đoán
if wx >= 0:
    y_pred = 1
else:
    y_pred = -1

print("Nhãn dự đoán =", y_pred)
print("Nhãn thực tế =", y)

# Kiểm tra phân lớp sai
if y_pred != y:
    print("=> Mẫu bị phân lớp sai.")

    # Cập nhật Perceptron
    w = w + y * x

    print("\nSau cập nhật:")
    print("w mới =", w)

    # Tính lại w^T x
    wx_new = np.dot(w, x)

    print("w^T x mới =", wx_new)

else:
    print("=> Mẫu được phân lớp đúng.")
