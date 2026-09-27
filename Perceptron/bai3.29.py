# Bài 3.29

import numpy as np


class Perceptron:

    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0

    # Hàm huấn luyện
    def fit(self, X, y):

        # Khởi tạo trọng số = 0
        n_features = X.shape[1]
        self.w = np.zeros(n_features)

        for epoch in range(self.epochs):

            errors = 0

            for xi, yi in zip(X, y):

                # Tính w^T*x + b
                z = np.dot(xi, self.w) + self.b

                # Dự đoán
                if z >= 0:
                    y_pred = 1
                else:
                    y_pred = -1

                # Nếu dự đoán sai
                if y_pred != yi:

                    # Cập nhật trọng số
                    self.w = self.w + \
                             self.learning_rate * yi * xi

                    # Cập nhật bias
                    self.b = self.b + \
                             self.learning_rate * yi

                    errors += 1

            # Nếu không còn lỗi
            if errors == 0:
                print("Mô hình hội tụ tại epoch:", epoch + 1)
                break

    # Hàm dự đoán
    def predict(self, X):

        z = np.dot(X, self.w) + self.b

        return np.where(z >= 0, 1, -1)
# Dữ liệu huấn luyện
X_train = np.array([
    [2, 3],
    [3, 4],
    [-2, -3],
    [-3, -4]
])

y_train = np.array([
    1,
    1,
    -1,
    -1
])


# Tạo model
model = Perceptron(
    learning_rate=0.1,
    epochs=100
)

# Huấn luyện
model.fit(X_train, y_train)


# Dự đoán dữ liệu mới
X_test = np.array([
    [4, 5],
    [-4, -5],
    [1, 2],
    [-1, -2]
])

y_pred = model.predict(X_test)

print("\nTrọng số:", model.w)
print("Bias:", model.b)
print("Dự đoán:", y_pred)
