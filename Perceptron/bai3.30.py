

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. XÂY DỰNG LỚP PERCEPTRON
# ============================================================

class Perceptron:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0

    # --------------------------------------------------------
    # Hàm fit: huấn luyện mô hình
    # --------------------------------------------------------
    def fit(self, X, y):

        # Số lượng đặc trưng
        n_features = X.shape[1]

        # Khởi tạo trọng số bằng 0
        self.w = np.zeros(n_features)

        # Huấn luyện nhiều lần
        for epoch in range(self.epochs):

            errors = 0

            # Duyệt qua từng mẫu
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
                    self.w = self.w + (
                        self.learning_rate * yi * xi
                    )

                    # Cập nhật bias
                    self.b = self.b + (
                        self.learning_rate * yi
                    )

                    errors += 1

            # Nếu không còn mẫu nào phân lớp sai
            if errors == 0:
                print("\nMô hình đã hội tụ tại epoch:", epoch + 1)
                break

        return self

    # --------------------------------------------------------
    # Hàm predict: dự đoán dữ liệu mới
    # --------------------------------------------------------
    def predict(self, X):

        # Tính w^T*x + b
        z = np.dot(X, self.w) + self.b

        # Nếu >= 0 thì dự đoán 1
        # Nếu < 0 thì dự đoán -1
        return np.where(z >= 0, 1, -1)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("========================================")
print("        BÀI 3.30 - PERCEPTRON")
print("========================================")

print("\n1. Đọc dữ liệu...")

data = load_breast_cancer()

X = data.data
y = data.target

print("Số lượng mẫu:", X.shape[0])
print("Số lượng đặc trưng:", X.shape[1])

print("Nhãn ban đầu:", np.unique(y))


# ============================================================
# 3. CHUYỂN NHÃN 0, 1 THÀNH -1, +1
# ============================================================

# Perceptron của chúng ta sử dụng nhãn -1 và +1

y = np.where(y == 0, -1, 1)

print("Nhãn sau khi chuyển:", np.unique(y))


# ============================================================
# 4. CHIA DỮ LIỆU TRAIN / TEST
# ============================================================

print("\n2. Chia dữ liệu Train/Test...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Số mẫu Train:", X_train.shape[0])
print("Số mẫu Test :", X_test.shape[0])


# ============================================================
# 5. CHUẨN HÓA DỮ LIỆU
# ============================================================

print("\n3. Chuẩn hóa dữ liệu...")

scaler = StandardScaler()

# Fit scaler trên tập train
X_train = scaler.fit_transform(X_train)

# Dùng scaler đã học để biến đổi tập test
X_test = scaler.transform(X_test)


# ============================================================
# 6. TẠO MÔ HÌNH PERCEPTRON
# ============================================================

print("\n4. Khởi tạo mô hình Perceptron...")

model = Perceptron(
    learning_rate=0.01,
    epochs=1000
)


# ============================================================
# 7. HUẤN LUYỆN MÔ HÌNH
# ============================================================

print("\n5. Bắt đầu huấn luyện...")

model.fit(X_train, y_train)


# ============================================================
# 8. DỰ ĐOÁN TRÊN TẬP TEST
# ============================================================

print("\n6. Dự đoán dữ liệu Test...")

y_pred = model.predict(X_test)


# ============================================================
# 9. TÍNH CÁC ĐỘ ĐO
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    pos_label=1
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=1
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# 11. IN KẾT QUẢ
# ============================================================

print("\n========================================")
print("              KẾT QUẢ")
print("========================================")

print("Accuracy : {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))
print("Recall   : {:.4f}".format(recall))
print("F1-score : {:.4f}".format(f1))

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 12. IN TRỌNG SỐ VÀ BIAS
# ============================================================

print("\n========================================")
print("         THÔNG SỐ MÔ HÌNH")
print("========================================")

print("Weights:")
print(model.w)

print("\nBias:")
print(model.b)


