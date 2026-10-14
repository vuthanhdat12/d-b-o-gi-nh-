# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np

# Cố định seed ngẫu nhiên
np.random.seed(2)

# 1. Tập dữ liệu đầu vào
X_raw = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
         2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]
y = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]

N = len(X_raw)

# 2. Mở rộng dữ liệu (Thêm hằng số bias 1.0 vào mỗi điểm x)
# X là danh sách các cột dạng: [[1.0, x1], [1.0, x2], ...]
X = [[1.0, X_raw[i]] for i in range(N)]

# 3. Các hàm toán học thủ công (Manual Functions)

def manual_exp(x):
    """Tính e^x thủ công bằng chuỗi Taylor mở rộng (100 phần tử)"""
    # Xử lý tránh tràn số khi x quá nhỏ
    if x < -50:
        return 0.0
    if x > 50:
        return 1e22
    result = 1.0
    term = 1.0
    for i in range(1, 100):
        term *= x / i
        result += term
    return result

def manual_sigmoid(s):
    """Tính hàm Sigmoid thủ công: 1 / (1 + e^-s)"""
    return 1.0 / (1.0 + manual_exp(-s))

def manual_dot(vec1, vec2):
    """Tính tích vô hướng của 2 vectơ dạng danh sách: sum(v1[i] * v2[i])"""
    dot_sum = 0.0
    for i in range(len(vec1)):
        dot_sum += vec1[i] * vec2[i]
    return dot_sum

def manual_norm(vec1, vec2):
    """Tính khoảng cách L2-norm giữa 2 vectơ thủ công: sqrt(sum((v1[i] - v2[i])^2))"""
    squared_diff_sum = 0.0
    for i in range(len(vec1)):
        squared_diff_sum += (vec1[i] - vec2[i]) ** 2
    return squared_diff_sum ** 0.5

def manual_permutation(n):
    """Xáo trộn ngẫu nhiên danh sách chỉ số từ 0 đến n-1 bằng thuật toán Fisher-Yates Shuffle"""
    arr = list(range(n))
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr

# 4. Thuật toán Logistic Regression viết thủ công
def manual_logistic_sigmoid_regression(X, y, w_init, eta, tol=1e-4, max_count=10000):
    # Khởi tạo lịch sử danh sách trọng số w
    w = [list(w_init)]
    N = len(X)
    d = len(X[0])
    count = 0
    check_w_after = 20

    while count < max_count:
        # Xáo trộn thứ tự các mẫu dữ liệu thủ công
        mix_id = manual_permutation(N)

        for i in mix_id:
            xi = X[i]        # Vectơ đặc trưng mẫu thứ i: [1.0, x_i]
            yi = y[i]        # Nhãn thực tế: 0 hoặc 1
            
            # Tính tích vô hướng s = w^T * x
            s = manual_dot(w[-1], xi)
            
            # Tính xác suất dự đoán zi = sigmoid(s)
            zi = manual_sigmoid(s)
            
            # Cập nhật trọng số w_new thủ công từng phần tử
            # w_new = w[-1] + eta * (yi - zi) * xi
            error = yi - zi
            w_new = [0.0] * d
            for j in range(d):
                w_new[j] = w[-1][j] + eta * error * xi[j]
            
            count += 1
            
            # Kiểm tra điều kiện dừng hội tụ sau mỗi 20 lần cập nhật
            if count % check_w_after == 0:
                # Lấy trọng số ở 20 bước trước đó
                w_old = w[-check_w_after]
                # So sánh độ chênh lệch L2-norm
                if manual_norm(w_new, w_old) < tol:
                    w.append(w_new)
                    return w

            w.append(w_new)

    return w

# 5. Chạy thuật toán và in kết quả
eta = 0.05
# Khởi tạo ngẫu nhiên trọng số ban đầu [w0, w1]
w_init = np.random.randn(2).tolist()

w_history = manual_logistic_sigmoid_regression(X, y, w_init, eta)

print("Trọng số tối ưu tìm được [w0, w1]:")
print(w_history[-1])