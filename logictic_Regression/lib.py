# Logistic Regression dùng thư viện scikit-learn
import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Dữ liệu: số giờ học và kết quả đỗ (1) / trượt (0)
X_raw = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
                  2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# scikit-learn yêu cầu X có shape (n_samples, n_features) = (20, 1)
# KHÔNG cần tự thêm cột bias, thư viện tự xử lý (fit_intercept=True)
X = X_raw.reshape(-1, 1)

# 2. Khởi tạo và huấn luyện mô hình
# Mặc định sklearn có điều chuẩn L2 (C=1). Để gần với code thủ công
# (không điều chuẩn), đặt C rất lớn.
model = LogisticRegression(C=1e5, solver="lbfgs", max_iter=10000)
model.fit(X, y)

# 3. Lấy trọng số, ghép lại thành w = [w0, w1] giống code thủ công
w0 = model.intercept_[0]   # bias
w1 = model.coef_[0][0]     # hệ số của số giờ học
w = np.array([[w0], [w1]])

print("Trọng số tìm được [w0, w1]:")
print(w.ravel())
