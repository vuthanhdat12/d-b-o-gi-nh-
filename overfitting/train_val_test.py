import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
# ============================================================
# ĐỌC DỮ LIỆU (tự tìm file CSV cùng thư mục với file .py này)
# ============================================================
thu_muc_hien_tai = os.path.dirname(os.path.abspath(__file__))
duong_dan_csv = os.path.join(thu_muc_hien_tai, "house_prices_simple.csv")
df = pd.read_csv(duong_dan_csv)
 
X = df[["dien_tich", "so_phong_ngu", "khoang_cach_trung_tam"]]
y = df["gia_nha"]
 
# Chia dữ liệu: Train 60% / Validation 20% / Test 20%
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
 
print(f"Số mẫu - Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}\n")
 
 
def danh_gia(model, X, y):
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    return mae, rmse, r2
 
 
def in_bang(ten, model, Xtr=None, Xva=None, Xte=None):
    Xtr = X_train if Xtr is None else Xtr
    Xva = X_val if Xva is None else Xva
    Xte = X_test if Xte is None else Xte
    mae_tr, rmse_tr, r2_tr = danh_gia(model, Xtr, y_train)
    mae_va, rmse_va, r2_va = danh_gia(model, Xva, y_val)
    mae_te, rmse_te, r2_te = danh_gia(model, Xte, y_test)
    print(f"--- {ten} ---")
    print(f"{'Tập':<12}{'MAE':>10}{'RMSE':>10}{'R2':>10}")
    print(f"{'Train':<12}{mae_tr:>10.2f}{rmse_tr:>10.2f}{r2_tr:>10.3f}")
    print(f"{'Validation':<12}{mae_va:>10.2f}{rmse_va:>10.2f}{r2_va:>10.3f}")
    print(f"{'Test':<12}{mae_te:>10.2f}{rmse_te:>10.2f}{r2_te:>10.3f}")
    print(f"Chênh lệch R2 (Train - Validation): {r2_tr - r2_va:.3f}  <-- càng lớn càng overfit\n")
 
 
# ============================================================
# BƯỚC 1: CỐ TÌNH GÂY OVERFITTING
# ============================================================
print("=" * 65)
print("BƯỚC 1: GÂY OVERFITTING")
print("=" * 65)
 
print("\n>>> Cách 1: Decision Tree KHÔNG giới hạn độ sâu")
tree_overfit = DecisionTreeRegressor(random_state=42)  # không giới hạn gì cả
tree_overfit.fit(X_train, y_train)
in_bang("Decision Tree (không giới hạn)", tree_overfit)
print(f"Độ sâu cây thực tế: {tree_overfit.get_depth()} | Số lá: {tree_overfit.get_n_leaves()}\n")
 
print(">>> Cách 2: Hồi quy đa thức bậc rất cao (degree=6) trên chỉ 3 biến")
poly_overfit = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=6, include_bias=False),
    LinearRegression()
)
poly_overfit.fit(X_train, y_train)
in_bang("Hồi quy đa thức bậc 6", poly_overfit)
n_feat = poly_overfit.named_steps["polynomialfeatures"].n_output_features_
print(f"Số feature sau khi mở rộng đa thức: {n_feat} (từ 3 feature gốc, {len(X_train)} mẫu train)\n")
 
 
# ============================================================
# BƯỚC 2: ÁP DỤNG CÁC KỸ THUẬT KHẮC PHỤC OVERFITTING
# ============================================================
print("=" * 65)
print("BƯỚC 2: ÁP DỤNG CÁC KỸ THUẬT KHẮC PHỤC")
print("=" * 65)
 
print("\n>>> Kỹ thuật 1: PRUNING - giới hạn max_depth, min_samples_leaf")
tree_pruned = DecisionTreeRegressor(max_depth=3, min_samples_leaf=8, random_state=42)
tree_pruned.fit(X_train, y_train)
in_bang("Decision Tree (max_depth=3, min_leaf=8)", tree_pruned)
 
print(">>> Kỹ thuật 2: CROSS-VALIDATION để tự động chọn max_depth tốt nhất")
param_grid = {"max_depth": [1, 2, 3, 4, 5, 6, None], "min_samples_leaf": [1, 4, 8, 12]}
grid = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5, scoring="r2")
grid.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))  # CV tự chia fold trong train+val
print(f"Tham số tốt nhất theo Cross-Validation: {grid.best_params_}")
best_tree = grid.best_estimator_
best_tree.fit(X_train, y_train)  # fit lại chỉ trên train để so sánh công bằng
in_bang("Decision Tree (tham số chọn qua CV)", best_tree)
 
print(">>> Kỹ thuật 3: REGULARIZATION - Ridge (L2) trên đa thức bậc 6")
ridge_model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=6, include_bias=False),
    Ridge(alpha=50.0)
)
ridge_model.fit(X_train, y_train)
in_bang("Ridge (alpha=50) + đa thức bậc 6", ridge_model)
 
print(">>> Kỹ thuật 4: REGULARIZATION - Lasso (L1) trên đa thức bậc 6, tự loại bớt feature")
lasso_model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=6, include_bias=False),
    Lasso(alpha=0.5, max_iter=20000)
)
lasso_model.fit(X_train, y_train)
in_bang("Lasso (alpha=0.5) + đa thức bậc 6", lasso_model)
he_so = lasso_model.named_steps["lasso"].coef_
print(f"Lasso đã đưa {np.sum(he_so == 0)}/{len(he_so)} hệ số về 0 (tự loại bỏ độ phức tạp thừa)\n")
 
print(">>> Kỹ thuật 5: Mô hình đơn giản hơn - Linear Regression (không mở rộng đa thức)")
lr_simple = LinearRegression()
lr_simple.fit(X_train, y_train)
in_bang("Linear Regression (3 feature gốc, không đa thức)", lr_simple)
print("Hệ số hồi quy:", dict(zip(X.columns, lr_simple.coef_.round(2))))
print("Hằng số (intercept):", round(lr_simple.intercept_, 2))