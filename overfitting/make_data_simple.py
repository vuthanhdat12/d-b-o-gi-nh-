import numpy as np
import pandas as pd

np.random.seed(42)
n = 300

dien_tich = np.random.normal(85, 25, n).clip(25, 250)              # m2
so_phong_ngu = np.random.randint(1, 6, n)                          # 1-5 phòng
khoang_cach_trung_tam = np.random.uniform(0.5, 30, n)              # km, càng nhỏ càng gần trung tâm

# Công thức tạo giá thật (đơn vị: trăm triệu VND)
gia = (
    dien_tich * 1.0
    + so_phong_ngu * 8
    - khoang_cach_trung_tam * 2.2
    + 50
    + np.random.normal(0, 10, n)   # nhiễu tự nhiên (yếu tố khác không đo được: hướng nhà, view, pháp lý...)
)
gia = gia.clip(20, None)

df = pd.DataFrame({
    "dien_tich": dien_tich.round(1),
    "so_phong_ngu": so_phong_ngu,
    "khoang_cach_trung_tam": khoang_cach_trung_tam.round(2),
    "gia_nha": gia.round(1),
})

df.to_csv("house_prices_simple.csv", index=False)
print(df.shape)
print(df.head(10))
