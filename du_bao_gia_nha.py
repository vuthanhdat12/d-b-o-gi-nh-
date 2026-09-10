file = open("du_lieu_gia_nha.txt", "r")

x = []  # diện tích
y = []  # giá nhà

for line in file:
    data = line.split()

    dien_tich = float(data[0])
    gia = float(data[1])

    x.append(dien_tich)
    y.append(gia)

file.close()


# Tính giá trị trung bình
x_tb = sum(x) / len(x)
y_tb = sum(y) / len(y)


# Tính hệ số a và b
tu_so = 0
mau_so = 0

for i in range(len(x)):
    tu_so += (x[i] - x_tb) * (y[i] - y_tb)
    mau_so += (x[i] - x_tb) ** 2

a = tu_so / mau_so
b = y_tb - a * x_tb


# Hiển thị công thức
print("===================================")
print("     CHƯƠNG TRÌNH DỰ BÁO GIÁ NHÀ")
print("===================================")

print("Công thức:")
print("Gia = a * Dien tich + b")

print("a =", round(a, 2))
print("b =", round(b, 2))


# Nhập diện tích cần dự báo
dien_tich = float(input("\nNhập diện tích nhà (m2): "))

gia_du_bao = a * dien_tich + b


# Hiển thị kết quả
print("\n===================================")
print("KẾT QUẢ DỰ BÁO")
print("===================================")

print("Diện tích:", dien_tich, "m2")
print("Giá dự báo:", round(gia_du_bao, 2), "triệu đồng")
print("Tương đương:", round(gia_du_bao / 1000, 2), "tỷ đồng")