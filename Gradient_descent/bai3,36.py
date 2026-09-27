def f(x):
    return x**2 - 4*x + 5

def df(x):
    return 2*x - 4


# Tham số
x = 5
eta = 0.2

print("Bước 0:")
print(f"x = {x:.4f}")
print(f"f(x) = {f(x):.6f}")

# Thực hiện 4 bước cập nhật
for i in range(1, 5):
    gradient = df(x)

    # Gradient Descent
    x = x - eta * gradient

    print(f"\nBước {i}:")
    print(f"x = {x:.4f}")
    print(f"f(x) = {f(x):.6f}")