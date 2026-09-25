def grad(x):
    return x**2 - 1

def cost(x):
    return (1/3)*x**3 - x

def myGD(x0, eta):
    x = [x0]

    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])

        if abs(grad(x_new)) < 1e-3:
            break

        x.append(x_new)

    return (x, it)


# Chạy Gradient Descent
x, it = myGD(0, 0.1)

print("solution x1 = %f, cost = %f, after %d interations"%(x[-1], cost(x[-1]), it))