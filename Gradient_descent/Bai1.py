def grad(x):
    return 2*x

def cost(x):
    return x**2 - 2

def myGD(x0, eta):
    x = [x0]

    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])

        if abs(grad(x_new)) < 1e-3:
            break

        x.append(x_new)

    return (x, it)


# Chạy Gradient Descent
x, it = myGD(4, 0.1)
print("solution x1 = %f, cost = %f, after %d interations"%(x[-1], cost(x[-1]), it))
