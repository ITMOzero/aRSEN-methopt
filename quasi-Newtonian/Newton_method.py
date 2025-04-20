from backtracking import backtracking

def newton_method(f: callable, grad: callable, hess: callable,
                  x_0: float, eps: float=1e-6, max_iter: int=100,
                  params: dict=None) -> tuple:
    x = x_0
    path = [x]
    i = 0
    while i <= max_iter:
        grad_x = grad(x)
        if abs(grad_x) < eps or (hessian_x := hess(x)) == 0: break
        p = -grad_x / hessian_x
        alpha = backtracking(f, x, f(x), grad_x, p, **(params or {}))
        x += alpha * p
        path.append(x)
        i += 1
    return x, i, path