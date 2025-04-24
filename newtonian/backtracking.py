
def backtracking(f: callable, x: float,
                 f_x: float, grad_x: float, p: float,
                 alpha_0: float=1.0, q: float=0.5, c: float=1e-4, max_iter: int=50) -> float:
    alpha = alpha_0
    for _ in range(max_iter):
        if f(x + alpha * p) <= f_x + c * alpha * grad_x * p: break
        alpha *= q
    return alpha