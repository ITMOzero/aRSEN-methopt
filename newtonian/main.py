from autograd import grad, hessian

from functions import select_function
from graphics import *
from newton_method import newton_method
from optimize_with_optuna import newton_optimize_with_optuna, scipy_optimize_with_optuna
from scipy_method import scipy_method


# TODO gif for f3-5 functions
def wrap_count(func):
    class Wrapper:
        def __init__(self, f):
            self.f = f
            self.n_calls = 0

        def __call__(self, x):
            self.n_calls += 1
            return self.f(x)

        def reset(self):
            self.n_calls = 0

    return Wrapper(func)


def print_info(name: str, x: float, it: int, f_calls: int, grad_calls: int = None, hess_calls: int = None):
    print(f"{name} method result:", x)
    print("Iterations:", it)
    print("Function calls:", f_calls)
    if grad_calls is not None: print("Gradient calls:", grad_calls)
    if hess_calls is not None: print("Hessian calls:", hess_calls)
    print()


if __name__ == '__main__':
    x_0 = 5.0

    func = select_function('f3')

    grad_f = grad(func)
    hess_f = hessian(func)

    f = wrap_count(func)
    grad = wrap_count(grad_f)
    hess = wrap_count(hess_f)

    params = {'alpha_0': 10.0, 'q': 0.5, 'c': 0.5, 'max_iter': 100}

    # Newton method
    f.reset();
    grad.reset();
    hess.reset()
    x_newton, it_newton, path_newton = newton_method(f, grad, hess, x_0=x_0, eps=1e-6, max_iter=1000, params=params)
    print_info("Newton", x_newton, it_newton, f.n_calls, grad.n_calls, hess.n_calls)

    # Newton-CG
    f.reset();
    grad.reset();
    hess.reset()
    x_ncg, it_ncg, path_ncg = scipy_method(f, grad, hess=hess, x_0=x_0, method='Newton-CG',
                                           options={'xtol': 1e-6, 'maxiter': 100})
    print_info("Newton-CG", x_ncg, it_ncg, f.n_calls, grad.n_calls, hess.n_calls)

    # BFGS
    f.reset();
    grad.reset();
    hess.reset()
    x_bfgs, it_bfgs, path_bfgs = scipy_method(f, grad, x_0=x_0, method='BFGS', options={'gtol': 1e-6, 'maxiter': 100})
    print_info("BFGS", x_bfgs, it_bfgs, f.n_calls, grad.n_calls, hess.n_calls)

    # fixme: 9999+ iterations
    # # Our BFGS
    # f.reset(); grad.reset()
    # x_bfgs_mod_x0 = np.array([x_0])
    # x_bfgs_mod_result, path_bfgs_mod = bfgs_modified(f,grad,x_bfgs_mod_x0,max_iter=1000,tol=1e-6)
    # print_info("BFGS (modified)", x_bfgs_mod_result[0], len(path_bfgs_mod), f.n_calls + grad.n_calls)

    # f.reset(); grad.reset(); hess.reset()
    # x_bfgs_optuna_x0, it_bfgs_optuna_trials, path_bfgs_optuna = bfgs_optimize_with_optuna(f, grad, eps=1e-6, max_iter=1000)

    # Newton method optimized with optuna
    f.reset();
    grad.reset();
    hess.reset()
    x_newton_optimized, it_newton_optimized, path_newton_optimized = newton_optimize_with_optuna(f, grad, hess,
                                                                                                 eps=1e-6,
                                                                                                 max_iter=1000,
                                                                                                 max_iter_for_backtracking=100)
    print_info("Optimized Newton", x_newton_optimized, it_newton_optimized, f.n_calls, grad.n_calls, hess.n_calls)

    # Newton-CG method optimized with optuna
    f.reset();
    grad.reset();
    hess.reset()
    x_ncg_optimized, it_ncg_optimized, path_ncg_optimized = scipy_optimize_with_optuna(f, grad, hess=hess,
                                                                                       method='Newton-CG',
                                                                                       options={'xtol': 1e-6,
                                                                                                'maxiter': 100})
    print_info("Optimized Newton-CG", x_ncg_optimized, it_ncg_optimized, f.n_calls, grad.n_calls, hess.n_calls)

    # BFGS method optimized with optuna
    f.reset();
    grad.reset();
    hess.reset()
    x_bfgs_optimized, it_bfgs_optimized, path_bfgs_optimized = scipy_optimize_with_optuna(f, grad, hess=None,
                                                                                          method='BFGS',
                                                                                          options={'gtol': 1e-6,
                                                                                                   'maxiter': 100})
    print_info("Optimized BFGS", x_bfgs_optimized, it_bfgs_optimized, f.n_calls, grad.n_calls, hess.n_calls)

    # Визуализация всех
    paths = {
        'Newton': path_newton,
        'BFGS': path_bfgs,
        'Newton-CG': path_ncg,
        # 'Modified BFGS': path_bfgs_optuna,
        'Optimized Newton': path_newton_optimized,
        'Optimized BFGS': path_bfgs_optimized,
        'Optimized Newton-CG': path_ncg_optimized,
    }
    for k, v in paths.items():
        plot_function(func, {k: v})
    # all_points = path_newton + path_bfgs + path_ncg + path_dfp
    # all_points = (path_newton + path_bfgs + path_ncg + path_newton_optimized + path_newton_optimized + path_ncg_optimized + path_bfgs_optuna)
    all_points = (
                path_newton + path_bfgs + path_ncg + path_newton_optimized + path_newton_optimized + path_ncg_optimized)
    finite_vals = [x for x in all_points if np.isfinite(x)]
    if finite_vals:
        b = max(abs(max(finite_vals)), abs(min(finite_vals)))
        for k, v in paths.items():
            animate(func, {k: v}, bounds=(-b, b))
    else:
        print("incorrect values")
