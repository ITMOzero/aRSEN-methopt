import optuna

from newtonian.newton_method import newton_method
from newtonian.scipy_method import scipy_method


def newton_optimize_with_optuna(f, grad, hess, eps: float=1e-6, max_iter: int=1000, max_iter_for_backtracking: int=100):
    def objective(trial):
        alpha_0 = trial.suggest_float('alpha_0', 0.1, 2.0)
        q = trial.suggest_float('q', 0.5, 1.0)
        c = trial.suggest_float('c', 1e-4, 1e-2)
        x_0 = trial.suggest_float('initial_x', -10, 10)

        x_newton, it_newton, path_newton = newton_method(f, grad, hess, x_0=x_0, eps=eps, max_iter=max_iter, params={'alpha_0': alpha_0, 'q': q, 'c': c, 'max_iter': max_iter_for_backtracking})
        return it_newton


    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    optimal_params = study.best_params
    print(f"Optimal parameters: {optimal_params}")

    x_newton, it_newton, path_newton = newton_method(f, grad, hess, x_0=optimal_params['initial_x'], eps=1e-6, max_iter=1000, params={'alpha_0': optimal_params['alpha_0'], 'q': optimal_params['q'], 'c': optimal_params['c'], 'max_iter': 100})

    return x_newton, it_newton, path_newton

def scipy_optimize_with_optuna(f, grad, hess, method, options: dict=None):
    def objective(trial):
        x_0 = trial.suggest_float('initial_x', -10, 10)

        x_newton, it_newton, path_newton = scipy_method(f, grad, hess=hess, x_0=x_0, method=method, options=options)
        return it_newton


    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    optimal_params = study.best_params
    print(f"Optimal parameters: {optimal_params}")

    x_newton, it_newton, path_newton = scipy_method(f, grad, x_0=optimal_params['initial_x'], method=method, options=options)

    return x_newton, it_newton, path_newton