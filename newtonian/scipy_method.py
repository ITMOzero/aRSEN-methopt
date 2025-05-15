import scipy.optimize as opt
import numpy as np

def scipy_method(f: callable, grad: callable, hess: callable=None,
                 x_0: float=None, method: str= 'BFGS', options: dict=None):
    path = []
    callback = lambda x_k: path.append(float(x_k[0]) if isinstance(x_k, np.ndarray) else float(x_k))
    res = opt.minimize(f, np.array(x_0), method=method, jac=grad, hess=hess,
                       callback=callback, options=options)
    return res.x[0], res.nit, path

