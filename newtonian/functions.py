import autograd.numpy as anp

def select_function(f_type):

    if f_type == 'f1':
        return lambda x: x ** 2 + x / 3 - 5

    elif f_type == 'f2':
        return lambda x: 3 * x ** 4 - 3 * x ** 3 - x ** 2 - x

    elif f_type == 'f3':
        return lambda x: x ** 2 + 5 * anp.sin(x)

    elif f_type == 'f4':
        return lambda x: 5 * anp.sin(3 * x) + x ** 2

    elif f_type == 'f5':
        return lambda x: 3 * anp.sin(3 * x) * anp.exp(-x ** 2 / 50)

    elif f_type == 'scipy_f1':
        return lambda x: x ** 4 + 2 * x ** 2 + 3 * x - 10

    else:
        raise ValueError(f"Unknown function type: {f_type}")

