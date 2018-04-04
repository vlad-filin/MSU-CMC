def grad_finite_diff(function, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    res = np.zeros(w.shape)
    for i in range(0, w.shape[0]):
        e_i = np.zeros(w.shape)
        e_i[i] = 1
        res[i] = (function(w + eps * e_i) - f(w))/eps
    return res
