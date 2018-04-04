import numpy as np
import oracles as oracle
import time as t
from scipy.spatial.distance import euclidean
from scipy.special import expit


import numpy as np
from importlib import reload
import oracles as oracle
from scipy.spatial.distance import euclidean
from scipy.special import expit
reload(oracle)

class SubGDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    def __init__(self, step_alpha=1,
                 tolerance=1e-5, max_iter=1000, **kwargs):
        """

        step_alpha - float, параметр выбора шага из текста задания


        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если (f(x_{k+1}) - f(x_{k})) < tolerance: то выход
        
        max_iter - максимальное число итераций
        
        **kwargs - аргументы, необходимые для инициализации
        """
        self.alpha = step_alpha
        self.tol = tolerance
        self.max_iter = max_iter
        self.kwargs = kwargs

    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        self.my_oracle = oracle.BinaryHinge(**self.kwargs)
        if w_0 is None:
            self.w = np.zeros(X.shape[1])
        else:
            self.w = w_0
        if trace is True:
            history = {'time': [0], 'func': [self.my_oracle.func(X, y, self.w)]}
        self.best = self.w.copy()
        F_best = self.my_oracle.func(X, y, self.best)
        for i in range(1, self.max_iter + 1):
            if trace is True:
                start_time = t.time()
                w_next = self.w - (self.alpha) * self.my_oracle.grad(X, y, self.w)
                history['time'].append(t.time() - start_time)
                history['func'].append(self.my_oracle.func(X, y, w_next))
            else:
                w_next = self.w - (self.alpha) * self.my_oracle.grad(X, y, self.w)
            f = self.my_oracle.func(X, y, w_next)
            if f < F_best:
                F_best = f
                self.best = w_next.copy()
            if abs(self.my_oracle.func(X, y, self.best) - self.my_oracle.func(X, y, w_next)) < self.tol:
                self.w = w_next
                break
            else:
                self.w = w_next
        if trace is True:
            return history
        
    def predict(self, X):
        """
        Получение меток ответов на выборке X
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: одномерный numpy array с предсказаниями
        """
        return  2 * (X.dot(self.best) > 0) - 1 

            
    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: float
        """
        return self.my_oracle.func(X, y, self.w)
        
    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: numpy array, размерность зависит от задачи
        """
        return self.my_oracle.grad(X, y, self.w)
    
    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w


class SSubGDClassifier(SubGDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    
    def __init__(self, batch_size, step_alpha=1,
                 tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        - 'multinomial_logistic' - многоклассовая логистическая регрессия
        
        batch_size - размер подвыборки, по которой считается градиент
        
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если (f(x_{k+1}) - f(x_{k})) < tolerance: то выход 
        
        
        max_iter - максимальное число итераций
        
        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.
        
        **kwargs - аргументы, необходимые для инициализации
        """

        self.alpha = step_alpha
        self.tol = tolerance
        self.max_iter = max_iter
        self.kwargs = kwargs
        self.batch_size = batch_size
        self.seed = random_seed
        
    def fit(self, X, y, w_0=None, trace=False, log_freq=0.1):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
                
        w_0 - начальное приближение в методе
        
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет 
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}
        
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления. 
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.
        
        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        np.random.seed(self.seed)
        
        self.my_oracle = oracle.BinaryHinge(**self.kwargs)
        if w_0 is None:
            self.w = np.random.rand(X.shape[1])
        else:
            self.w = w_0
        if trace is True:
            history = {'epoch_num': [0], 'time': [0], 'func': [self.my_oracle.func(X, y, self.w)]}
        cur_epoch = prev_epoch = 0
        prev_w = self.w
        prev_t = t.time()
        permut = np.random.permutation(X.shape[0])
        cur_index = 0
        self.best = self.w.copy()
        F_best = self.my_oracle.func(X, y, self.best)
        for i in range(1, self.max_iter + 1):
            if (trace is True) and (cur_epoch - prev_epoch > log_freq):
                prev_epoch = cur_epoch
                history['epoch_num'].append(prev_epoch) 
                permut = np.random.permutation(X.shape[0])
                cur_index = 0
                w_next = (self.w - ((self.alpha) *
                          self.my_oracle.grad(X[permut[cur_index: cur_index + self.batch_size], :],
                                              y[permut[cur_index: cur_index + self.batch_size]], self.w)))
                cur_index += self.batch_size
                history['time'].append(t.time() - prev_t)
                prev_t = t.time()
                history['func'].append(self.my_oracle.func(X, y, self.w))
                f = self.my_oracle.func(X, y, w_next)
                print(f)
                if f < F_best:
                    F_best = f
                    self.best = w_next.copy()
                if abs(self.my_oracle.func(X, y, self.best) - self.my_oracle.func(X, y, prev_w)) < self.tol:
                    prev_w = self.w
                    self.w = w_next
                    break
                else:
                    prev_w = self.w 
                    self.w = w_next   
            else:
                self.w = (self.w - ((self.alpha) *
                          self.my_oracle.grad(X[permut[cur_index: cur_index + self.batch_size], :],
                                                y[permut[cur_index: cur_index + self.batch_size]], self.w)))
                cur_index += self.batch_size
                if (cur_index >= permut.shape[0]) or (cur_epoch - prev_epoch > log_freq):
                    cur_index = 0
                    prev_epoch = cur_epoch
                    permut = np.random.permutation(X.shape[0])
                f = self.my_oracle.func(X, y, self.w)
                if f < F_best:
                    F_best = f
                    self.best = self.w.copy()
                
            cur_epoch += self.batch_size / X.shape[0]
            
        if trace is True:
            return history