
import time as t
import numpy as np
class PEGASOSMethod:
    """
    Реализация метода Pegasos для решения задачи svm.
    """
    def __init__(self, step_lambda, batch_size, num_iter):
        """
        step_lambda - величина шага, соответствует 
        
        batch_size - размер батча
        
        num_iter - число итераций метода, предлагается делать константное
        число итераций 
        """
        self.step_lambda = step_lambda
        self.batch_size = batch_size
        self.num_iter = num_iter
    def func(self, X, y, w):
        tmp = 1-(X.dot(w.T))*y
        hinge = np.where(tmp > 0, tmp, np.zeros(tmp.shape))
        return  np.mean(hinge) + 0.5 * self.step_lambda * w.dot(w.T)
    def fit(self, X, y, trace=False):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        trace - переменная типа bool
      
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        self.w = np.zeros(X.shape[1])
        self.best = np.zeros(X.shape[1])
        F_best = self.func(X, y, self.best)
        history = {'time':[], 'func':[0]}
        permut = np.random.permutation(X.shape[0])
        ind = 0
        for k in range(1, self.num_iter + 1):
            if (ind + 1) * self.batch_size >= X.shape[0]:
                permut = np.random.permutation(X.shape[0])
                ind = 0
            t_s = t.time()
            alpha_k = 1/(self.step_lambda * k)
            I_k = permut[ind: ind + self.batch_size]
            ind = ind + self.batch_size
            index = np.where(1-X.dot(self.w.T)*y <= 0)[0]
            tmp = X.copy()
            tmp[index, :] = 0
            w_next = (1 - 1/k) * self.w + (alpha_k / self.batch_size) *  np.sum(tmp[I_k,:] * y[I_k,np.newaxis], axis=0)
            self.w = max(1, 1/(self.step_lambda * w_next.dot(w_next.T)) ** 0.5) * w_next
            f = self.func(X,y, self.w)
            if f < F_best:
                F_best = f
                self.best = self.w
            if trace is True:
                history['time'].append(t.time() - t_s)
                history['func'].append(f)
                
        if trace is True:
            return history
                          
            
        
    def predict(self, X):
        """
        Получить предсказания по выборке X
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        """
        return  2 * (X.dot(self.best) > 0) - 1 