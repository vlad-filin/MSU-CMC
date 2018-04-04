import numpy as np
import scipy
from cvxopt import solvers
from cvxopt import matrix
from scipy.spatial.distance import cdist

class SVMSolver:
    """
    Класс с реализацией SVM через метод внутренней точки.
    """
    def __init__(self, C, method, kernel='linear', gamma=None, degree=None):
        """
        C - float, коэффициент регуляризации
        
        method - строка, задающая решаемую задачу, может принимать значения:
            'primal' - соответствует прямой задаче
            'dual' - соответствует двойственной задаче
        kernel - строка, задающая ядро при решении двойственной задачи
            'linear' - линейное
            'polynomial' - полиномиальное
            'rbf' - rbf-ядро
        gamma - ширина rbf ядра, только если используется rbf-ядро
        d - степень полиномиального ядра, только если используется полиномиальное ядро
        Обратите внимание, что часть функций класса используется при одном методе решения,
        а часть при другом
        """
        self.C = C
        self.method = method
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.w = None
        #if method != 'primal' and method != 'dual':
          #  raise ValueError("bad method:" + method)
       # if kernel != 'linear' and kernel != 'polynomial' and kernel != 'rbf':
         #   raise ValueError("bad kernel:" + kernel)
       # if kernel == 'polynomial' and degree is None:
            #raise ValueError("No degree for polynomial kernel")
        #if kernel == 'rbf' and gamma is None:
           # raise ValueError("No gamma for rbf kernel")
            
    
    def compute_primal_objective(self, X, y):
        """
        Метод для подсчета целевой функции SVM для прямой задачи
        
        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        """
        return 0.5 * self.w.dot(self.w.transpose()) + self.C / (X.shape[0] + 1) * np.sum(self.ksi)
        
    def compute_dual_objective(self, X, y):
        """
        Метод для подсчёта целевой функции SVM для двойственной задачи
        
        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        """ 
        tmp = y[:,np.newaxis].dot(y[np.newaxis,:])
        tmp_1 = self.lam.dot(self.lam.T)
        if self.kernel == 'linear':
            return -(np.sum(self.lam) - 0.5 * np.sum(X.dot(X.T) * tmp * tmp_1))
        elif self.kernel == 'rbf':
            ker = np.exp(-self.gamma * (cdist(X, X) ** 2))
            return -(np.sum(self.lam) - 0.5 * np.sum(ker * tmp * tmp_1))
        elif self.kernel == 'polynomial':
            ker =  (X.dot(X.T) + 1) ** self.degree
            return -(np.sum(self.lam) - 0.5 * np.sum(ker * tmp * tmp_1))
    def fit(self, X, y, tolerance=1e-7, max_iter=10000, tres=1e-3):
        """
        Метод для обучения svm согласно выбранной в method задаче
        
        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        tolerance - требуемая точность для метода обучения
        max_iter - максимальное число итераций в методе
        
        """
        # suppose x to be (w0, w, ksi)
        solvers.options['reltol'] = tolerance
        solvers.options['maxiters'] = max_iter
        solvers.options['show_progress'] = False
        if self.method != 'primal' and self.method != 'dual':
            raise TypeError("bad name" + method)
        if self.method == 'primal':
            P =  np.eye(1 + X.shape[0] + X.shape[1])
            P[0,0] = 0
            P[1+X.shape[1]:, 1+X.shape[1]:] = 0   
            q = np.zeros(1 + X.shape[0] + X.shape[1])
            q[X.shape[1] + 1:] = self.C / X.shape[0]
            P = matrix(P, tc='d')
            q = matrix(q, tc='d')
            G = np.hstack((-y[:,np.newaxis], - X * y[:, np.newaxis]))
            G = np.hstack((G,-1 * np.eye(X.shape[0])))
            tmp = np.hstack((np.zeros((X.shape[0], 1 +  X.shape[1])), -1 * np.eye(X.shape[0])))
            G = np.vstack((G, tmp))
            h = -np.ones(G.shape[0])
            h[X.shape[0]:] = 0
            G = matrix(G, tc='d')
            h = matrix(h, tc='d')
            solv = solvers.qp(P, q, G, h)
            self.w_0 = solv['x'][0]
            self.w = np.array(solv['x'][1:X.shape[1] + 1]).reshape(1, X.shape[1])
            self.ksi = np.array(solv['x'][X.shape[1] + 1:]).reshape(1, X.shape[0])
        elif self.method == 'dual':
            if self.kernel == 'linear':
                tmp = y[:,np.newaxis].dot(y[np.newaxis,:])
                P = X.dot(X.T) * tmp
            elif self.kernel == 'polynomial':
                tmp = y[:,np.newaxis].dot(y[np.newaxis,:])
                ker =  (X.dot(X.T) + 1) ** self.degree
                P = ker * tmp
            elif self.kernel == 'rbf':
                tmp = y[:,np.newaxis].dot(y[np.newaxis,:])
                ker = np.exp(-self.gamma * (cdist(X, X) ** 2))
                P = ker * tmp
            else:
                raise ValueError('bad kernel:' + self.kernel)
            q = -np.ones(X.shape[0])
            q = matrix(q, tc='d')
            P = matrix(P, tc='d')
            h = np.hstack((np.zeros(X.shape[0]),
                           (self.C / X.shape[0]) * np.ones(X.shape[0])))
            G = np.vstack((- 1 * np.eye(X.shape[0]), np.eye(X.shape[0])))
            G = matrix(G, tc='d')
            h = matrix(h, tc='d')
            A = y[np.newaxis,:]
            A = matrix(A, tc ='d')
            b = matrix(0.0, (1,1))
            solv = solvers.qp(P, q, G, h, A, b)
            self.lam = np.array(solv['x'])
            self.lam[self.lam < tres] = 0
            ind = np.where(self.lam !=0)[0]
            if self.kernel == 'linear':
                self.w = (self.lam * X * y[:, np.newaxis]).sum(axis=0)
                ind = np.where(self.lam !=0)[0]
                self.support = X[ind,:]
                self.y = y[ind]
                self.w_0 = np.mean((self.w.dot(X[ind,:].T).transpose()) + y[ind])
            else:
                self.support = X[ind,:]
                self.y = y[ind]
                self.lamd = self.lam[ind]
            
            
            
            
    def predict(self, X):
        """
        Метод для получения предсказаний на данных
        
        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        """
        if self.kernel == 'linear':
            return 2 * (X.dot(self.w.T).ravel() + self.w_0 >0) - 1
        elif self.kernel == 'rbf':
            ker = (np.exp(-self.gamma * (cdist(X, self.support) ** 2))).T
            return 2*((self.lamd * ker * self.y[:, np.newaxis]).sum(axis=0)>0)-1
        elif self.kernel == 'polynomial':
            tmp = self.support.dot(X.T)
            ker = (tmp + np.ones(tmp.shape)) ** self.degree
            return 2*((self.lamd * ker * self.y[:, np.newaxis]).sum(axis=0)>0)-1
            
        
    def get_w(self, X=None, y=None):
        """
        Получить прямые переменные (без учёта w_0)
        
        Если method = 'dual', а ядро линейное, переменные должны быть получены
        с помощью выборки (X, y) 
        
        return: одномерный numpy array
        """
        return self.w
        
    def get_w0(self, X=None, y=None):
        """
        Получить вектор сдвига
        
        Если method = 'dual', а ядро линейное, переменные должны быть получены
        с помощью выборки (X, y) 
        
        return: float
        """
        return self.w_0
        
    def get_dual(self):
        """
        Получить двойственные переменные
        
        return: одномерный numpy array
        """ 
        return self.lam