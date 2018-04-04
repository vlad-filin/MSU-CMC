import sklearn.datasets as data
import numpy as np
def my_circles(samples=100, dim=2):
    # dim - 2/3 = mediana(chi^2(dim))
    X = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=(samples))
    R = np.diag(X.dot(X.T))
    y = np.ones(samples)
    y[R > (dim - 2/3)] = -1
    return  X,y
def create_data(samples=100, dim=2, disbalance=True, untypical=True):
    centers = 2 * np.ones((2, dim))
    centers[1:] = -2
    X_simple, y_simple = data.make_blobs(n_samples=samples, n_features=dim, centers=centers)
    y_simple[y_simple==0] = -1
    datasets = [(X_simple, y_simple)]
    X_NLinear, y_NLinear = my_circles(samples, dim)
    y_NLinear[y_NLinear==0] = -1
    datasets.append((X_NLinear, y_NLinear))
    centers = np.zeros((2,dim))
    tmp = np.random.randn(dim)
    centers[1,:] = tmp / (np.sum(tmp ** 2) ** 0.5)
    X_bad, y_bad = data.make_blobs(n_samples=samples, n_features=dim, centers=centers)
    y_bad[y_bad==0] = -1
    datasets.append((X_bad, y_bad))
    if disbalance is True:
        tmp = int(0.9 * samples)
        X0 = np.random.multivariate_normal(mean= 2 * np.ones(dim), cov=np.eye(dim), size=(tmp))
        y0 = -np.ones(tmp)
        X1 = np.random.multivariate_normal(mean= -2 * np.ones(dim), cov=np.eye(dim), size=(samples - tmp))
        y1 = np.ones(samples-tmp)
        X = np.vstack((X0,X1))
        y = np.hstack((y0, y1))
        y[y==0] = -1
        datasets.append((X, y))
    if untypical is True:
        size = int(0.3 * samples)
        centers = 2 * np.ones((2, dim))
        centers[1:] = -2
        X,y = data.make_blobs(n_samples=samples - size, n_features=dim, centers=centers)
        X_min = np.amin(X, axis=0)
        X_max = np.amax(X, axis=0)
        X_new = None
        for m1,m2 in zip(X_min, X_max):
            l = np.linspace(m1,m2, num=size)[:,np.newaxis]
            if X_new is None:
                X_new = l
            else:
                X_new = np.hstack((X_new, l))
        y_new = np.random.randint(0,2,size=(1,size)).ravel()
        X = np.vstack((X, X_new))
        y = np.hstack((y,y_new))
        y[y==0] = -1
        datasets.append((X, y))
    return datasets