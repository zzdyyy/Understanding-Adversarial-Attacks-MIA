from __future__ import absolute_import
from __future__ import print_function

import multiprocessing as mp
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale
import keras.backend as K
import tensorflow as tf
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.decomposition import PCA
from scipy.optimize import minimize

def lid_tle(XA, XB, k):
    """
    new Tight LID estimator: esstimate lid values for each sample in XA w.r.t. neighbors in XB.
    Paper: Intrinsic Dimensionality Estimation within Tight Localities∗ Laurent
    Link: https://epubs.siam.org/doi/pdf/10.1137/1.9781611975673.21
    :param XA: for which lid is estimated
    :param XB: neighbor candidate set
    :param k:
    :return:
    """
    epsilon = 0.0001
    XB = np.asarray(XB, dtype=np.float32)
    XA = np.asarray(XA, dtype=np.float32)

    if XB.ndim == 1:
        XB = XA.reshape((-1, XB.shape[0]))
    if XA.ndim == 1:
        XA = XA.reshape((-1, XA.shape[0]))

    k = min(k, XB.shape[0]-1)
    f = lambda v: v/v[-1]
    D = cdist(XA, XB)

    lids = np.zeros(XA.shape[0])

    for i in range(XA.shape[0]):
        x = XA[i, None]

        D = cdist(x, XB)
        D = D[0]
        '''
        x = np.array([[1,1]])
        XB = np.array([[1,1], [2,2], [3,3], [4,4], [5,5], [6,6]])
        D = array([[0., 1.41421356, 2.82842712, 4.24264069, 5.65685425, 7.07106781]])
        '''
        idx = np.argsort(D)[1:k+1] # indexes for k-nearest neighbors in XB:
        X = XB[idx] # k-nearest neighbor samples in XB
        dists = D[idx] # k-nearest distances
        r = dists[-1] # distance to k-th neighbor
        k = len(dists) # number of nearest neighbors
        '''
        idx = array([1, 2, 3])
        X = array([[2, 2], [3, 3], [4, 4]])
        dists = array([1.41421356, 2.82842712, 4.24264069])
        r = 4.242640687119285
        k = 3
        '''

        ## start TLE estimation process
        V = squareform(pdist(X)) # symmetric matrix representation of: pairwise distances between neighbors
        '''
        V = array([[0.        , 1.41421356, 2.82842712],
       [1.41421356, 0.        , 1.41421356],
       [2.82842712, 1.41421356, 0.        ]])
        '''

        Di = np.tile(dists.reshape((1, k)).T, (1, k))
        '''
        Di = array([[1.41421356, 1.41421356, 1.41421356],
       [2.82842712, 2.82842712, 2.82842712],
       [4.24264069, 4.24264069, 4.24264069]])
        '''
        Dj = Di.T
        '''
        Dj = array([[1.41421356, 2.82842712, 4.24264069],
       [1.41421356, 2.82842712, 4.24264069],
       [1.41421356, 2.82842712, 4.24264069]])
        '''
        Z2 = 2*Di**2 + 2*Dj**2 - V**2
        '''
        Z2 = array([[ 8., 18., 32.],
       [18., 32., 50.],
       [32., 50., 72.]])
        '''
        S = r * (((Di**2 + V**2 - Dj**2)**2 + 4*V**2 * (r**2 - Di**2))**0.5 - (Di**2 + V**2 - Dj**2)) / (2*(r**2 - Di**2) + 1e-12)
        T = r * (((Di**2 + Z2   - Dj**2)**2 + 4*Z2   * (r**2 - Di**2))**0.5 - (Di**2 + Z2   - Dj**2)) / (2*(r**2 - Di**2) + 1e-12)
        ''' 
        S = array([[0.        , 2.12132034, 4.24264069],
       [0.84852814, 0.        , 4.24264069],
       [0.        , 0.        , 0.        ]])
        T = array([[2.12132034, 3.18198052, 4.24264069],
       [2.54558441, 3.39411255, 4.24264069],
       [0.        , 0.        , 0.        ]])
        '''

        Dr = dists == r # handle case 1: repeating k-NN distances
        '''
        Dr = array([[False, False,  True]])
        '''
        Dr = Dr[0] # add this extra step in python to reduce dimension

        '''
        Dr = array([False, False,  True])
        '''
        S[Dr, :] = r * V[Dr, :]**2 / (r**2 + V[Dr, :]**2 - Dj[Dr, :]**2 + 1e-12)
        T[Dr, :] = r * Z2[Dr, :] / (r**2 + Z2[Dr, :] - Dj[Dr, :]**2 + 1e-12)
        '''
        S[Dr, :] = array([[1.41421356, 0.70710678, 0. ]])
        T[Dr, :] = array([[2.82842712, 3.53553391, 4.24264069]])
        '''

        ## Boundary case 2: If $u_i = 0$, then for all $1\leq j\leq k$ the measurements $s_{ij}$ and $t_{ij}$ reduce to $u_j$.
        Di0 = Di == 0
        '''
        Di0 = array([[False, False, False],
       [False, False, False],
       [False, False, False]])
        '''
        T[Di0] = Dj[Di0]
        S[Di0] = Dj[Di0]
        '''
        T = array([[2.12132034, 3.18198052, 4.24264069],
       [2.54558441, 3.39411255, 4.24264069],
       [2.82842712, 3.53553391, 4.24264069]])
       S = array([[0.        , 2.12132034, 4.24264069],
       [0.84852814, 0.        , 4.24264069],
       [1.41421356, 0.70710678, 0.        ]])
        '''

        ## Boundary case 3: If $u_j = 0$, then for all $1\leq j\leq k$ the measurements $s_{ij}$ and $t_{ij}$ reduce to $\frac{r v_{ij}}{r + v_{ij}}$.
        Dj0 = Dj == 0
        T[Dj0] = r * V[Dj0] / (r + V[Dj0] + 1e-12)
        S[Dj0] = r * V[Dj0] / (r + V[Dj0] + 1e-12)
        '''
        Dj0 = array([[False, False, False],
       [False, False, False],
       [False, False, False]])
       T[Dj0] = array([[2.12132034, 3.18198052, 4.24264069],
       [2.54558441, 3.39411255, 4.24264069],
       [2.82842712, 3.53553391, 4.24264069]])
       S[Dj0] = array([[0.        , 2.12132034, 4.24264069],
       [0.84852814, 0.        , 4.24264069],
       [1.41421356, 0.70710678, 0.        ]])
        '''

        ## Boundary case 4: If $v_{ij} = 0$, then the measurement $s_{ij}$ is zero and must be dropped. The measurement $t_{ij}$ should be dropped as well.
        V0 = V == 0
        '''
        V0 = array([[ True, False, False],
       [False,  True, False],
       [False, False,  True]])
        '''
        V0[np.eye(k).astype(bool)] = 0
        '''
        V0 = array([[False, False, False],
       [False, False, False],
       [False, False, False]])
       '''
        T[V0] = r # by setting to r, $t_{ij}$ will not contribute to the sum s1t
        S[V0] = r # by setting to r, $s_{ij}$ will not contribute to the sum s1s
        '''
        T = array([[2.12132034, 3.18198052, 4.24264069],
       [2.54558441, 3.39411255, 4.24264069],
       [2.82842712, 3.53553391, 4.24264069]])
       S = array([[0.        , 2.12132034, 4.24264069],
       [0.84852814, 0.        , 4.24264069],
       [1.41421356, 0.70710678, 0.        ]])
        '''
        nV0 = np.sum(V0[:]) # will subtract twice this number during ID computation below
        '''
        nV0 = 0
        '''
        ## Drop T & S measurements below epsilon (V4: If $s_{ij}$ is thrown out, then for the sake of balance, $t_{ij}$ should be thrown out as well (or vice versa).)
        TSeps = np.logical_or(T < epsilon, S < epsilon)
        '''
        TSeps = array([[ True, False, False],
       [False,  True, False],
       [False, False,  True]])'''
        TSeps[np.eye(k).astype(bool)] = 0
        '''
        TSeps = array([[False, False, False],
       [False, False, False],
       [False, False, False]])'''
        nTSeps = np.sum(TSeps[:])
        '''
        nTSeps = 0
        '''
        T[TSeps] = r
        '''
        T = array([[2.12132034, 3.18198052, 4.24264069],
       [2.54558441, 3.39411255, 4.24264069],
       [2.82842712, 3.53553391, 4.24264069]])
        '''
        T = np.log(T/r + 1e-12)
        '''
        T = array([[-6.93147181e-01, -2.87682072e-01,  9.68780611e-13],
       [-5.10825624e-01, -2.23143551e-01,  9.50128864e-13],
       [-4.05465108e-01, -1.82321557e-01,  9.86322135e-13]])
        '''
        S[TSeps] = r
        '''
        S = array([[0.        , 2.12132034, 4.24264069],
       [0.84852814, 0.        , 4.24264069],
       [1.41421356, 0.70710678, 0.        ]])
        '''
        S = np.log(S/r + 1e-12)
        '''
        S = array([[-2.76310211e+01, -6.93147181e-01,  9.68780611e-13],
       [-1.60943791e+00, -2.76310211e+01,  9.50128864e-13],
       [-1.09861229e+00, -1.79175947e+00, -2.76310211e+01]])
        '''

        T[np.eye(k).astype(bool)] = 0 # delete diagonal elements
        S[np.eye(k).astype(bool)] = 0
        '''
        T = array([[ 0.00000000e+00, -2.87682072e-01,  9.68780611e-13],
       [-5.10825624e-01,  0.00000000e+00,  9.50128864e-13],
       [-4.05465108e-01, -1.82321557e-01,  0.00000000e+00]])
       S = array([[ 0.00000000e+00, -6.93147181e-01,  9.68780611e-13],
       [-1.60943791e+00,  0.00000000e+00,  9.50128864e-13],
       [-1.09861229e+00, -1.79175947e+00,  0.00000000e+00]])
        '''

        ## Sum over the whole matrices
        s1t = np.sum(T[:])
        s1s = np.sum(S[:])
        '''
        s1t = -1.3862943611123903
        s1s = -5.192956850872497
        '''

        ## Drop distances below epsilon and compute sum
        '''
        dists = array([[1.41421356, 2.82842712, 4.24264069]])
        '''
        Deps = dists < epsilon
        nDeps = int(np.sum(Deps))
        '''
        Deps = array([False, False, False])
        nDeps = 0
        '''
        dists = dists[nDeps:] # python index start from 0
        '''
         dists = array([1.41421356, 2.82842712, 4.24264069])
        '''
        s2 = np.sum(np.log(dists/r + 1e-12))
        '''
        s2 = -1.504077396770774
        '''

        ## Compute ID, subtracting numbers of dropped measurements
        lid = -2*(k**2 - nTSeps - nDeps - nV0) / (s1t + s1s + 2*s2)
        '''
        lid = 1.8774629956866669
        '''
        lids[i] = lid

    return lids


def lid_term(logits, batch_size=100):
    """Calculate LID loss term for a minibatch of logits
    :param logits:
    :return:
    """
    # y_pred = tf.nn.softmax(logits)
    y_pred = logits

    # calculate pairwise distance
    r = tf.reduce_sum(tf.square(y_pred), axis=1)
    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    D = r - 2 * tf.matmul(y_pred, tf.transpose(y_pred)) + tf.transpose(r)

    # find the k nearest neighbor
    D1 = tf.sqrt(D + 1e-9)
    D2, _ = tf.nn.top_k(-D1, k=21, sorted=True)
    D3 = -D2[:, 1:]

    m = tf.transpose(tf.multiply(tf.transpose(D3), 1.0 / D3[:, -1]))
    v_log = tf.reduce_sum(tf.log(m + 1e-9), axis=1)  # to avoid nan
    lids = -20 / v_log

    ## batch normalize lids
    # lids = tf.nn.l2_normalize(lids, dim=0, epsilon=1e-12)

    return lids

def lid_adv_term(clean_logits, adv_logits, batch_size=100):
    """Calculate LID loss term for a minibatch of advs logits
    :param logits: clean logits
    :param A_logits: adversarial logits
    :return:
    """
    # y_pred = tf.nn.softmax(logits)
    c_pred = tf.reshape(clean_logits, (batch_size, -1))
    a_pred = tf.reshape(adv_logits, (batch_size, -1))

    # calculate pairwise distance
    r_a = tf.reduce_sum(tf.square(a_pred), axis=1)
    # turn r_a into column vector
    r_a = tf.reshape(r_a, [-1, 1])

    r_c = tf.reduce_sum(tf.square(c_pred), axis=1)
    # turn r_c into row vector
    r_c = tf.reshape(r_c, [1, -1])

    D = r_a - 2 * tf.matmul(a_pred, tf.transpose(c_pred)) + r_c

    # find the k nearest neighbor
    D1 = tf.sqrt(D + 1e-9)
    D2, _ = tf.nn.top_k(-D1, k=21, sorted=True)
    D3 = -D2[:, 1:]

    m = tf.transpose(tf.multiply(tf.transpose(D3), 1.0 / D3[:, -1]))
    v_log = tf.reduce_sum(tf.log(m + 1e-9), axis=1)  # to avoid nan
    lids = -20 / v_log

    ## batch normalize lids
    lids = tf.nn.l2_normalize(lids, dim=0, epsilon=1e-12)

    return lids


def get_mc_predictions(model, X, nb_iter=50, batch_size=256):
    """
    TODO
    :param model:
    :param X:
    :param nb_iter:
    :param batch_size:
    :return:
    """
    output_dim = model.layers[-1].output.shape[-1].value
    get_output = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[-1].output]
    )

    def predict():
        n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
        output = np.zeros(shape=(len(X), output_dim))
        for i in range(n_batches):
            output[i * batch_size:(i + 1) * batch_size] = \
                get_output([X[i * batch_size:(i + 1) * batch_size], 1])[0]
        return output

    preds_mc = []
    for i in tqdm(range(nb_iter)):
        preds_mc.append(predict())

    return np.asarray(preds_mc)


def get_deep_representations(model, X, index=-4, batch_size=256):
    """
    TODO
    :param model:
    :param X:
    :param batch_size:
    :return:
    """
    # last hidden layer is always at index 'index'
    output_dim = model.layers[index].output.shape[-1].value
    get_encoding = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[index].output]
    )

    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    output = np.zeros(shape=(len(X), output_dim))
    for i in range(n_batches):
        output[i * batch_size:(i + 1) * batch_size] = \
            get_encoding([X[i * batch_size:(i + 1) * batch_size], 0])[0]

    return output

def get_layer_wise_activations(model, dataset):
    """
    Get the deep activation outputs.
    :param model:
    :param dataset: 'mnist', 'cifar', 'svhn', has different submanifolds architectures  
    :return: 
    """
    assert dataset in ['mnist', 'cifar', 'svhn', 'dr', 'cxr', 'derm'], \
        "dataset parameter must be either 'mnist' 'cifar' or 'svhn'"

    if dataset in ['dr', 'cxr', 'derm']:
        acts = [model.layers[-1].input]
    elif dataset == 'mnist':
        # mnist model
        # acts = [model.layers[0].input] +
        acts = [layer.output for layer in model.layers[:]]
    elif dataset == 'cifar-10':
        # cifar-10 model
        # acts = [model.layers[0].input] +
        acts = [layer.output for layer in model.layers[:]]
    else:
        # svhn model
        # acts = [model.layers[0].input] +
        acts = [layer.output for layer in model.layers[:]]
    return acts

# lid of a single query point x
def mle_single(data, x, k=20):
    data = np.asarray(data, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    # print('x.ndim',x.ndim)
    if x.ndim == 1:
        x = x.reshape((-1, x.shape[0]))
    # dim = x.shape[1]

    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))
    a = cdist(x, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a[0]

# lid of a batch of query points X
def mle_batch(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

def mle_q(lid, d_i, k, q):
    sum = 0
    for p in range(0, k):
        term = (lid*(d_i[p])**(lid-1))**(1-q)
        sum = sum+term
    objective = -sum/(1-q) #take the negative,
    #since we will use a solver that minimizes rather than maximizes
    return objective

def mle_q_batch(XA, XB, k, q):
    XA = np.asarray(XA, dtype=np.float32)
    XB = np.asarray(XB, dtype=np.float32)

    if XA.ndim == 1:
        XA = XA.reshape((-1, XA.shape[0]))
    if XB.ndim == 1:
        XB = XB.reshape((-1, XB.shape[0]))

    k = min(k, XB.shape[0]-1)
    f = lambda v: v/v[-1]
    D = cdist(XA, XB)
    D_k = np.apply_along_axis(np.sort, axis=1, arr=D)[:, 1:k+1]
    D_r = np.apply_along_axis(f, axis=1, arr=D_k)

    f = lambda v: - k / np.sum(np.log(v))
    lids = np.apply_along_axis(f, axis=1, arr=D_r)

    if q <= 1.0:
        return lids
    else:
        lids_q = []
        for i in range(D_r.shape[0]):
            lid_0 = lids[i]
            lid_opt = minimize(mle_q, lid_0, args=(D_r[i], k, q), method='Nelder-Mead', tol=1e-2)
            lids_q.append(lid_opt.x[0])
        return np.array(lids_q)


# mean distance of x to its k nearest neighbours
def kmean_batch(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    f = lambda v: np.mean(v)
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

# mean distance of x to its k nearest neighbours
def kmean_pca_batch(data, batch, k=10):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)
    a = np.zeros(batch.shape[0])
    for i in np.arange(batch.shape[0]):
        tmp = np.concatenate((data, [batch[i]]))
        tmp_pca = PCA(n_components=2).fit_transform(tmp)
        a[i] = kmean_batch(tmp_pca[:-1], tmp_pca[-1], k=k)
    return a


def get_lids_random_batch(model, X, X_adv, dataset, k=10, q=1.0, batch_size=100):
    """
    Get the local intrinsic dimensionality of each Xi in X_adv
    estimated by k close neighbours in the random batch it lies in.
    :param model:
    :param X: normal images
    :param X_noisy: noisy images
    :param X_adv: advserial images
    :param dataset: 'mnist', 'cifar', 'svhn', has different DNN architectures
    :param k: the number of nearest neighbours for LID estimation
    :param batch_size: default 100
    :return: lids: LID of normal images of shape (num_examples, lid_dim)
            lids_adv: LID of advs images of shape (num_examples, lid_dim)
    """
    # get deep representations
    funcs = [K.function([model.layers[0].input, K.learning_phase()], [out])
                 for out in get_layer_wise_activations(model, dataset)]
    lid_dim = len(funcs)
    print("Number of layers to estimate: ", lid_dim)

    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(X), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch = np.zeros(shape=(n_feed, lid_dim))
        lid_batch_adv = np.zeros(shape=(n_feed, lid_dim))
        lid_batch_noisy = np.zeros(shape=(n_feed, lid_dim))
        for i, func in enumerate(funcs):
            X_act = func([X[start:end], 0])[0]
            X_act = np.asarray(X_act, dtype=np.float32).reshape((n_feed, -1))
            # print("X_act: ", X_act.shape)

            X_adv_act = func([X_adv[start:end], 0])[0]
            X_adv_act = np.asarray(X_adv_act, dtype=np.float32).reshape((n_feed, -1))
            # print("X_adv_act: ", X_adv_act.shape)

            # X_noisy_act = func([X_noisy[start:end], 0])[0]
            # X_noisy_act = np.asarray(X_noisy_act, dtype=np.float32).reshape((n_feed, -1))
            # print("X_noisy_act: ", X_noisy_act.shape)

            # random clean samples
            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            # lid_batch[:, i] = mle_q_batch(X_act, X_act, k=k, q=q)
            lid_batch[:, i] = lid_tle(X_act, X_act, k=k)
            # print("lid_batch: ", lid_batch.shape)
            # lid_batch_adv[:, i] = mle_q_batch(X_adv_act, X_act, k=k, q=q)
            lid_batch_adv[:, i] = lid_tle(X_adv_act, X_act, k=k)
            # print("lid_batch_adv: ", lid_batch_adv.shape)
            # lid_batch_noisy[:, i] = mle_q_batch(X_noisy_act, X_act, k=k, q=q)
            # lid_batch_noisy[:, i] = lid_tle(X_noisy_act, X_act, k=k)
            # print("lid_batch_noisy: ", lid_batch_noisy.shape)
        return lid_batch, lid_batch_adv

    lids = []
    lids_adv = []
    # lids_noisy = []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    for i_batch in tqdm(range(n_batches)):
        lid_batch, lid_batch_adv = estimate(i_batch)
        lids.extend(lid_batch)
        lids_adv.extend(lid_batch_adv)
        # lids_noisy.extend(lid_batch_noisy)
        # print("lids: ", lids.shape)
        # print("lids_adv: ", lids_noisy.shape)
        # print("lids_noisy: ", lids_noisy.shape)

    lids = np.asarray(lids, dtype=np.float32)
    # lids_noisy = np.asarray(lids_noisy, dtype=np.float32)
    lids_adv = np.asarray(lids_adv, dtype=np.float32)

    return lids, lids_adv

def get_kmeans_random_batch(model, X, X_noisy, X_adv, dataset, k=10, batch_size=100, pca=False):
    """
    Get the mean distance of each Xi in X_adv to its k nearest neighbors.
    :param model:
    :param X: normal images
    :param X_noisy: noisy images
    :param X_adv: advserial images
    :param dataset: 'mnist', 'cifar', 'svhn', has different DNN architectures
    :param k: the number of nearest neighbours for LID estimation
    :param batch_size: default 100
    :param pca: using pca or not, if True, apply pca to the referenced sample and a
            minibatch of normal samples, then compute the knn mean distance of the referenced sample.
    :return: kms_normal: kmean of normal images (num_examples, 1)
            kms_noisy: kmean of normal images (num_examples, 1)
            kms_adv: kmean of adv images (num_examples, 1)
    """
    # get deep representations
    funcs = [K.function([model.layers[0].input, K.learning_phase()], [model.layers[-2].output])]
    km_dim = len(funcs)
    print("Number of layers to use: ", km_dim)

    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(X), (i_batch + 1) * batch_size)
        n_feed = end - start
        km_batch = np.zeros(shape=(n_feed, km_dim))
        km_batch_adv = np.zeros(shape=(n_feed, km_dim))
        km_batch_noisy = np.zeros(shape=(n_feed, km_dim))
        for i, func in enumerate(funcs):
            X_act = func([X[start:end], 0])[0]
            X_act = np.asarray(X_act, dtype=np.float32).reshape((n_feed, -1))
            # print("X_act: ", X_act.shape)

            X_adv_act = func([X_adv[start:end], 0])[0]
            X_adv_act = np.asarray(X_adv_act, dtype=np.float32).reshape((n_feed, -1))
            # print("X_adv_act: ", X_adv_act.shape)

            X_noisy_act = func([X_noisy[start:end], 0])[0]
            X_noisy_act = np.asarray(X_noisy_act, dtype=np.float32).reshape((n_feed, -1))
            # print("X_noisy_act: ", X_noisy_act.shape)

            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            if pca:
                km_batch[:, i] = kmean_pca_batch(X_act, X_act, k=k)
            else:
                km_batch[:, i] = kmean_batch(X_act, X_act, k=k)
            # print("lid_batch: ", lid_batch.shape)
            if pca:
                km_batch_adv[:, i] = kmean_pca_batch(X_act, X_adv_act, k=k)
            else:
                km_batch_adv[:, i] = kmean_batch(X_act, X_adv_act, k=k)
            # print("lid_batch_adv: ", lid_batch_adv.shape)
            if pca:
                km_batch_noisy[:, i] = kmean_pca_batch(X_act, X_noisy_act, k=k)
            else:
                km_batch_noisy[:, i] = kmean_batch(X_act, X_noisy_act, k=k)
                # print("lid_batch_noisy: ", lid_batch_noisy.shape)
        return km_batch, km_batch_noisy, km_batch_adv

    kms = []
    kms_adv = []
    kms_noisy = []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    for i_batch in tqdm(range(n_batches)):
        km_batch, km_batch_noisy, km_batch_adv = estimate(i_batch)
        kms.extend(km_batch)
        kms_adv.extend(km_batch_adv)
        kms_noisy.extend(km_batch_noisy)
        # print("kms: ", kms.shape)
        # print("kms_adv: ", kms_noisy.shape)
        # print("kms_noisy: ", kms_noisy.shape)

    kms = np.asarray(kms, dtype=np.float32)
    kms_noisy = np.asarray(kms_noisy, dtype=np.float32)
    kms_adv = np.asarray(kms_adv, dtype=np.float32)

    return kms, kms_noisy, kms_adv

def score_point(tup):
    """
    TODO
    :param tup:
    :return:
    """
    x, kde = tup

    return kde.score_samples(np.reshape(x, (1, -1)))[0]


def score_samples(kdes, samples, preds, n_jobs=None):
    """
    TODO
    :param kdes:
    :param samples:
    :param preds:
    :param n_jobs:
    :return:
    """
    if n_jobs is not None:
        p = mp.Pool(n_jobs)
    else:
        p = mp.Pool()
    results = np.asarray(
        p.map(
            score_point,
            [(x, kdes[i]) for x, i in zip(samples, preds)]
        )
    )
    p.close()
    p.join()

    return results


def normalize(normal, adv, noisy):
    """Z-score normalisation
    TODO
    :param normal:
    :param adv:
    :param noisy:
    :return:
    """
    n_samples = len(normal)
    total = scale(np.concatenate((normal, adv, noisy)))

    return total[:n_samples], total[n_samples:2*n_samples], total[2*n_samples:]


def train_lr(X, y):
    """
    TODO
    :param X: the data samples
    :param y: the labels
    :return:
    """
    lr = LogisticRegressionCV(n_jobs=-1, cv=5, max_iter=1000).fit(X, y)
    return lr


def train_lr_rfeinman(densities_pos, densities_neg, uncerts_pos, uncerts_neg):
    """
    TODO
    :param densities_pos:
    :param densities_neg:
    :param uncerts_pos:
    :param uncerts_neg:
    :return:
    """
    values_neg = np.concatenate(
        (densities_neg.reshape((1, -1)),
         uncerts_neg.reshape((1, -1))),
        axis=0).transpose([1, 0])
    values_pos = np.concatenate(
        (densities_pos.reshape((1, -1)),
         uncerts_pos.reshape((1, -1))),
        axis=0).transpose([1, 0])

    values = np.concatenate((values_neg, values_pos))
    labels = np.concatenate(
        (np.zeros_like(densities_neg), np.ones_like(densities_pos)))

    lr = LogisticRegressionCV(n_jobs=-1).fit(values, labels)

    return values, labels, lr


def compute_roc(y_true, y_pred, plot=False):
    """
    TODO
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot:
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score


def compute_roc_rfeinman(probs_neg, probs_pos, plot=False):
    """
    TODO
    :param probs_neg:
    :param probs_pos:
    :param plot:
    :return:
    """
    probs = np.concatenate((probs_neg, probs_pos))
    labels = np.concatenate((np.zeros_like(probs_neg), np.ones_like(probs_pos)))
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score

def random_split(X, Y):
    """
    Random split the data into 80% for training and 20% for testing
    :param X: 
    :param Y: 
    :return: 
    """
    print("random split 80%, 20% for training and testing")
    num_samples = X.shape[0]
    num_train = int(num_samples * 0.8)
    rand_pert = np.random.permutation(num_samples)
    X = X[rand_pert]
    Y = Y[rand_pert]
    X_train, X_test = X[:num_train], X[num_train:]
    Y_train, Y_test = Y[:num_train], Y[num_train:]

    return X_train, Y_train, X_test, Y_test

def block_split(X, Y):
    """
    Split the data into 80% for training and 20% for testing
    in a block size of 100.
    :param X: 
    :param Y: 
    :return: 
    """
    print("Isolated split 80%, 20% for training and testing")
    num_samples = X.shape[0]
    partition = int(num_samples / 3)
    X_adv, Y_adv = X[:partition], Y[:partition]
    X_norm, Y_norm = X[partition: 2*partition], Y[partition: 2*partition]
    X_noisy, Y_noisy = X[2*partition:], Y[2*partition:]
    num_train = int(partition*0.008) * 100

    X_train = np.concatenate((X_norm[:num_train], X_adv[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_adv[:num_train]))

    X_test = np.concatenate((X_norm[num_train:], X_adv[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_adv[num_train:]))

    # X_train = np.concatenate((X_norm[:num_train], X_noisy[:num_train], X_adv[:num_train]))
    # Y_train = np.concatenate((Y_norm[:num_train], Y_noisy[:num_train], Y_adv[:num_train]))
    #
    # X_test = np.concatenate((X_norm[num_train:], X_noisy[num_train:], X_adv[num_train:]))
    # Y_test = np.concatenate((Y_norm[num_train:], Y_noisy[num_train:], Y_adv[num_train:]))

    return X_train, Y_train, X_test, Y_test


def local_shuffle(image):
    """
    Shuffle the local path of image.
    :return:
    """
    width = image.shape[0]
    height = image.shape[1]
    channel = image.shape[2]
    patch_size = 4

    img_new = np.zeros_like(image)
    for i in range(0, width, patch_size):
        i_e = np.minimum(i + patch_size, width)
        for j in range(0, height, patch_size):
            j_e = np.minimum(j + patch_size, height)
            patch = image[i:i_e, j:j_e, :]
            w = patch.shape[0]
            h = patch.shape[1]
            c = patch.shape[2]
            patch = patch.reshape(-1, c)
            idxes = np.arange(start=0, stop=patch.shape[0], step=1)
            np.random.shuffle(idxes)
            patch = patch[idxes, :]
            patch_rot = patch.reshape(w, h, c)
            img_new[i:i_e, j:j_e, :] = patch_rot
    return img_new


if __name__ == "__main__":
    # unit test
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([6, 7, 8, 9, 10])
    c = np.array([11, 12, 13, 14, 15])

    a_z, b_z, c_z = normalize(a, b, c)
    print(a_z)
    print(b_z)
    print(c_z)
