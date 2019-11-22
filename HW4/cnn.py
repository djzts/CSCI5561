import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main


def get_mini_batch(im_train, label_train, batch_size):
    # TO DO
    label_train = label_train[0]
    #print(label_train)
    image_vec, n_train = im_train.shape   #Decompose the im

    batch_perm = np.random.permutation(n_train)

    n_batch = np.int(n_train//batch_size)

    mini_batch_x = []
    mini_batch_y = []

    for i in range(n_batch):

        end_id = np.min([i*batch_size+batch_size,n_train])
        id_x =batch_perm[np.arange(i*batch_size,end_id,1)]

        mini_batch_x.append(im_train[:, id_x])

        num_class = 10
        targets = np.array([label_train[id_x]]).reshape(-1)
        one_hot_targets = np.eye(num_class)[targets]

        mini_batch_y.append(one_hot_targets)
    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    # TO DO
    # w:batch*196    # x:196*1    # b:batch*1

    y = w @ x.T+b

    y = y.reshape(1,-1)

    return y


def fc_backward(dl_dy, x, w, b, y):
    # TO DO

    dl_dx = w.T @ dl_dy[0,:]  # 196 * batch

    dl_dw = np.outer(dl_dy , x).reshape((1,np.size(w)))

    dl_db = dl_dy

    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    # TO DO


    l = np.linalg.norm((y_tilde-y),2)**2

    dl_dy = 2 * (y_tilde- y)

    return l, dl_dy

def loss_cross_entropy_softmax(x, y):
    # TO DO

    x_tilde = np.exp(x)

    x_sum = np.sum(x_tilde)

    y_tilde = x_tilde / x_sum

    l = - np.sum(y * np.log(y_tilde) )

    dl_dy = y_tilde - y

    return l, dl_dy

def relu(x):
    # TO DO

    y = np.max(0, x)

    return y


def relu_backward(dl_dy, x, y):
    # TO DO
    # dLdy 1 x 10;  dLdx 1x10

    dy_dx = x

    dy_dx = np.where(x > 0,1,0)

    dLdx = dLdy * dydx;

    return dl_dx

def im2col_sliding_strided(A, BSZ, stepsize=1):
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]

def conv(x, w_conv, b_conv):
    # TO DO
    x_shape = x.shape
    w_conv_shape = w_conv.shape

    y = np.zeros((x_shape[0],x_shape[1],w_conv_shape[-1]))

    for id_i in range(x_shape[-1]):
        x_padding = np.pad(x[:,:,id_i], (1, 1), 'constant', constant_values=0)

        X = im2col_sliding_strided(x_padding, [3,3])

        for id_j in range(w_conv_shape[-1]):
            w = np.flip(w_conv[:,:,id_i,id_j].reshape((1,w_conv_shape[0]*w_conv_shape[1])))

            b = b_conv[id-j]

            y[:,:,id_j] = y[:,:,id_j] +  (w @ X - b).reshape((x_shape[0],x_shape[1]))

    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO
    x_shape = x.shape
    w_conv_shape = w_conv.shape

    dl_dw = np.zeros_like(w_conv)
    dl_db = np.zeros_like(b_conv)

    for j in range(w_conv_shape[-1])
        L = dl_dy[:,:,j].reshape((1,x_shape[0]*x_shape[1]))
        for i in range(x_shape[-1]):
            x_padding = np.pad(x[:,:,id_i], (1, 1), 'constant', constant_values=0)

            X = im2col_sliding_strided(x_padding, [x_shape[0],x_shape[1]])

            dl_dw_pre = (L @ X).reshape((w_conv_shape[0],w_conv_shape[1]))

            dl_dw[:,:,i,j] = dl_dw_pre

        dl_db[1, jj] = np.sum(dl_dy[:,:,j])

    return dl_dw, dl_db

def pool2x2(x):
    # TO DO
    x_size = x.shape

    new_size = np.ceil(x_size/2)

    y = np.zeros((new_size[0],new_size[1],x_size[-1]))

    for i in range(x_size[-1]):

        x_padding  = np.pad(x[:,:,i], ((0,x_size[0]%2), (0,x_size[1]%2)), 'constant', constant_values=0)

        M, N = x_padding.shape
        K = 2
        L = 2

        MK = M // K
        NL = N // L

        y[:,:,i] = x_padding[:MK*K, :NL*L].reshape(MK, K, NL, L).max(axis=(1, 3))

    return y

def pool2x2_backward(dl_dy, x, y):
    # TO DO
    x_size = size(x)

    new_size = np.ceil(x_size/2)

    dl_dx_pre = np.zeros((2*new_size[0],2*new_size[1],x_size[-1]))

    for k = 1:C1

        K = 2
        L = 2

        x_padding  = np.pad(x[:,:,k], ((0,x_size[0]%K), (0,x_size[1]%L)), 'constant', constant_values=0)

        x_pre = x_padding[:new_size[0]*K, :new_size[1]*L].reshape(new_size[0], K, new_size[1], L)
        x_temp = np.zero_like(x_pre)

        for i in range(new_size[0]):
            for j in range(new_size[1]):
                x_temp[i,:,j,:] = np.where( x_pre[i,:,j,:] == np.max( x_pre[i,:,j,:]), dl_dy[i,j,k] , 0)

        dl_dx_pre[:,:,k] = x_temp.reshape((2*new_size[0],2*new_size[1]))
    dl_dx = dl_dx_pre[:H, :W, :]

    return dl_dx


def flattening(x):
    # TO DO
    y = x.reshape((np.size(x),1))
    return y


def flattening_backward(dl_dy, x, y):
    # TO DO
    dl_dx = dl_dy.reshape(x.shape)
    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    # TO DO
    # learning rate
    gamma = 0.0001

    # decay rate
    lamda = 0.985

    # init weight
    w = np.random.rand(10,196)
    b =  np.random.rand(10, 1)

    # number of iteration
    n_Iter = 2000

    # number of minibatch
    nBatch = len(mini_batch_x)

    loss_curve = []

    for id_x in range(n_Iter):
        if ((id_x % 10) == 0):
            gamma = gamma * lamda

        dL_dw = np.zeros_like(w)
        dL_db = np.zeros_like(b)
        #print(dL_db.s)
        if (id_x%80 == 0):
            print("Done",id_x/n_Iter)

        for i in range(nBatch):
            L = 0
            batch_size = len((mini_batch_x[i]).T)


            for j in range(batch_size):

                x = ((mini_batch_x[i])[:,j]).reshape((1,196))
                y_tilde = fc(x, w, b);
                y = ((mini_batch_y[i])[j,:]).reshape((1,10))

                # loss
                loss, dl_dy = loss_euclidean(y_tilde, y)

                # dldy -- n*1
                dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y)

                dL_dw = dL_dw + dl_dw.reshape((10, 196))
                dL_db = dL_db + dl_db.reshape(b.shape)
                L = L + loss

            w = w - gamma/batch_size*dL_dw

            b = b - (gamma/batch_size*dL_db).reshape(b.shape)

            L = L / batch_size

        loss_curve.append(loss)

    loss_np = np.array(loss_curve)
    np.save("slp_loss.npy", loss_np)
    sio.savemat('slp_linear.mat', {
        'w': w ,
        'b': b
        })
    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    # TO DO
    # TO DO
    # learning rate
    gamma = 0.0001

    # decay rate
    lamda = 0.96

    # init weight
    w = np.random.rand(10,196)
    b =  np.random.rand(10, 1)

    # number of iteration
    n_Iter = 2000

    # number of minibatch
    nBatch = len(mini_batch_x)

    loss_curve = []

    for id_x in range(n_Iter):
        if ((id_x % 20) == 0):
            gamma = gamma * lamda

        dL_dw = np.zeros_like(w)
        dL_db = np.zeros_like(b)


        if (id_x%80 == 0):
            print("Done",id_x/n_Iter)

        for i in range(nBatch):
            L = 0
            batch_size = len((mini_batch_x[i]).T)


            for j in range(batch_size):

                x = ((mini_batch_x[i])[:,j]).reshape((1,196))
                y_tilde = fc(x, w, b);
                y = ((mini_batch_y[i])[j,:]).reshape((1,10))

                # loss
                loss, dl_dy = loss_euclidean(y_tilde, y)

                # dldy -- n*1
                dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y)

                dL_dw = dL_dw + dl_dw.reshape((10, 196))
                dL_db = dL_db + dl_db.reshape(b.shape)
                L = L + loss

            w = w - gamma/batch_size*dL_dw

            b = b - (gamma/batch_size*dL_db).reshape(b.shape)

            L = L / batch_size

        loss_curve.append(loss)

    loss_np = np.array(loss_curve)
    np.save("slp_loss.npy", loss_np)
    sio.savemat('slp_linear.mat', {
        'w': w ,
        'b': b
        })
    % learning rate

    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    # TO DO
    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    # TO DO
    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()
