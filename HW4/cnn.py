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
    w = w.reshape((np.max(b.shape),np.max(w.shape)))
    x = x.reshape((np.max(x.shape),1))
    b = b.reshape((np.max(b.shape),1))

    y = w @ x+b

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

    #y = np.max(0, x)
    y = np.where(x>0,x,0)

    return y


def relu_backward(dl_dy, x, y):
    # TO DO
    # dLdy 1 x 10;  dLdx 1x10

    dy_dx = x

    dy_dx = np.where(x > 0,1,0)

    try:
        dl_dx = dl_dy * dy_dx
    except:
        dl_dx = np.zeros_like(dy_dx)
        for i in range(len(dl_dy)):
            dl_dx[:,:,i] = dl_dy[i]* dy_dx[:,:,i]
    #print(dl_dy.shape,dy_dx.shape,dl_dx.shape)
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
            w = np.flip(w_conv[:,:,id_i,id_j].reshape((w_conv_shape[0]*w_conv_shape[1])),axis=0)

            b = b_conv[:,id_j]

            #print(w.shape,X.shape,b_conv.shape)

            y[:,:,id_j] = y[:,:,id_j] +  (w @ X - b).reshape((x_shape[0],x_shape[1]))

    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO
    x_shape = x.shape
    w_conv_shape = w_conv.shape

    dl_dw = np.zeros_like(w_conv)
    dl_db = np.zeros_like(b_conv)

    for j in range(w_conv_shape[-1]):
        L = dl_dy[:,:,j].reshape((1,x_shape[0]*x_shape[1]))
        for i in range(x_shape[-1]):
            x_padding = np.pad(x[:,:,i], (1, 1), 'constant', constant_values=0)

            X = im2col_sliding_strided(x_padding, [x_shape[0],x_shape[1]])

            dl_dw_pre = (L @ X).reshape((w_conv_shape[0],w_conv_shape[1]))

            dl_dw[:,:,i,j] = dl_dw_pre

        dl_db[0, j] = np.sum(dl_dy[:,:,j])

    return dl_dw, dl_db

def pool2x2(x):
    # TO DO
    x_size = x.shape

    new_size = np.ceil(np.array(x_size)/2)

    y = np.zeros((np.int(new_size[0]),np.int(new_size[1]),x_size[-1]))

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
    x_size = x.shape

    new_size = np.ceil(np.array(x_size)/2)

    dl_dx_pre = np.zeros((np.int(2*new_size[0]),np.int(2*new_size[1]),x_size[-1]))

    for k in range(x_size[-1]):

        K = 2
        L = 2
        add_1 = np.int(x_size[0]%K)
        add_2 = np.int(x_size[1]%L)

        x_padding  = np.pad( x[:,:,k], ((0,add_1), (0,add_2)), 'constant', constant_values=0)

        end_1 = np.int(new_size[0]*K)
        end_2 = np.int(new_size[1]*L)

        x_pre = x_padding[:end_1, :end_2].reshape(np.int(new_size[0]), K, np.int(new_size[1]), L)
        x_temp = np.zeros_like(x_pre)

        for i in range(np.int(new_size[0])):
            for j in range(np.int(new_size[1])):
                x_temp[i,:,j,:] = np.where( x_pre[i,:,j,:] == np.max( x_pre[i,:,j,:]), dl_dy[i,j,k] , 0)

        dl_dx_pre[:,:,k] = x_temp.reshape((np.int(2*new_size[0]),np.int(2*new_size[1])))


    if (add_1!=0):
        if (add_2!=0):
            dl_dx = dl_dx_pre[:-1, :-1, :]
        else:
            dl_dx = dl_dx_pre[:-1, :, :]
    else:
        if (add_2!=0):
            dl_dx = dl_dx_pre[:, :-1, :]
        else:
            dl_dx = dl_dx_pre[:, :, :]

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

        loss_curve.append(L)

    loss_np = np.array(loss_curve)
    np.save("slp_linear_loss.npy", loss_np)
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

                y_tilde = fc(x, w, b)

                y = ((mini_batch_y[i])[j,:]).reshape((1,10))

                # loss
                loss, dl_dy = loss_cross_entropy_softmax(y_tilde, y)

                # dldy -- n*1
                dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y)

                dL_dw = dL_dw + dl_dw.reshape((10, 196))
                dL_db = dL_db + dl_db.reshape(b.shape)
                L = L + loss

            w = w - gamma/batch_size*dL_dw

            b = b - (gamma/batch_size*dL_db).reshape(b.shape)

        L = L / batch_size

        loss_curve.append(L)

    loss_np = np.array(loss_curve)
    np.save("slp_loss.npy", loss_np)
    sio.savemat('slp.mat', {
        'w': w ,
        'b': b
        })

    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    # TO DO
    # Loss = [];
    x_shape = (mini_batch_x[0]).shape
    y_shape = (mini_batch_y[0]).shape
    nBatch = len(mini_batch_x)
    batch_size = x_shape[1]
    print(y_shape)
    lr =1e-4  #1.77e-5
    lamda =0.96  #0.82

    #w1_size = [30, x_shape[0]]
    #w2_size = [y_shape[0], 30]
    #b1_size = [30, 1]
    #b2_size = [y_shape[0], 1]

    w1 = np.random.rand(30,196)
    w2 = np.random.rand(10, 30)
    b1 = np.random.rand(30, 1)
    b2 = np.random.rand(10, 1)
    nIter = 150
    #k = 1

    loss_curve = []

    #Training Loop
    for id_x in range(nIter):

        if (id_x % 15 == 0):
            print("Done",id_x/nIter)
        #print("Done",id_x)

        if ((id_x % 5) == 0):
            lr = lr * lamda
        dL_dw1 = np.zeros((1,np.size(w1)))
        dL_dw2 = np.zeros((1,np.size(w2)))
        dL_db1 = np.zeros((1,np.size(b1)))
        dL_db2 = np.zeros((1,np.size(b2)))

        for i in range(nBatch):
            L = 0

            for j in range(batch_size):
                x = ((mini_batch_x[i])[:,j]).reshape((1,196))
                a_1 = fc(x, w1, b1);     # a_1 = w * x + b
                f_1 = relu(a_1)
                a_2 = fc(f_1, w2, b2)    # a_2 = w * f_1 + b

                y = ((mini_batch_y[i])[j,:]).reshape((1,10))
                # loss
                loss, dl_da_2 = loss_cross_entropy_softmax(a_2, y)  #dldy -- n*1
                dl_df_1, dl_dw2, dl_db2 = fc_backward(dl_da_2, f_1, w2, b2, y)
                dl_da_1 = relu_backward(dl_df_1, a_1, f_1)
                [dl_dx, dl_dw1, dl_db1] = fc_backward(dl_da_1, x, w1, b1, a_1)


                dL_dw1 = dL_dw1 + dl_dw1
                dL_db1 = dL_db1 + dl_db1
                dL_dw2 = dL_dw2 + dl_dw2
                #print(dL_db2.shape , dl_db2.shape)
                dL_db2 = dL_db2 + dl_db2
                L = L + loss



            w1 = w1 - lr/batch_size*dL_dw1.reshape(w1.shape)
            b1 = b1 - lr/batch_size*dL_db1.reshape(b1.shape)
            w2 = w2 - lr/batch_size*dL_dw2.reshape(w2.shape)
            b2 = b2 - lr/batch_size*dL_db2.reshape(b2.shape)


        L = L / batch_size;

        loss_curve.append(L)

    loss_np = np.array(loss_curve)
    np.save("mlp_loss.npy", loss_np)
    sio.savemat('mlp.mat', {
        'w1': w1 ,
        'b1': b1 ,
        'w2': w2 ,
        'b2': b2
        })

    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    # TO DO
    x_shape = (mini_batch_x[0]).shape
    y_shape = (mini_batch_y[0]).shape
    nBatch = len(mini_batch_x)
    batch_size = x_shape[1]

    lr = 1e-5
    lamda = 0.89


    #w_conv_size = [3 3 1 3]
    #b_conv_size = [1 3]
    #w_fc_size = [10 147]
    #b_fc_size = [10 1]


    w_conv = np.random.rand(3,3,1,3)
    b_conv = np.random.rand(1,3)
    w_fc = np.random.rand(10,147)
    b_fc = np.random.rand(10,1)
    nIter = 15

    loss_curve = []
    for id_x in range(nIter):
        if ((id_x % 4) == 0):
            print("Done",id_x/nIter)

        if ((id_x % 6) == 0):
            lr = lr * lamda;


            dL_dw_conv = np.zeros(w_conv.shape)
            dL_db_conv = np.zeros(b_conv.shape)
            dL_dw_fc = np.zeros(w_fc.shape)
            dL_db_fc = np.zeros(b_fc.shape)

        for i in range(nBatch):

            for j in range(batch_size):
                x = ((mini_batch_x[i])[:,j]).reshape((14,14,1)) # input
                a_1 = conv(x, w_conv, b_conv)      # Conv
                #print(a_1.shape)
                f_1 = relu(a_1)                    # Relu
                f_2 = pool2x2(f_1)                # Pooling
                f_3 = flattening(f_2)             # Flatten

                a_2 = fc(f_3, w_fc, b_fc)        # FC
                y = ((mini_batch_y[i])[j,:]).reshape((1,10))
                # loss
                #if(np.random.rand(1)>0.9999):
                    #print(a_2,y)

                loss, dl_da_2 = loss_cross_entropy_softmax(a_2, y)  #dldy -- n*1
                dl_df_3, dl_dw_fc, dl_db_fc = fc_backward(dl_da_2, f_3, w_fc, b_fc, a_2)
                dL_dw_fc = dL_dw_fc + dl_dw_fc.reshape(w_fc.shape)
                dL_db_fc = dL_db_fc + dl_db_fc.reshape(dL_db_fc.shape)

                dl_df_2 = flattening_backward(dl_df_3, f_2, f_3)

                dl_df_1 = pool2x2_backward(dl_df_2, f_1, f_2)

                dl_da_1 = relu_backward(dl_df_1, a_1, f_1)
                #if(np.random.rand(1)>0.9999):
                    #print(dl_da_1,dl_da_1.shape)



                dl_dw_conv, dl_db_conv = conv_backward(dl_da_1, x, w_conv, b_conv, a_1)
                dL_dw_conv = dL_dw_conv + dl_dw_conv
                dL_db_conv = dL_db_conv + dl_db_conv


            w_conv = w_conv - lr/batch_size*dL_dw_conv
            b_conv = b_conv - lr/batch_size*dL_db_conv
            w_fc = w_fc - lr/batch_size*dL_dw_fc
            b_fc = b_fc - lr/batch_size*dL_db_fc
            #if(np.random.rand(1)>0.9):
                #print(loss, dl_da_2)

        print(loss, dl_da_2)

        loss_curve.append(loss/batch_size)





    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    main.main_slp_linear()
    #main.main_slp()
    #main.main_mlp()
    #main.main_cnn()
