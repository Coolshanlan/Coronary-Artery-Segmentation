from functools import partial
from skimage import measure
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import math
connected_enable=True
def get_connected_num(y_true,y_pred):
    global connected_enable
    connected_enable=True
    y_true = y_true[0]
    y_pred = y_pred[0]
    loss_connected=0
    finalloss=[]
    beta=1
    for i in range(y_pred.shape[0]):
        _, num_true = measure.label(y_true[i] , connectivity=3,return_num=True)
        tf.print("\n T_N:",num_true)
        labels, num_pred = measure.label(y_pred[i] , connectivity=2, return_num=True)
        num_pred = 27 if  num_pred >= 27 else num_pred
        num_pred = 1 if (num_pred <=4 and num_pred != 0) else num_pred
        tf.print("\n T_N:",num_pred)
        if num_pred == 0:
            labels, num_pred = measure.label(y_true[i] , connectivity=2, return_num=True)
            if num_pred == 0:
                connected_enable=False##notwork
                return tf.convert_to_tensor([-1.0])
            else:
                return tf.convert_to_tensor([-0.0])
        #loss_connected = math.log(num_pred,6)-1
        loss_connected = math.log(num_pred,7)-math.log(num_pred,100)-1
        properties = measure.regionprops(label_image=labels)
        minarea =-1
        maxarea =-1
        sumarea=0
        for p in properties:
            tmp_area = p.bbox_area
            if minarea == -1 or minarea>tmp_area:
                minarea = tmp_area
            if minarea == -1 or maxarea<tmp_area:
                maxarea = tmp_area
            sumarea += tmp_area
        loss_area=0
        if minarea < 400:
            loss_area=-(minarea/400)
        else:
            loss_area=-1.0

        final_loss = (loss_area+loss_connected)/2
        #final_loss = ((1+beta**2)*loss_area*loss_connected)/((beta**2)*loss_area+loss_connected)
        finalloss.append(final_loss)

    return tf.convert_to_tensor(finalloss)
def connected_loss(y_true,y_pred):
    global connected_enable
    losslist=[]
    loss = tf.numpy_function(get_connected_num, [y_true,y_pred], tf.float32)
    tf.print("\n C:",loss[0])
    return tf.convert_to_tensor(loss)
def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    loss  = K.mean(2. * (K.sum(y_true * y_pred,axis=axis) + smooth/2)/(K.sum(y_true,axis=axis) + K.sum(y_pred,axis=axis) + smooth))
    #tf.print(" W:",loss,"\n\n")
    return loss


def weighted_dice_coefficient_loss(y_true, y_pred):
    beta=1 #>1 不重要
    #global connected_enable
    closs = connected_loss(y_true,y_pred)
    wloss =-weighted_dice_coefficient(y_true, y_pred)
    #if not connected_enable:
    #    return wloss

    return wloss + closs*0.2#((1+beta**2)*wloss*closs)/((beta**2)*wloss+closs)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
