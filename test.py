import time
import numpy as np
import tensorflow as tf
from data import *
from model import *
from utils import *
import matplotlib.pyplot as plt

np.random.seed(0)
sd = 16
# Data
tf.app.flags.DEFINE_string('input', 'data/gridworld_{}.mat'.format(sd), 'Path to data')
tf.app.flags.DEFINE_integer('imsize', sd, 'Size of input image')
# Parameters
tf.app.flags.DEFINE_float('lr', 0.001, 'Learning rate for RMSProp')
tf.app.flags.DEFINE_integer('epochs', 50, 'Maximum epochs to train for')
tf.app.flags.DEFINE_integer('k', 20, 'Number of value iterations')
tf.app.flags.DEFINE_integer('ch_i', 2, 'Channels in input layer')
tf.app.flags.DEFINE_integer('ch_h', 150, 'Channels in initial hidden layer')
tf.app.flags.DEFINE_integer('ch_q', 10, 'Channels in q layer (~actions)')
tf.app.flags.DEFINE_integer('batchsize', 12, 'Batch size')
tf.app.flags.DEFINE_integer('statebatchsize', 10, 'Number of state inputs for each sample (real number, technically is k+1)')
tf.app.flags.DEFINE_boolean('untied_weights', False, 'Untie weights of VI network')
# Misc.
tf.app.flags.DEFINE_integer('display_step', 1, 'Print summary output every n epochs')
tf.app.flags.DEFINE_boolean('log', True, 'Enable for tensorboard summary')
tf.app.flags.DEFINE_string('logdir', '/tmp/vintf/', 'Directory to store tensorboard summary')

config = tf.app.flags.FLAGS
# symbolic input image tensor where typically first channel is image, second is the reward prior
X = tf.placeholder(tf.float32, name="X", shape=[None, config.imsize, config.imsize, config.ch_i])
# symbolic input batches of vertical positions
S1 = tf.placeholder(tf.int32, name="S1", shape=[None, config.statebatchsize])
# symbolic input batches of horizontal positions
S2 = tf.placeholder(tf.int32, name="S2", shape=[None, config.statebatchsize])
y = tf.placeholder(tf.int32, name="y", shape=[None])
action_vecs_unnorm = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1)]
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('my-model{}{}.meta'.format(sd,'untied' if config.untied_weights else ''))
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    print("MODEL LOADED")
    Xtrain, S1train, S2train, ytrain, Xtest, S1test, S2test, ytest = process_gridworld_data(input=config.input, imsize=config.imsize)
    for it in range(len(Xtest)):
        img = np.expand_dims(Xtest[it], axis=0)
        flags = [False] * 10
        s1 = np.expand_dims(S1test[it], 0)
        s2 = np.expand_dims(S2test[it], 0)
        yt = ytest[:10]
        _, gx, gy, _ = np.where(img == 10)
        gx,gy = gx[0],gy[0]
        acc = [[] for i in range(10)]
        cpt = 0
        while cpt < 100:
            cpt +=1
            ext = True
            y_ = sess.run(y, feed_dict={X: img, S1: s1, S2: s2, y: yt})
            s1a, s2a = [], []
            for i, (xp, yp, m) in enumerate(zip(s1[0], s2[0], y_)):
                if xp != gx or yp != gy:
                    ext = False
                    xa, ya = action_vecs_unnorm[m]
                    xp = xp + xa
                    yp = yp + ya
                    acc[i].append((xp, yp))
                else:
                    print("DONE with",i)
                    flags[i] = True
                s1a.append(xp)
                s2a.append(yp)
            if ext:
                break
            else:
                s1, s2 = [s1a], [s2a]
        print("DONE Reached : ", sum(flags))
        import cv2
        size = 300
        fx,fy = size / sd, size/sd
        im = cv2.resize(cv2.cvtColor(img[0,:,:,0],cv2.COLOR_GRAY2RGB),(size,size),interpolation=cv2.INTER_NEAREST) * 255.
        cv2.rectangle(im,(int((gx- 0.5) * fx),int((gy- 0.5)*fy)),(int((gx + 0.5) * fx),int((gy+ 0.5)*fy)),(0,255,0))
        for g in acc:
            for i in range(1,len(g)):
                x1,y1 = g[i-1]
                x2,y2 = g[i]
                cv2.line(im,(int(x1 * fx),int(y1 * fy)),(int(x2 * fx),int(y2 * fy)),(255,255,0),3)
        cv2.destroyAllWindows()
        cv2.imshow('Reached : {}'.format(sum(flags)),im)
        cv2.waitKey(10000)
