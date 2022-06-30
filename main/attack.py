import json
import os
from urllib.request import urlretrieve
from matplotlib import pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
# 输入变量
imagepath = "./images/out.JPEG" # 图片地址（单个）
modelpath = "C:\\Users\\LENOV\\Desktop\\tuAn\\model" # 模型地址，model文件夹下有inception_v3.ckpt
imagename = imagepath[9:]
# print(imagename)
savepath = "./save_attack"
demo_target = 265 # 想攻击到的目标类别，范围0-999, 可以是random
# 设置打印最低级别
tf.logging.set_verbosity(tf.logging.ERROR)
# 默认sess
sess = tf.InteractiveSession()

image = tf.Variable(tf.zeros((299, 299, 3)))

def inception(image,reuse):
    # * - 1... 预处理
    preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)
    # 生成网络中常用函数的默认参数，L2正则weight_decay=0.0
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits,_ = nets.inception.inception_v3(
            preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:,1:] # ignore background class，全连接层（往往是模型的最后一层）的值
        probs = tf.nn.softmax(logits) # probabilities，归一化的值，含义是属于该位置的概率
    return logits, probs
logits, probs = inception(image,reuse=False)

restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('InceptionV3/')
]
saver = tf.train.Saver(restore_vars)
saver.restore(sess,os.path.join(modelpath,"inception_v3.ckpt"))

img = Image.open(imagepath)
big_dim = max(img.width, img.height)

wide = img.width > img.height
new_w = 299 if not wide else int(img.width * 299 / img.height)
new_h = 299 if wide else int(img.height * 299 / img.width)
img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))
img = (np.asarray(img) / 255.0).astype(np.float32) # image->array 归一化

# cv2.imshow("img",img)
# cv2.waitKey(0)
# 对抗样本 or 对抗攻击
# op初始化
x = tf.placeholder(tf.float32, (299, 299, 3))
x_hat = image
assign_op = tf.assign(x_hat, x)

# 最小化交叉熵
learning_rate = tf.placeholder(tf.float32, ())
y_hat = tf.placeholder(tf.int32, ())

labels = tf.one_hot(y_hat, 1000)
loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])
optim_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=[x_hat])

# 投影
epsilon = tf.placeholder(tf.float32, ())

below = x - epsilon
above = x + epsilon
# 调整下限为below，上限为above
projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
with tf.control_dependencies([projected]):
    project_step = tf.assign(x_hat, projected)

demo_epsilon = 2.0/255.0 # a really small perturbation
demo_lr = 1e-1
demo_steps = 100


sess.run(assign_op, feed_dict={x: img})

# projected gradient descent
for i in range(demo_steps):
    # gradient descent step
    _, loss_value = sess.run(
        [optim_step, loss],
        feed_dict={learning_rate: demo_lr, y_hat: demo_target})
    # project step
    sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon})
    # if loss_value < 0.01:
    #     print('step %d, loss=%g' % (i+1, loss_value))
    #     break
    # if (i+1) % 10 == 0:
    #     print('step %d, loss=%g' % (i+1, loss_value))

adv = x_hat.eval()
# cv2.imshow("attack", adv)
# cv2.waitKey(0)

cv2.imwrite(os.path.join(savepath, "attack_"+imagename), adv*255, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
# cv2.imshow("adv", adv)
# cv2.waitKey(0)
sub_image = cv2.subtract(adv*255, img)
# cv2.imshow("sub", sub_image)
# cv2.waitKey(0)
cv2.imwrite(os.path.join(savepath, "attack_sub_"+imagename), sub_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
