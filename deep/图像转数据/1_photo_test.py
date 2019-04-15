# -*- coding:utf-8 -*-
import time
import tensorflow as tf
import forward
import backward
from tensorflow.python.ops.image_ops_impl import ResizeMethod
import numpy as np
from PIL import Image

def img_read(pngfile):
    image_data = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(pngfile),
                                                                   channels=1), dtype=tf.float32)
    height = 56
    width = 56
    image = tf.image.resize_images(image_data, (height, width), method=ResizeMethod.BILINEAR)
    image = tf.expand_dims(image, -1)
    image = tf.reshape(image, (1, 56, 56, 1))
    return image


def load_trained_model(logits):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            # 从训练模型恢复数据
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return
        predict = tf.argmax(logits, 1)
        output = predict.eval()
        print(output)




def main():
    data = img_read("D:/python_text/图像转数据/1.png")
    logits = forward.forward(data, True, backward.REGULARIZER)
    load_trained_model(logits=logits)

if __name__=="__main__":
    main()
