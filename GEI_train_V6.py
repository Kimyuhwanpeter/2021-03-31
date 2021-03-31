# -*- coding:utf-8 -*-
from random import random, shuffle
from original_model import *

import tensorflow as tf
import numpy as np
import os
import datetime
import easydict

FLAGS = easydict.EasyDict({"img_height": 128,
                           
                           "img_width": 88,
                           
                           "tr_txt_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/train.txt",
                           
                           "tr_txt_name": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI_IDList_train.txt",
                           
                           "tr_img_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI/",
                           
                           "te_txt_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/test.txt",
                           
                           "te_txt_name": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI_IDList_test_fix.txt",
                           
                           "te_img_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI/",
                           
                           "batch_size": 100,
                           
                           "epochs": 500,
                           
                           "num_classes": 64,
                           
                           "lr": 0.0001,
                           
                           "save_checkpoint": "",
                           
                           "graphs": "", 
                           
                           "train": True,
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": ""})

optim = tf.keras.optimizers.Adam(FLAGS.lr)

def tr_func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_png(img, 1)
    img = tf.image.resize(img, [FLAGS.img_height, FLAGS.img_width])
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    #lab = lab_list - 2
    n_age = 10
    generation = 0.
    lab = 0
    if lab_list >= 2 and lab_list < 10:
        generation = 0
        generation = tf.one_hot(generation, 9)
        lab = n_age - (10 - lab_list)
    if lab_list >= 10 and lab_list < 20:
        generation = 1
        generation = tf.one_hot(generation, 9)
        lab = n_age - (20 - lab_list)
    if lab_list >= 20 and lab_list < 30:
        generation = 2
        generation = tf.one_hot(generation, 9)
        lab = n_age - (30 - lab_list)
    if lab_list >= 30 and lab_list < 40:
        generation = 3
        generation = tf.one_hot(generation, 9)
        lab = n_age - (40 - lab_list)
    if lab_list >= 40 and lab_list < 50:
        generation = 4
        generation = tf.one_hot(generation, 9)
        lab = n_age - (50 - lab_list)
    if lab_list >= 50 and lab_list < 60:
        generation = 5
        generation = tf.one_hot(generation, 9)
        lab = n_age - (60 - lab_list)
    if lab_list >= 60 and lab_list < 70:
        generation = 6
        generation = tf.one_hot(generation, 9)
        lab = n_age - (70 - lab_list)
    if lab_list >= 70 and lab_list < 80:
        generation = 7
        generation = tf.one_hot(generation, 9)
        lab = n_age - (80 - lab_list)
    if lab_list >= 80:
        generation = 8
        generation = tf.one_hot(generation, 9)
        lab = n_age - (90 - lab_list)



    return img, lab, generation

def te_func(img, lab):

    img = tf.io.read_file(img)
    img = tf.image.decode_png(img, 1)
    img = tf.image.resize(img, [FLAGS.img_height, FLAGS.img_width])
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    #lab = lab - 2

    return img, lab

def make_label_V2(ori_label):
    l = []
    for i in range(FLAGS.batch_size):
        label = [1] * (ori_label[i].numpy() + 1) + [0] * (10 - (ori_label[i].numpy() + 1))
        label = tf.cast(label, tf.float32)
        l.append(label)
    return tf.convert_to_tensor(l, tf.float32)

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def modified_fea(logits, label, repeat, i):

    logit = logits.numpy()
    logit[label[i]] = logit[label[i]] * (1. - logit[label[i]])
    for j in range(repeat):
        if j != label[i]:
            logit[j] = logit[j] * logit[label[i]]

    return logit

def cal_loss(model, images, age_labels, gener_labels):
    loss = 0.
    with tf.GradientTape() as tape:

        logits = run_model(model, images, True)
        
        mean_feature = tf.reduce_mean(logits, 2)    # [None, 9]
        
        age_generation_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(gener_labels, mean_feature) # 나이 세대에 관한 loss

        for i in range(9):

            logit = logits[:, i, :] # [None, 10]
            #(1 - gener_labels) * (세대가 맞지 않을때의 loss) + (gener_labels) * (세대가 맞을때의 loss)

            loss += (age_generation_loss) * (tf.keras.losses.BinaryCrossentropy(from_logits=True)(age_labels, logit))

        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def cal_mae(model, images, labels):

    logits = run_model(model, images, False)

    mean_feature = tf.nn.softmax(tf.reduce_mean(logits, 2), 1)    # [None, 9]

    predict_generation = tf.argmax(mean_feature, 1, tf.int32)
    #predict_generation = predict_generation.numpy()[0]


    #age_feature = logits[:, predict_generation, :]  # [None, 10]

    #age_predict = tf.nn.sigmoid(age_feature)
    #age_predict = tf.cast(tf.less(0.5, age_predict), tf.int32)
    #age_predict = tf.reduce_sum(age_predict, 1)


    #final_age = (predict_generation * 10) + age_predict

    #ae = tf.reduce_sum(tf.abs(final_age - labels))
    ae = 0
    predict_generation = predict_generation.numpy()
    for i in range(137):
        predict_generation_ = predict_generation[i]


        age_feature = logits[i, predict_generation_, :]  # [None, 10]

        age_predict = tf.nn.sigmoid(age_feature)
        age_predict = tf.cast(tf.less(0.5, age_predict), tf.int32)
        age_predict = tf.reduce_sum(age_predict)


        final_age = (predict_generation_ * 10) + age_predict

        ae += (tf.abs(final_age - labels[i]))


    return ae

def main():
    model = GL_network(input_shape=(FLAGS.img_height, FLAGS.img_width, 1))
    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored the latest checkpoint!!")


    if FLAGS.train:
        count = 0;

        tr_img = np.loadtxt(FLAGS.tr_txt_name, dtype="<U100", skiprows=0, usecols=0)
        tr_img = [FLAGS.tr_img_path + img + ".png"for img in tr_img]
        tr_lab = np.loadtxt(FLAGS.tr_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_img = np.loadtxt(FLAGS.te_txt_name, dtype="<U100", skiprows=0, usecols=0)
        te_img = [FLAGS.te_img_path + img + ".png" for img in te_img]
        te_lab = np.loadtxt(FLAGS.te_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_gener = tf.data.Dataset.from_tensor_slices((te_img, te_lab))
        te_gener = te_gener.map(te_func)
        te_gener = te_gener.batch(137)
        te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

        #############################
        # Define the graphs
        #current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #train_log_dir = FLAGS.graphs + current_time + '/train'
        #train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        #val_log_dir = FLAGS.graphs + current_time + '/val'
        #val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        #############################

        for epoch in range(FLAGS.epochs):
            TR = list(zip(tr_img, tr_lab))
            shuffle(TR)
            tr_img, tr_lab = zip(*TR)
            tr_img, tr_lab = np.array(tr_img), np.array(tr_lab, dtype=np.int32)

            tr_gener = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
            tr_gener = tr_gener.shuffle(len(tr_img))
            tr_gener = tr_gener.map(tr_func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(tr_gener)
            tr_idx = len(tr_img) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, batch_labels, batch_age_gener = next(tr_iter)
                batch_labels = make_label_V2(batch_labels)

                loss = cal_loss(model, batch_images, batch_labels, batch_age_gener)

                if count % 10 == 0:
                    print(loss)

                if count % 100 == 0:
                    te_iter = iter(te_gener)
                    te_idx = len(te_img) // 137
                    ae = 0
                    for i in range(te_idx):
                        imgs, labs = next(te_iter)

                        ae += cal_mae(model, imgs, labs)
                        if i % 100 == 0:
                            print("{} mae = {}".format(i + 1, ae / ((i + 1) * 137)))

                    MAE = ae / len(te_img)
                    print("================================")
                    print("step = {}, MAE = {}".format(count, MAE))
                    print("================================")


                count += 1

if __name__ == "__main__":
    main()