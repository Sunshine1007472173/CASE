# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import threading
import random
import math
import tensorflow as tf
from read import read
from read import pair
from evaluation import evaluation
from un_generate_label import un_generate_label
import time
import os
import sys

dealedDataIndex = sys.argv[1]
dataindex = int(sys.argv[2])
epochs = int(sys.argv[3])
top_count = int(sys.argv[4])
batch_size = int(sys.argv[5])

readpath = ""
writepath = ""


learning_rate = 0.1
embedding_dim = 10
cs_alpha = 1
ss_alpha = 1


if not os.path.exists(writepath):
    os.mkdir(writepath)

########################################################## dataset ####################################################
myRead = read()

filename_claim = readpath + "claim"
words_claim = myRead.read_words(filename_claim)
dict_claim, reverse_dict_claim, count_list_claim = myRead.build_dict(words_claim)
del words_claim
filename_source = readpath +"source"
words_source = myRead.read_words(filename_source)
dict_source, reverse_dict_source, count_list_source = myRead.build_dict(words_source)
del words_source

pairlist_cs = myRead.build_dataset(filename=readpath + "cs",
                                   centerDict=dict_source,
                                   contextDict=dict_claim)
booklist_claim, claim_book_dict = myRead.build_book_claim(filename=readpath + "set_claim",
                                                          claimDict=dict_claim)
source_claim_dict, claim_source_dict = myRead.build_source_claim(pairlist_cs=pairlist_cs)
sspair_list = myRead.build_sspair_list(claim_source_dict=claim_source_dict, booklist=booklist_claim, writefile=readpath+"sspair")



print("cs size:",pairlist_cs.__len__())
print("booklist_cs size:", booklist_claim.__len__())
print("claim_book_dict size:",claim_book_dict.__len__())
print("sspair_list:",sspair_list.__len__())

########################################################## parameter ####################################################

claim_size = dict_claim.__len__()
source_size = dict_source.__len__()
semi_alpha = 1.01 - 1.0
semi_beta = 1.01 - 1.0


##########################################################graph begins##################################################
graph = tf.Graph()
with graph.as_default():
    ############## Input data. ##############
    train_centerWord = tf.placeholder(tf.int32, shape=[batch_size])
    train_contextWord = tf.placeholder(tf.int32, shape=[None])
    # for cs
    train_labels = tf.placeholder(tf.float32, shape=[batch_size, None])
    # for ss
    ss_same_count = tf.placeholder(tf.float32, shape=[1,1])

    ############## Ops and variables pinned to the CPU because of missing GPU implementation ##############
    ############## Ops and variables begins ##############
    with tf.device('/cpu:0'):
        # Embeddings for claims.
        claim_np = (np.random.random([claim_size, embedding_dim]).astype(np.float32)-0.5)
        claim_np_norm = np.linalg.norm(claim_np, keepdims=True)
        claim_np = claim_np * math.sqrt(5) / claim_np_norm
        sm_w_t_claim = tf.Variable(claim_np, name="sm_w_t_claim")
        # Embeddings for source
        source_np = np.zeros([source_size, embedding_dim], dtype=np.float32)
        for s_index in source_claim_dict:
            for c_index in source_claim_dict[s_index]:
                source_np[s_index] += claim_np[c_index]
            source_np[s_index] /= source_claim_dict[s_index].__len__()
        source_embeddings = tf.Variable(source_np,name="source_embeddings")
    ####################################  learning_rate  ####################################
    lr = learning_rate
    ########################################## cs #########################################
    # Embeddings for centerSource/contextClaim: [batchsize, embedding_dim]
    train_centerSource_embedding = tf.nn.embedding_lookup(source_embeddings, train_centerWord)
    train_contextClaim_embedding = tf.nn.embedding_lookup(sm_w_t_claim, train_contextWord)
    # logits_cs(c*s)
    logits_cs = tf.matmul(train_centerSource_embedding, train_contextClaim_embedding, transpose_b=True)
    # loss
    loss_tensor_cs = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_cs, labels=train_labels))
    #  sgd and optimizer
    sgd_cs = tf.train.GradientDescentOptimizer(lr)
    optimizer_cs = sgd_cs.minimize(loss_tensor_cs,
                           gate_gradients=sgd_cs.GATE_NONE)

    ################################################## ss ###################################################
    train_contextSource_embedding = tf.nn.embedding_lookup(source_embeddings, train_contextWord)
    # logits_ss(s*s)
    logits_ss = tf.matmul(train_centerSource_embedding, train_contextSource_embedding, transpose_b=True)
    # loss
    # p_ij = tf.sigmoid(logits_ss)
    loss_tensor_ss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_ss, labels=ss_same_count))
    #  sgd and optimizer
    sgd_ss = tf.train.GradientDescentOptimizer(lr)
    optimizer_ss = sgd_ss.minimize(loss_tensor_ss,
                           gate_gradients=sgd_ss.GATE_NONE)


    # Add variable initializer.
    init = tf.global_variables_initializer()

########################################################## thread.run() ################################################


def target():
    global  pairlist_cs, index_cs, avg_loss_cs, \
        sspair_list, index_ss, avg_loss_ss

    batch_cs_center = np.ndarray(shape=(batch_size), dtype=np.int32)
    batch_ss_center = np.ndarray(shape=(batch_size), dtype=np.int32)
    batch_ss_context = np.ndarray(shape=(batch_size), dtype=np.int32)

    same_count = np.ones([1,1], dtype=np.float32)


    while True:

        for cs_loop in range(0,cs_alpha):# cs
            # print(shanshan)
            batch_cs_center[0] = pairlist_cs[index_cs]._center
            bookIndex = claim_book_dict[pairlist_cs[index_cs]._context]
            bookclaims = booklist_claim[bookIndex]
            batch_labels_cs = np.zeros(shape=(1, bookclaims.__len__()), dtype=np.float32)
            claimInBook_index = bookclaims.index(pairlist_cs[index_cs]._context)
            batch_labels_cs[0, claimInBook_index] = 1
            index_cs = (index_cs + 1) % pairlist_cs.__len__()
            feed_dict_cs = {train_centerWord : batch_cs_center, train_contextWord : bookclaims, train_labels: batch_labels_cs}#, ss_same_count:pesudo_count, ss_diff_count:pesudo_count, semi_claim_embedding: pesudo_embeddings}
            _, loss_val_cs = session.run([optimizer_cs, loss_tensor_cs], feed_dict=feed_dict_cs)
            avg_loss_cs += loss_val_cs
            if index_cs == 0:
                break

        for ss_loop in range(0, ss_alpha):
            batch_ss_center[0] = sspair_list[index_ss]._center
            batch_ss_context[0] = sspair_list[index_ss]._context
            count_sum = sspair_list[index_ss]._samecount + sspair_list[index_ss]._diffcount
            same_count[0,0] = (sspair_list[index_ss]._samecount + semi_alpha*count_sum)/((1 + semi_alpha + semi_beta)*count_sum)
            index_ss = (index_ss + 1) % sspair_list.__len__()

            feed_dict_ss = {train_centerWord: batch_ss_center, train_contextWord: batch_ss_context, ss_same_count: same_count}#, ss_diff_count: diff_count, train_labels: pesudo_labels, semi_claim_embedding: pesudo_embeddings}
            _, loss_val_ss = session.run([optimizer_ss, loss_tensor_ss], feed_dict=feed_dict_ss)
            avg_loss_ss += loss_val_ss
        if index_cs == 0:
            break

####################################################### parameter ################################################

concurrent_threads = 20

######################################################## session ################################################
with tf.Session(graph=graph) as session:
    init.run()
    start = time.time()
    last_loss = 1000000.0

    max_length = pairlist_cs.__len__()

    for step in range(epochs):
        index_cs = 0
        index_ss = 0
        avg_loss_cs = 0
        avg_loss_ss = 0
        random.shuffle(pairlist_cs)
        random.shuffle(sspair_list)

        # workers = []
        # for _ in range(concurrent_threads):
        #   t = threading.Thread(target=target)
        #   t.start()
        #   workers.append(t)
        #
        # for t in workers:
        #   t.join()
        target()

        avg_cs = avg_loss_cs / max_length
        avg_ss = avg_loss_ss / max_length
        print("cs loss at step ", step, ":", avg_cs)
        print("ss loss at step ", step, ":", avg_ss)
        avg_loss = avg_cs + avg_ss
        print("loss", avg_loss)


    stop = time.time()
    lasting = stop - start

    final_embeddings_claim = sm_w_t_claim.eval()
    final_embeddings_source = source_embeddings.eval()
    np.savetxt(writepath + "final_embeddings_claim.txt",final_embeddings_claim)
    np.savetxt(writepath + "final_embeddings_source.txt",final_embeddings_source)

 
    mygenLabel = un_generate_label()
    mygenLabel.generate(readpath=readpath, resultpath=writepath, top_count=top_count)
    myevaluation = evaluation()
    myevaluation.evaluate(readpath=readpath, resultpath=writepath, lasting=lasting)


