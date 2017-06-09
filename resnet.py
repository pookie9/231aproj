from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import tensorflow.contrib.layers as layers

from util import Progbar, minibatches

# from evaluate import exact_match_score, f1_score

from IPython import embed

from tensorflow.python.ops.gen_math_ops import _batch_mat_mul as batch_matmul

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

class placesModel(object):
    def __init__(self, flags):
        """
        Initializes your System

        :param args: pass in more arguments as needed
        """
        self.flags = flags
        self.l2_reg=self.flags.l2_reg
        self.h_size = self.flags.state_size
        self.dropout = self.flags.dropout
        self.height = self.flags.input_height
        self.width = self.flags.input_width
        self.channels = 3
        self.layer_params=self.flags.layer_params
        self.train_all=flags.train_all
        # ==== set up placeholder tokens ========

        self.input_placeholder = tf.placeholder(tf.float32, shape=(None,self.height,self.width,self.channels), name='input_placeholder')
        self.label_placeholder = tf.placeholder(tf.int32, shape=(None,), name='label_placeholder')
        self.is_train_placeholder = tf.placeholder(tf.bool, shape=(), name='is_train_placeholder')

        # ==== assemble pieces ====
        with tf.variable_scope("places_model", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_graph()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        self.global_step = tf.Variable(0, trainable=False)
        self.starter_learning_rate = self.flags.learning_rate

        self.learning_rate = self.starter_learning_rate

        self.optimizer = get_optimizer("adam")
        
        if self.flags.grad_clip:
            # gradient clipping
            self.optimizer = self.optimizer(self.learning_rate)
            grads = self.optimizer.compute_gradients(self.loss)
            for i, (grad, var) in enumerate(grads):
                if grad is not None:
                    grads[i] = (tf.clip_by_norm(grad, self.flags.max_gradient_norm), var)
            self.train_op = self.optimizer.apply_gradients(grads, global_step=self.global_step)
        else:
            # no gradient clipping
            self.train_op = self.optimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.saver=tf.train.Saver()


    def setup_graph(self):
        with tf.variable_scope("simple_feed_forward",reuse=False):
            cur_in=self.input_placeholder
            prev_depth=self.channels
            prev_res=None
            prev_res_depth=None
            counter=0
            num_layer=0
            for params in self.layer_params:
                num_layer+=1
                #params is a tuple of (type, number,shape,stride,depth,use_batch_norm,
                for i in range(params[1]):
                    counter+=1
                    if params[0]=='fc':
                        flat=layers.flatten(cur_in)
                        #W_shape=[flat.get_shape()[-1],params[2]]
                        #b_shape=[params[2]]
                        #W=tf.get_variable('FC_W'+str(counter),shape=W_shape,initializer=layers.xavier_initializer())
                        #b=tf.get_variable('FC_b'+str(counter),shape=b_shape,initializer=tf.constant_initializer(0.0))
                        #cur_in=tf.matmul(flat,W)+b
                        name='FC_'+str(counter)
                        trainable=self.train_all
                        if i==params[1]-1 and num_layer==len(self.layer_params) :
                            name='OUTPUT_LAYER'
                            trainable=True
                        cur_in = tf.layers.dense(inputs=flat,
                                                 units=params[2],
                                                 activation=None,
                                                 kernel_initializer=layers.xavier_initializer(),
                                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg, scope=name),
                                                 name=name,
                                                 trainable=trainable)
                        
                        if i<params[1]-1 and num_layer!=len(self.layer_params) :
                            cur_in=tf.nn.relu(cur_in)
                            cur_in = tf.layers.dropout(cur_in,rate=self.dropout,training=self.is_train_placeholder,name='fc_do'+str(counter))
                    if params[0]=='batchnorm':
                        cur_in=tf.layers.batch_normalization(cur_in,training=self.is_train_placeholder,name="bn"+str(counter),trainable=self.train_all)
                    if params[0]=='relu':
                        cur_in=tf.nn.relu(cur_in)
                    if params[0]=='maxpool':
                        cur_in=tf.layers.max_pooling2d(cur_in,pool_size=params[2],strides=params[3])
                        if params[6]:                           
                            prev_res=cur_in
                            prev_res_depth=prev_depth#cur_in.get_shape()[-1]
                            #print "HERE",params,prev_depth,prev_res_depth
                    if params[0]=='avgpool':
                        cur_in=tf.nn.pool(cur_in,window_shape=params[2],pooling_type='AVG',padding='SAME')
                    if params[0]=='conv':
                        W_shape=[params[2][0],params[2][1],prev_depth,params[4]]
                        b_shape=[params[4]]
                        prev_depth=params[4]
                        W=tf.get_variable('W'+str(counter),shape=W_shape,initializer=layers.xavier_initializer(),trainable=self.train_all)
                        b = tf.get_variable('b'+str(counter)+'conv',shape=b_shape,initializer=tf.constant_initializer(0.0),trainable=self.train_all)         
                        z = tf.nn.conv2d(cur_in,W,params[3],'SAME') +b
                        if params[6]:
                            if prev_res!=None:
                                if prev_res_depth<prev_depth:
                                    #Takes care of the diffences in cross-sectional areas
                                    if prev_res.get_shape()[1]!=z.get_shape()[1]:
                                        prev_res=4*tf.nn.pool(prev_res,window_shape=(2,2),strides=(2,2),pooling_type='AVG',padding='SAME')
                                    #Takes care of when you increase the depth, zero pads out to new (presumably larger) depth
                                    prev_res=tf.pad(prev_res,paddings=([0,0],[0,0],[0,0],[(prev_depth-prev_res_depth)//2]*2),mode='CONSTANT')
                                elif prev_res_depth!=prev_depth:
                                    print ("ERROR: residual of greater size then current size",prev_res_depth,"=>",prev_depth)
                                    exit(1)
                                z=prev_res+z
                        if params[5]:
                            bn = tf.layers.batch_normalization(z,training=self.is_train_placeholder,name="bn"+str(counter),trainable=self.train_all)
                        h=tf.nn.relu(bn)
                        if params[6]:                            
                            prev_res=h
                            prev_res_depth=prev_depth
                        cur_in=h
            self.label_predictions=cur_in            
            

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_placeholder, logits=self.label_predictions))


    def optimize(self, session, image_batch, label_batch):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}
        input_feed[self.input_placeholder] = image_batch
        input_feed[self.label_placeholder] = label_batch
        input_feed[self.is_train_placeholder]=True
        output_feed = [self.train_op, self.loss]
        _, loss = session.run(output_feed, input_feed)

        return loss

    def forward_pass(self, session, image_batch, label_batch):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}
        input_feed[self.input_placeholder] = image_batch
        input_feed[self.label_placeholder] = label_batch
        input_feed[self.is_train_placeholder]=True
        output_feed = [self.label_predictions]
        outputs = session.run(output_feed, input_feed)

        return outputs[0]

    def answer(self, session, data):

        scores = []
        prog_train = Progbar(target=1 + int(len(data[0]) / self.flags.batch_size))
        for i, batch in enumerate(self.minibatches(data, self.flags.batch_size, shuffle=False)):
            score = self.forward_pass(session, *batch)  
            scores.append(score)
            prog_train.update(i + 1, [("Predicting Images....",0.0)])
        print("")
        scores=np.vstack(scores)
        predictions = np.argmax(scores, axis=-1)
        return predictions

    def validate(self, session, image_batch, label_batch):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        input_feed = {}

        input_feed[self.input_placeholder] = image_batch
        input_feed[self.label_placeholder] = label_batch
        input_feed[self.is_train_placeholder]=True

        output_feed = [self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs[0]

    def evaluate_answer(self, session, dataset, sample=100, log=False, eval_set='train'):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        if sample is None:
            sampled = dataset
            sample = len(dataset[0])
        else:
            #np.random.seed(0)
            inds = np.random.choice(len(dataset[0]), sample,replace=False)
            sampled = [elem[inds] for elem in dataset]
        
        predictions = self.answer(session, sampled)
        images, labels = sampled
        accuracy = np.mean(predictions == labels)

        if log:
            logging.info("{}, accuracy: {}, for {} samples".format(eval_set, accuracy, sample))

        return accuracy

    def run_epoch(self, sess, train_set, val_set):
        prog_train = Progbar(target=1 + int(len(train_set[0]) / self.flags.batch_size))
        for i, batch in enumerate(self.minibatches(train_set, self.flags.batch_size)):
            loss = self.optimize(sess, *batch)
            prog_train.update(i + 1, [("train loss", loss)])
        print("")

        #if self.flags.debug == 0:
        prog_val = Progbar(target=1 + int(len(val_set[0]) / self.flags.batch_size))
        for i, batch in enumerate(self.minibatches(val_set, self.flags.batch_size)):
            val_loss = self.validate(sess, *batch)
            prog_val.update(i + 1, [("val loss", val_loss)])
        print("")

        self.evaluate_answer(session=sess,
                             dataset=train_set,
                             sample=len(val_set[0]),
                             log=True,
                             eval_set="-Epoch TRAIN-")

        self.evaluate_answer(session=sess,
                             dataset=val_set,
                             sample=None,
                             log=True,
                             eval_set="-Epoch VAL-")


    def minibatches(self, data, batch_size, shuffle=True):
        num_data = len(data[0])
        images,labels = data
        indices = np.arange(num_data)
        if shuffle:
            np.random.shuffle(indices)
        for minibatch_start in np.arange(0, num_data, batch_size):
            minibatch_indices = indices[minibatch_start:minibatch_start + batch_size]
            yield [images[minibatch_indices],labels[minibatch_indices]]

    def train(self, session, train_dataset, val_dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious approach can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        #self.saver=saver
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        # context_ids, question_ids, answer_spans, ctx_mask ,q_mask, train_context = dataset
        # train_dataset = [context_ids, question_ids, answer_spans, ctx_mask ,q_mask]

        # val_context_ids, val_question_ids, val_answer_spans, val_ctx_mask, val_q_mask, val_context = val_dataset
        # val_dataset = [val_context_ids, val_question_ids, val_answer_spans, val_ctx_mask, val_q_mask]

        
        num_epochs = self.flags.epochs

        # print train_dataset[0].shape,train_dataset[1].shape
        # print val_dataset[0].shape,val_dataset[1].shape

        #if self.flags.debug:
        #    train_dataset = [elem[:self.flags.batch_size*1] for elem in train_dataset]
        #    val_dataset = [elem[:self.flags.batch_size*1] for elem in val_dataset]
        #    num_epochs = 100
        
        # print train_dataset[0].shape,train_dataset[1].shape
        # print val_dataset[0].shape,val_dataset[1].shape
        # assert False

        self.saver.save(session, train_dir+"/"+self.flags.run_name+".ckpt")
        for epoch in range(num_epochs):
            logging.info("Epoch %d out of %d", epoch + 1, self.flags.epochs)
            self.run_epoch(sess=session,
                           train_set=train_dataset, 
                           val_set=val_dataset)
            logging.info("Saving model in %s", train_dir)
            self.saver.save(session, train_dir+"/"+self.flags.run_name+".ckpt")






