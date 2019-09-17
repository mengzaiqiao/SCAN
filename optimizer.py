import tensorflow as tf
import numpy as np

flags = tf.flags
FLAGS = flags.FLAGS


class OptimizerSCVA(object):

    def __init__(self, preds, labels, model, num_nodes, num_features, pos_weight_u, norm_u, pos_weight_a, norm_a):
        
        preds_sub_u, preds_sub_a = preds
        labels_sub_u, labels_sub_a = labels
        
        # compute reconstruction loss
        self.cost_u = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub_u, targets=labels_sub_u, pos_weight=pos_weight_u))
        self.cost_a = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub_a, targets=labels_sub_a, pos_weight=pos_weight_a))
        self.cost_recon = FLAGS.alpha*self.cost_u + (1-FLAGS.alpha)*self.cost_a
        
        # compute kl divergence
        self.kl_u = FLAGS.alpha*(0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_u_log_std - tf.square(model.z_u_mean) - 
                                                                   tf.square(tf.exp(model.z_u_log_std)), 1))
        self.kl_a = (1-FLAGS.alpha)*(0.5 / num_features) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_a_log_std - tf.square(model.z_a_mean) - 
                                                                  tf.square(tf.exp(model.z_a_log_std)), 1))

        self.kl = FLAGS.beta * ((num_nodes + num_features) *3* self.kl_u  / num_nodes + num_nodes / num_features * self.kl_a)
        
        
        # compute P(y) of labelled nodes
        labelled_pos = model.labels_pos
        num_labelled_nodes = np.sum(labelled_pos)
        num_unlabelled_nodes = len(labelled_pos) - num_labelled_nodes
        unlabelled_pos = np.logical_not(labelled_pos)
        labelled_y_pred_logits = tf.boolean_mask(model.y_pred_logits, labelled_pos)
        labelled_y_true = tf.boolean_mask(model.y_train, labelled_pos)
        unlabelled_y_pred_logits = tf.boolean_mask(model.y_pred_logits, unlabelled_pos)
        unlabelled_y_true = tf.boolean_mask(model.y_pred_reconstruction, unlabelled_pos)
        
        label_entropy = FLAGS.lamb * tf.nn.softmax_cross_entropy_with_logits_v2(labels=labelled_y_true, logits=labelled_y_pred_logits)
        unlabel_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=unlabelled_y_true, logits=unlabelled_y_pred_logits)
        
        two_label_py = tf.reduce_sum(tf.add(tf.reshape(label_entropy, [-1, 1]), tf.tile(tf.reshape(label_entropy, [1, -1]), [num_labelled_nodes, 1])))
        two_unlabel_py = tf.reduce_sum(tf.add(tf.reshape(unlabel_entropy, [-1, 1]), tf.tile(tf.reshape(unlabel_entropy, [1, -1]), [num_unlabelled_nodes, 1])))
        one_label_py = 2 * tf.reduce_sum(tf.add(tf.reshape(label_entropy, [-1, 1]), tf.tile(tf.reshape(unlabel_entropy, [1, -1]), [num_labelled_nodes, 1])))
        
        self.nodes_py = (two_label_py + two_unlabel_py + one_label_py) / (num_nodes * num_nodes)
        self.features_py = (tf.reduce_sum(label_entropy) + tf.reduce_sum(unlabel_entropy)) / (num_nodes)
        
        self.py = FLAGS.alpha*self.nodes_py + (1-FLAGS.alpha)*self.features_py
        
        # compute shannoy entropy of P(y) of unlabelled nodes
        unlabelled_y_pred_prob = tf.boolean_mask(model.y_pred_prob, unlabelled_pos)
        entropy_unlabelled_y = -tf.reduce_sum(unlabelled_y_pred_prob * tf.log(unlabelled_y_pred_prob), axis=1)
        two_unlabel_entropy_y = tf.reduce_sum(tf.add(tf.reshape(entropy_unlabelled_y, [-1, 1]), tf.tile(tf.reshape(entropy_unlabelled_y, [1, -1]), [num_unlabelled_nodes, 1])))
        one_unlabel_entropy_y = 2 * tf.reduce_sum(tf.tile(tf.reshape(entropy_unlabelled_y, [1, -1]), [num_labelled_nodes, 1]))
        self.entropy_y = (two_unlabel_entropy_y + one_unlabel_entropy_y) / ((num_labelled_nodes + num_unlabelled_nodes) * num_unlabelled_nodes)
        self.entropy_y = FLAGS.alpha*self.entropy_y + (1-FLAGS.alpha)*tf.reduce_mean(entropy_unlabelled_y)  # add entropy of unlabel nodes for attributes
        
        """final cost"""
        self.cost = self.cost_recon - self.kl + self.py + self.entropy_y
        

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        

        self.correct_prediction_u = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub_u), 0.5), tf.int32),
                                           tf.cast(labels_sub_u, tf.int32))
        self.correct_prediction_a = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub_a), 0.5), tf.int32),
                                           tf.cast(labels_sub_a, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction_u, tf.float32)) + tf.reduce_mean(tf.cast(self.correct_prediction_a, tf.float32))

        
