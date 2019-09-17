from layers import GraphConvolution, GraphConvolutionSparse, InnerDecoder, Dense
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS


class Model(object):

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class SCVA(Model):

    def __init__(self, placeholders,adj,num_features, num_nodes, features_nonzero,num_labels,labels_pos,y_train,one_gcn=True,**kwargs):
        super(SCVA, self).__init__(**kwargs)

        self.Fn = placeholders['Fn']
        self.Fa = placeholders['Fa']
        self.adj = adj
        self.num_nodes = num_nodes
        self.num_feas = num_features
        self.num_labels = num_labels
        self.features_nonzero = features_nonzero
        self.dropout = placeholders['dropout']
        self.labels_pos = labels_pos
        self.y_train = y_train
        self.one_gcn = one_gcn
        
        self.build()

    def _build(self):
        
        if not self.one_gcn:
            self.Fn = tf.sparse_tensor_dense_matmul(self.adj,tf.sparse_tensor_to_dense(self.Fn,validate_indices=False))
        
        self.hidden1 = Dense(input_dim = self.num_nodes + self.num_feas,
                             output_dim=FLAGS.hidden1,
                             act=tf.nn.tanh,
                             sparse_inputs=self.one_gcn,
                             dropout=self.dropout)(self.Fn)
                             
        #predition of y
        node_fea = tf.sparse_tensor_dense_matmul(self.adj,self.hidden1)
        self.y_pred_logits = Dense(input_dim = FLAGS.hidden1,
                            output_dim = self.num_labels,
                            act = lambda x:x,
                            sparse_inputs=False,
                            dropout = self.dropout)(self.hidden1)
                            
        self.y_pred_prob = tf.nn.softmax(self.y_pred_logits)
        
        
        self.hidden2 = Dense(input_dim=self.num_nodes,
                             output_dim=FLAGS.hidden1,
                             act=tf.nn.tanh,
                             sparse_inputs=True,
                             dropout=self.dropout)(tf.sparse_transpose(self.Fa))

        #embeddings of nodes: mean and log variance
        yz = self.yz() # concat information y
        
        #use convolution results only
        self.z_u_mean = Dense(input_dim = FLAGS.hidden1 + self.num_labels,
                             output_dim=FLAGS.hidden2,
                             act=lambda x: x,
                             sparse_inputs=False,
                             dropout=self.dropout)(tf.concat([node_fea,yz],axis=1))
      
        self.z_u_log_std = Dense(input_dim=FLAGS.hidden1 + self.num_labels,
                                 output_dim=FLAGS.hidden2,
                                 act=lambda x: x,
                                 sparse_inputs=False,
                                 dropout=self.dropout)(tf.concat([node_fea,yz],axis=1))
        
        #embeddings of features
        self.z_a_mean = Dense(input_dim=FLAGS.hidden1,
                              output_dim=FLAGS.hidden2 + self.num_labels,
                              act=lambda x: x,
                              sparse_inputs=False,
                              dropout=self.dropout)(self.hidden2)

        self.z_a_log_std = Dense(input_dim=FLAGS.hidden1,
                                 output_dim=FLAGS.hidden2 + self.num_labels,
                                 act=lambda x: x,
                                 sparse_inputs=False,
                                 dropout=self.dropout)(self.hidden2)
        
        
        #sampling from embeddings of nodes and features
        self.z_u = self.z_u_mean + tf.random_normal([self.num_nodes, FLAGS.hidden2]) * tf.exp(self.z_u_log_std)
        self.z_a = self.z_a_mean + tf.random_normal([self.num_feas, FLAGS.hidden2 + self.num_labels]) * tf.exp(self.z_a_log_std)

        #sampling from y_pred
        self.y_pred = self.gumbel_softmax()
        
        #get y for reconstruction
        self.y_pred_reconstruction = self.reconstruction_y()
        
        #concat z_u and y_pred
        zy = tf.concat([self.z_u,self.y_pred_reconstruction],axis = 1)
        self.reconstructions = InnerDecoder(input_dim=FLAGS.hidden2 + self.num_labels,
                                      act=lambda x: x,
                                      logging=self.logging)((zy, self.z_a))
        
        
    def gumbel_softmax(self):
        """
        sample from categorical distribution using gumbel softmax trick
        """
        g = -tf.log(-tf.log(tf.random_uniform([self.num_nodes,self.num_labels])))
        y_pred = tf.exp((tf.log(self.y_pred_prob) + g) / FLAGS.temperature)
        y_pred = y_pred / tf.reshape(tf.reduce_sum(y_pred,axis=1),(-1,1))
            
        return y_pred
    
    
    def reconstruction_y(self):
        """
        get y_pred for reconstruction: replace probabilities of nodes with
        label with label one-hot encoding
        """
        #one hot encoding for label data
        y_pred_reconstruct = tf.where(self.labels_pos,x =self.y_train,y = self.y_pred)
        return y_pred_reconstruct
    
    def yz(self):
        """
        get probability of y to compute z
        """
        yz = tf.where(self.labels_pos,x = self.y_train, y=self.y_pred_prob)
        return yz
    