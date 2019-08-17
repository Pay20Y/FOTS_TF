import tensorflow as tf
from tensorflow.contrib import slim, rnn
import numpy as np
import config
import os


class Recognition(object):
	def __init__(self, rnn_hidden_num=256, keepProb=0.8, weight_decay=1e-5, is_training=True):
		self.rnn_hidden_num = rnn_hidden_num
		self.batch_norm_params = {'decay': 0.997, 'epsilon': 1e-5, 'scale': True, 'is_training': is_training}
		self.keepProb = keepProb if is_training else 1.0
		self.weight_decay = weight_decay
		self.num_classes = config.NUM_CLASSES
	
	def cnn(self, rois):
		with tf.variable_scope('recog/cnn'):
			with slim.arg_scope([slim.conv2d],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=self.batch_norm_params,
                weights_regularizer=slim.l2_regularizer(self.weight_decay)):
				conv1 = slim.conv2d(rois, 64, 3, stride=1, padding='SAME')
				conv1 = slim.conv2d(conv1, 64, 3, stride=1, padding='SAME')
				pool1 = slim.max_pool2d(conv1, kernel_size=[2,1], stride=[2,1], padding='SAME')
				conv2 = slim.conv2d(pool1, 128, 3, stride=1, padding='SAME')
				conv2 = slim.conv2d(conv2, 128, 3, stride=1, padding='SAME')
				pool2 = slim.max_pool2d(conv2, kernel_size=[2,1], stride=[2,1], padding='SAME')
				conv3 = slim.conv2d(pool2, 256, 3, stride=1, padding='SAME')
				conv3 = slim.conv2d(conv3, 256, 3, stride=1, padding='SAME')
				pool3 = slim.max_pool2d(conv3, kernel_size=[2,1], stride=[2,1], padding='SAME')

				return pool3

	def bilstm(self, input_feature, seq_len):
		with tf.variable_scope("recog/rnn"):
			lstm_fw_cell = rnn.LSTMCell(self.rnn_hidden_num)
			lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, input_keep_prob=self.keepProb, output_keep_prob=self.keepProb)
			lstm_bw_cell = rnn.LSTMCell(self.rnn_hidden_num)
			lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, input_keep_prob=self.keepProb, output_keep_prob=self.keepProb)
			infer_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_feature, sequence_length=seq_len, dtype=tf.float32)
			infer_output = tf.concat(infer_output, axis=-1)
			return infer_output

	def build_graph(self, rois, seq_len):
		num_rois = tf.shape(rois)[0]

		cnn_feature = self.cnn(rois) # N * 1 * W * C

		cnn_feature = tf.squeeze(cnn_feature, axis=1) # N * W * C

		lstm_output = self.bilstm(cnn_feature, seq_len) # N * T * 2H

		logits = tf.reshape(lstm_output, [-1, self.rnn_hidden_num * 2]) # (N * T) * 2H
		
		W = tf.Variable(tf.truncated_normal([self.rnn_hidden_num * 2, self.num_classes], stddev=0.1), name="W")
		b = tf.Variable(tf.constant(0., shape=[self.num_classes]), name="b")

		logits = tf.matmul(logits, W) + b # (N * T) * Class

		logits = tf.reshape(logits, [num_rois, -1, self.num_classes])
		
		logits = tf.transpose(logits, (1, 0, 2))

		return logits

	def loss(self, logits, targets, seq_len):
		# Loss and cost calculation
		loss = tf.nn.ctc_loss(targets, logits, seq_len, ignore_longer_outputs_than_inputs=True)
		recognition_loss = tf.reduce_mean(loss)
		return recognition_loss

	def decode(self, logits, seq_len):
		decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
		dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)

		return decoded, dense_decoded
	def decode_with_lexicon(self, logits, seq_len, lexicon_path):
		pass
