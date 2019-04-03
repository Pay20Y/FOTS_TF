import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 8, '')
tf.app.flags.DEFINE_integer('num_readers', 8, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('gpu_list', '1', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints/', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')

import icdar
# import synth
from module import Backbone_branch, Recognition_branch, RoI_rotate

FLAGS = tf.app.flags.FLAGS

# gpus = list(range(len(FLAGS.gpu_list.split(','))))

detect_part = Backbone_branch.Backbone(is_training=True)
roi_rotate_part = RoI_rotate.RoIRotate()
recognize_part = Recognition_branch.Recognition(is_training=True) 
"""
def build_graph(input_images, input_transform_matrix, input_box_mask, input_box_widths, input_box_nums, input_seq_len):

    shared_feature, f_score, f_geometry = detect_part.model(input_images)
    pad_rois = roi_rotate_part.roi_rotate_tensor(shared_feature, input_transform_matrix, input_box_mask, input_box_widths, input_box_nums)
    recognition_logits = recognize_part.build_graph(pad_rois, input_seq_len, input_box_nums)
    _, dense_decode = recognize_part.decode(recognition_logits, input_seq_len)
    return f_score, f_geometry, recognition_logits, dense_decode
    # return f_score, f_geometry
"""

def build_graph(input_images, input_transform_matrix, input_box_masks, input_box_widths, input_seq_len):
    
    shared_feature, f_score, f_geometry = detect_part.model(input_images)
    pad_rois = roi_rotate_part.roi_rotate_tensor(shared_feature, input_transform_matrix, input_box_masks, input_box_widths)
    recognition_logits = recognize_part.build_graph(pad_rois, input_seq_len)
    _, dense_decode = recognize_part.decode(recognition_logits, input_seq_len)
    return f_score, f_geometry, recognition_logits, dense_decode
    # return f_score, f_geometry
"""
def compute_loss(f_score, f_geometry, recognition_logits, input_score_maps, input_geo_maps, input_training_masks, input_transcription, input_seq_len, lamda=0.01):
    detection_loss = detect_part.loss(input_score_maps, f_score, input_geo_maps, f_geometry, input_training_masks)
    recognition_loss = recognize_part.loss(recognition_logits, input_transcription, input_seq_len)

    tf.summary.scalar('detect_loss', detection_loss)
    tf.summary.scalar('recognize_loss', recognition_loss)

    return detection_loss, recognition_loss, detection_loss + lamda * recognition_loss
"""
def compute_loss(f_score, f_geometry, recognition_logits, input_score_maps, input_geo_maps, input_training_masks, input_transcription, input_seq_len, lamda=0.01):
    detection_loss = detect_part.loss(input_score_maps, f_score, input_geo_maps, f_geometry, input_training_masks)
    recognition_loss = recognize_part.loss(recognition_logits, input_transcription, input_seq_len)

    tf.summary.scalar('detect_loss', detection_loss)
    tf.summary.scalar('recognize_loss', recognition_loss)

    return detection_loss, recognition_loss, detection_loss + lamda * recognition_loss

def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    else:
        if not FLAGS.restore:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
            tf.gfile.MkDir(FLAGS.checkpoint_path)

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_maps')
    input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='input_geo_maps')
    input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')
    input_transcription = tf.sparse_placeholder(tf.int32, name='input_transcription')
    
    input_transform_matrix = tf.placeholder(tf.float32, shape=[None, 6], name='input_transform_matrix')
    input_box_masks = []
    # input_box_mask = tf.placeholder(tf.int32, shape=[None], name='input_box_mask')
    input_box_widths = tf.placeholder(tf.int32, shape=[None], name='input_box_widths')
    input_seq_len = input_box_widths[tf.argmax(input_box_widths, 0)] * tf.ones_like(input_box_widths)
    # input_box_nums = tf.placeholder(tf.int32, name='input_box_nums')
    # input_seq_len = tf.placeholder(tf.int32, shape=[None], name='input_seq_len')

    for i in range(FLAGS.batch_size_per_gpu):
        input_box_masks.append(tf.placeholder(tf.int32, shape=[None], name='input_box_masks_' + str(i)))

    # f_score, f_geometry, recognition_logits, dense_decode = build_graph(input_images, input_transform_matrix, input_box_mask, input_box_widths, input_box_nums, input_seq_len)
    f_score, f_geometry, recognition_logits, dense_decode = build_graph(input_images, input_transform_matrix, input_box_masks, input_box_widths, input_seq_len)
    # f_score, f_geometry = build_graph(input_images, input_transform_matrix, input_box_mask, input_box_widths, input_box_nums, input_seq_len)

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=10000, decay_rate=0.94, staircase=True)
    # add summary
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)
    # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)

    # d_loss, r_loss, model_loss = compute_loss(f_score, f_geometry, recognition_logits, input_score_maps, input_geo_maps, input_training_masks, input_transcription, input_seq_len)
    d_loss, r_loss, model_loss = compute_loss(f_score, f_geometry, recognition_logits, input_score_maps, input_geo_maps, input_training_masks, input_transcription, input_seq_len)
    # total_loss = detect_part.loss(input_score_maps, f_score, input_geo_maps, f_geometry, input_training_masks)
    tf.summary.scalar('total_loss', model_loss)
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    # total_loss = model_loss
    batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        ckpt = tf.train.latest_checkpoint(FLAGS.pretrained_model_path)
        variable_restore_op = slim.assign_from_checkpoint_fn(ckpt, slim.get_trainable_variables(), ignore_missing_vars=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        if FLAGS.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        
        data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
                                         input_size=FLAGS.input_size,
                                         batch_size=FLAGS.batch_size_per_gpu)
        
        """
        data_generator = synth.get_batch(num_workers=FLAGS.num_readers,
                                         input_size=FLAGS.input_size,
                                         batch_size=FLAGS.batch_size_per_gpu)
        """
        start = time.time()
        for step in range(FLAGS.max_steps):
            data = next(data_generator)
            inp_dict = {input_images: data[0],
                        input_score_maps: data[2],
                        input_geo_maps: data[3],
                        input_training_masks: data[4],
                        input_transform_matrix: data[5],
                        input_box_widths: data[7],
                        input_transcription: data[8]}

            for i in range(FLAGS.batch_size_per_gpu):
                inp_dict[input_box_masks[i]] = data[6][i]


            dl, rl, tl,  _ = sess.run([d_loss, r_loss, total_loss, train_op], feed_dict=inp_dict)
            if np.isnan(tl):
                print('Loss diverged, stop training')
                break
            

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start)/10
                avg_examples_per_second = (10 * FLAGS.batch_size_per_gpu)/(time.time() - start)
                start = time.time()
                print('Step {:06d}, detect_loss {:.4f}, recognize_loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                    step, dl, rl, tl, avg_time_per_step, avg_examples_per_second))
                
                """
                print "recognition results: "
                for pred in result:
                    print icdar.ground_truth_to_word(pred)
                """
            
            if step % FLAGS.save_checkpoint_steps == 0:
                saver.save(sess, FLAGS.checkpoint_path + 'model.ckpt', global_step=global_step)

            if step % FLAGS.save_summary_steps == 0:
                """
                _, tl, summary_str = sess.run([train_op, total_loss, summary_op], feed_dict={input_images: data[0],
                                                                                             input_score_maps: data[2],
                                                                                             input_geo_maps: data[3],
                                                                                             input_training_masks: data[4]})
                """
                dl, rl, tl, _, summary_str = sess.run([d_loss, r_loss, total_loss, train_op, summary_op], feed_dict=inp_dict)
                
                summary_writer.add_summary(summary_str, global_step=step)

if __name__ == '__main__':
    tf.app.run()
