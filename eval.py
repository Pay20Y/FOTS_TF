import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf

import locality_aware_nms as nms_locality
import lanms
import Levenshtein

# tf.app.flags.DEFINE_string('test_data_path', 'training_samples/', '')
tf.app.flags.DEFINE_string('test_data_path', '/data2/data/15ICDAR/test/image/', '')
tf.app.flags.DEFINE_string('gpu_list', '1', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints/', '')
tf.app.flags.DEFINE_string('output_dir', 'outputs/', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')
tf.app.flags.DEFINE_bool('use_vacab', True, 'strong, normal or weak')

from module import Backbone_branch, Recognition_branch, RoI_rotate
from icdar import restore_rectangle, ground_truth_to_word

FLAGS = tf.app.flags.FLAGS
detect_part = Backbone_branch.Backbone(is_training=False)
roi_rotate_part = RoI_rotate.RoIRotate()
recognize_part = Recognition_branch.Recognition(is_training=False)
font = cv2.FONT_HERSHEY_SIMPLEX

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files
"""
# It seems do not need it
def resize_image(im, max_side_len=2400, input_size=512):
    new_h, new_w, _ = im.shape
    max_h_w_i = np.max([new_h, new_w, input_size])
    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
    im_padded[:new_h, :new_w, :] = im.copy()
    im = im_padded
    # resize the image to input size
    new_h, new_w, _ = im.shape
    resize_h = input_size
    resize_w = input_size
    im = cv2.resize(im, dsize=(resize_w, resize_h))

    resize_ratio_3_x = resize_w/float(new_w)
    resize_ratio_3_y = resize_h/float(new_h)

    return im, (resize_ratio_3_y, resize_ratio_3_x)
"""

def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer
"""
def get_project_matrix_and_width(text_polyses, target_height=8.0):
    project_matrixes = []
    box_widths = []
    filter_box_masks = []
    # max_width = 0
    max_width = 0

    for i in range(text_polyses.shape[0]):
        x1, y1, x2, y2, x3, y3, x4, y4 = text_polyses[i] / 4

        rotated_rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
        box_w, box_h = rotated_rect[1][0], rotated_rect[1][1]

        if box_w <= box_h:
            box_w, box_h = box_h, box_w

        mapped_x1, mapped_y1 = (0, 0)
        mapped_x4, mapped_y4 = (0, 8)

        width_box = math.ceil(8 * box_w / box_h)
        width_box = int(min(width_box, 128)) # not to exceed feature map's width
        # width_box = int(min(width_box, 512)) # not to exceed feature map's width

        if width_box > max_width: 
            max_width = width_box 

        mapped_x2, mapped_y2 = (width_box, 0)
        mapped_x3, mapped_y3 = (width_box, 8)

        src_pts = np.float32([(x1, y1), (x2, y2),(x3, y3), (x4, y4)])
        dst_pts = np.float32([(mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x3, mapped_y3), (mapped_x4, mapped_y4)])

        project_matrix = cv2.getPerspectiveTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
        project_matrix = project_matrix.flatten()[:8]

        project_matrixes.append(project_matrix)
        box_widths.append(width_box)

    project_matrixes = np.array(project_matrixes)
    box_widths = np.array(box_widths)

    return project_matrixes, box_widths, max_width
"""
def get_project_matrix_and_width(text_polyses, target_height=8.0):
    project_matrixes = []
    box_widths = []
    filter_box_masks = []
    # max_width = 0
    # max_width = 0

    for i in range(text_polyses.shape[0]):
        x1, y1, x2, y2, x3, y3, x4, y4 = text_polyses[i] / 4

        rotated_rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
        box_w, box_h = rotated_rect[1][0], rotated_rect[1][1]

        if box_w <= box_h:
            box_w, box_h = box_h, box_w

        mapped_x1, mapped_y1 = (0, 0)
        mapped_x4, mapped_y4 = (0, 8)

        width_box = math.ceil(8 * box_w / box_h)
        width_box = int(min(width_box, 128)) # not to exceed feature map's width
        # width_box = int(min(width_box, 512)) # not to exceed feature map's width
        """
        if width_box > max_width: 
            max_width = width_box 
        """
        mapped_x2, mapped_y2 = (width_box, 0)
        # mapped_x3, mapped_y3 = (width_box, 8)

        src_pts = np.float32([(x1, y1), (x2, y2), (x4, y4)])
        dst_pts = np.float32([(mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)])
        affine_matrix = cv2.getAffineTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
        affine_matrix = affine_matrix.flatten()

        # project_matrix = cv2.getPerspectiveTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
        # project_matrix = project_matrix.flatten()[:8]

        project_matrixes.append(affine_matrix)
        box_widths.append(width_box)

    project_matrixes = np.array(project_matrixes)
    box_widths = np.array(box_widths)

    return project_matrixes, box_widths

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

def find_similar_word(input_str, word_set):
    min_distance = 10000
    best_word = input_str
    for word in word_set:
        dist = Levenshtein.distance(input_str, word)
        if dist < min_distance:
            min_distance = dist
            best_word = word

    return best_word

def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    word_set = set()
    if FLAGS.use_vacab:
        with open("vocab.txt", "r") as f:
            for line in f.readlines():
                line = line.strip()
                word_set.add(line)


    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        input_transform_matrix = tf.placeholder(tf.float32, shape=[None, 6], name='input_transform_matrix')
        # input_box_mask = tf.placeholder(tf.int32, shape=[None], name='input_box_mask')
        input_box_mask = []
        input_box_mask.append(tf.placeholder(tf.int32, shape=[None], name='input_box_masks_0'))
        input_box_widths = tf.placeholder(tf.int32, shape=[None], name='input_box_widths')
        # input_box_nums = tf.placeholder(tf.int32, name='input_box_nums')
        # input_seq_len = tf.placeholder(tf.int32, shape=[None], name='input_seq_len')
        input_seq_len = input_box_widths[tf.argmax(input_box_widths, 0)] * tf.ones_like(input_box_widths)
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        shared_feature, f_score, f_geometry = detect_part.model(input_images)
        pad_rois = roi_rotate_part.roi_rotate_tensor(shared_feature, input_transform_matrix, input_box_mask, input_box_widths)
        recognition_logits = recognize_part.build_graph(pad_rois, input_seq_len)
        _, dense_decode = recognize_part.decode(recognition_logits, input_seq_len)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            for im_fn in im_fn_list:
                im = cv2.imread(im_fn)[:, :, ::-1]
                start_time = time.time()
                im_resized, (ratio_h, ratio_w) = resize_image(im)
                # im_resized_d, (ratio_h_d, ratio_w_d) = resize_image_detection(im)

                timer = {'net': 0, 'restore': 0, 'nms': 0}
                start = time.time()
                score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                
                boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
                
                """
                if boxes is not None:
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h
                """
                # save to file
                if boxes is not None and boxes.shape[0] != 0:
                    res_file = os.path.join(
                        FLAGS.output_dir,
                        'res_' + '{}.txt'.format(
                            os.path.basename(im_fn).split('.')[0]))

                    input_roi_boxes = boxes[:, :8].reshape(-1, 8)
                    # input_roi_boxes = boxes[:, :8].reshape((-1, 4, 2))
                    
                    # input_roi_boxes = boxes.copy()
                    # input_roi_boxes[:, :, 0] *= ratio_w
                    # input_roi_boxes[:, :, 1] *= ratio_h
                    # input_roi_boxes = input_roi_boxes.reshape((-1, 8))
                    # boxes_masks = np.array([0] * input_roi_boxes.shape[0])
                    boxes_masks = [0] * input_roi_boxes.shape[0]
                    transform_matrixes, box_widths = get_project_matrix_and_width(input_roi_boxes)
                    # max_box_widths = max_width * np.ones(boxes_masks.shape[0]) # seq_len

                    # Run end to end
                    recog_decode = sess.run(dense_decode, feed_dict={input_images: [im_resized], input_transform_matrix: transform_matrixes, input_box_mask[0]: boxes_masks, input_box_widths: box_widths})
                    timer['net'] = time.time() - start
                    
                    # Preparing for draw boxes
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h

                    # print "recognition result: "
                    # for pred in recog_decode:
                       # print ground_truth_to_word(pred)
                    if recog_decode.shape[0] != boxes.shape[0]:
                        print "detection and recognition result are not equal!"
                        exit(-1)

                    with open(res_file, 'w') as f:
                        for i, box in enumerate(boxes):
                            # to avoid submitting errors
                            box = sort_poly(box.astype(np.int32))
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                continue
                            recognition_result = ground_truth_to_word(recog_decode[i])
                            recognition_result_best = find_similar_word(recognition_result, word_set)
                            f.write('{},{},{},{},{},{},{},{},{}\r\n'.format(
                                box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1], recognition_result_best
                            ))
                            # Draw bounding box
                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
                            # Draw recognition results area
                            text_area = box.copy()
                            text_area[2, 1] = text_area[1, 1]
                            text_area[3, 1] = text_area[0, 1]
                            text_area[0, 1] = text_area[0, 1] - 15
                            text_area[1, 1] = text_area[1, 1] - 15
                            cv2.fillPoly(im[:, :, ::-1], [text_area.astype(np.int32).reshape((-1, 1, 2))], color=(255, 255, 0))
                            im_txt = cv2.putText(im[:, :, ::-1], recognition_result_best, (box[0, 0], box[0, 1]), font, 0.5, (0, 0, 255), 1)
                else:
                    timer['net'] = time.time() - start
                    res_file = os.path.join(FLAGS.output_dir, 'res_' + '{}.txt'.format(os.path.basename(im_fn).split('.')[0]))
                    f = open(res_file, "w")
                    im_txt = None
                    f.close()

                print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                    im_fn, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

                duration = time.time() - start_time
                print('[timing] {}'.format(duration))

                if not FLAGS.no_write_images:
                    img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                    # cv2.imwrite(img_path, im[:, :, ::-1])
                    if im_txt is not None:
                        cv2.imwrite(img_path, im_txt)

if __name__ == '__main__':
    tf.app.run()
