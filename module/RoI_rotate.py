import sys
from .stn import spatial_transformer_network as transformer
sys.path.append("..")
import numpy as np
import cv2
import tensorflow as tf
import math
# import icdar
import config
import os


class RoIRotate(object):
	def __init__(self, height=8):
		self.height = height

	def roi_rotate_tensor(self, feature_map, transform_matrixs, box_masks, box_widths, is_debug=False):
		"""
		param:
		feature_map: N * H * W * C
		transform_matrixs: N' * 6
		box_masks: list of tensor N'
		box_widths: N'
		"""
		with tf.variable_scope("RoIrotate"):
			max_width = box_widths[tf.argmax(box_widths, 0, output_type=tf.int32)]
			box_widths = tf.cast(box_widths, tf.float32)
			tile_feature_maps = []
			# crop_boxes = []
			# crop_sizes = []
			# box_inds = []
			map_shape = tf.shape(feature_map)
			map_shape = tf.cast(map_shape, tf.float32)

			for i, mask in enumerate(box_masks): # box_masks is a list of num of rois in each feature map
				_feature_map = feature_map[i]
				# _crop_box = tf.constant([0, 0, 8/map_shape[0], box_widths[i]/map_shape[1]])
				# _crop_size = tf.constant([8, tf.cast(box_widths[i], tf.int32)])
				_feature_map = tf.expand_dims(_feature_map, axis=0)
				box_nums = tf.shape(mask)[0]
				_feature_map = tf.tile(_feature_map, [box_nums, 1, 1, 1])
				# crop_boxes.append(_crop_box)
				# crop_sizes.append(_crop_size)
				tile_feature_maps.append(_feature_map)
				# box_inds.append(i)

			tile_feature_maps = tf.concat(tile_feature_maps, axis=0) # N' * H * W * C where N' = N * B
			norm_box_widths = box_widths / map_shape[2]
			ones = tf.ones_like(norm_box_widths)
			norm_box_heights = ones * (8.0 / map_shape[1])
			zeros = tf.zeros_like(norm_box_widths)
			crop_boxes = tf.transpose(tf.stack([zeros, zeros, norm_box_heights, norm_box_widths]))
			"""
			box_height = ones * 8
			box_height = tf.cast(box_height, tf.int32)
			box_width = ones * max_width
			box_width = tf.cast(box_width, tf.int32)
			"""
			crop_size = tf.transpose(tf.stack([8, max_width]))
			# crop_boxes = tf.stack(crop_boxes, axis=0)
			# crop_sizes = tf.stack(crop_sizes, axis=0)

			trans_feature_map = transformer(tile_feature_maps, transform_matrixs)

			# box_inds = tf.concat(box_masks, axis=0)
			box_inds = tf.range(tf.shape(trans_feature_map)[0])
			rois = tf.image.crop_and_resize(trans_feature_map, crop_boxes, box_inds, crop_size)

			pad_rois = tf.image.pad_to_bounding_box(rois, 0, 0, 8, max_width)

			print("pad_rois: ", pad_rois)

			return pad_rois

	def roi_rotate_tensor_pad(self, feature_map, transform_matrixs, box_masks, box_widths):
		with tf.variable_scope("RoIrotate"):
			max_width = box_widths[tf.argmax(box_widths, 0, output_type=tf.int32)]
			# box_widths = tf.cast(box_widths, tf.float32)
			tile_feature_maps = []
			# crop_boxes = []
			# crop_sizes = []
			# box_inds = []
			map_shape = tf.shape(feature_map)
			map_shape = tf.cast(map_shape, tf.float32)

			for i, mask in enumerate(box_masks): # box_masks is a list of num of rois in each feature map
				_feature_map = feature_map[i]
				# _crop_box = tf.constant([0, 0, 8/map_shape[0], box_widths[i]/map_shape[1]])
				# _crop_size = tf.constant([8, tf.cast(box_widths[i], tf.int32)])
				_feature_map = tf.expand_dims(_feature_map, axis=0)
				box_nums = tf.shape(mask)[0]
				_feature_map = tf.tile(_feature_map, [box_nums, 1, 1, 1])
				# crop_boxes.append(_crop_box)
				# crop_sizes.append(_crop_size)
				tile_feature_maps.append(_feature_map)
				# box_inds.append(i)

			tile_feature_maps = tf.concat(tile_feature_maps, axis=0) # N' * H * W * C where N' = N * B
			trans_feature_map = transformer(tile_feature_maps, transform_matrixs)

			box_nums = tf.shape(box_widths)[0]
			pad_rois = tf.TensorArray(tf.float32, box_nums)
			i = 0

			def cond(pad_rois, i):
				return i < box_nums
			def body(pad_rois, i):
				_affine_feature_map = trans_feature_map[i]
				width_box = box_widths[i]
				# _affine_feature_map = tf.expand_dims(_affine_feature_map, 0)
				# roi = tf.image.crop_and_resize(after_transform, [[0, 0, 8/map_shape[0], width_box/map_shape[1]]], [0], [8, tf.cast(width_box, tf.int32)])
				roi = tf.image.crop_to_bounding_box(_affine_feature_map, 0, 0, 8, width_box)
				pad_roi = tf.image.pad_to_bounding_box(roi, 0, 0, 8, max_width)
				pad_rois = pad_rois.write(i, pad_roi)
				i += 1

				return pad_rois, i
			pad_rois, _ = tf.while_loop(cond, body, loop_vars=[pad_rois, i])
			pad_rois = pad_rois.stack()

			print("pad_rois shape: ", pad_rois)

			return pad_rois

	def roi_rotate_tensor_while(self, feature_map, transform_matrixs, box_masks, box_widths, is_debug=False):
		assert transform_matrixs.shape[-1] != 8
		"""
		Input:
			feature_map: N * H * W * C
			transform_matrixs: N' * 8
			box_masks: list of tensor N * ?
			box_widths: N'
		"""
		with tf.variable_scope("RoIrotate"):
			box_masks = tf.concat(box_masks, axis=0)
			box_nums = tf.shape(box_widths)[0]
			pad_rois = tf.TensorArray(tf.float32, box_nums)
			# after_transforms = []
			max_width = box_widths[tf.arg_max(box_widths, 0, tf.int32)]
			i = 0

			def cond(pad_rois, i):
				return i < box_nums

			def body(pad_rois, i):
				index = box_masks[i]
				matrix = transform_matrixs[i]
				_feature_map = feature_map[index]
				map_shape = tf.shape(_feature_map)
				map_shape = tf.to_float(map_shape)
				# _feature_map = feature_map[i]
				print(box_widths)
				width_box = box_widths[i]
				width_box = tf.cast(width_box, tf.float32)

				# Project transform
				after_transform = tf.contrib.image.transform(_feature_map, matrix, "BILINEAR")
				# after_transforms.append(after_transform)
				after_transform = tf.expand_dims(after_transform, 0)
				# roi = tf.image.crop_and_resize(after_transform, [[0, 0, 8/720.0, width_box/1280.0]], [0], [8, width_box])

				# roi = tf.image.crop_and_resize(after_transform, [[0, 0, 8/(config.INPUT_IMAGE_SIZE / 4.0), width_box/(config.INPUT_IMAGE_SIZE / 4.0)]], [0], [8, width_box])
				# There are some erros
				# roi = tf.image.crop_and_resize(after_transform, [[0, 0, 8/(config.INPUT_SIZE / 4.0), width_box/(config.INPUT_SIZE / 4.0)]], [0], [8, tf.cast(width_box, tf.int32)])
				roi = tf.image.crop_and_resize(after_transform, [[0, 0, 8/map_shape[0], width_box/map_shape[1]]], [0], [8, tf.cast(width_box, tf.int32)])
				#  = tf.image.crop_and_resize(after_transform, [[0, 0, 8/config.INPUT_IMAGE_SIZE, width_box/config.INPUT_IMAGE_SIZE]], [0], [8, width_box])
				# roi = tf.image.crop_and_resize(after_transform, [[0, 0, 8/128.0, width_box/128.0]], [0], [8, width_box])
				pad_roi = tf.image.pad_to_bounding_box(roi, 0, 0, 8, max_width)
				pad_rois = pad_rois.write(i, pad_roi)
				i += 1

				return pad_rois, i

			pad_rois, _ = tf.while_loop(cond, body, loop_vars=[pad_rois, i])
			pad_rois = pad_rois.stack()
			pad_rois = tf.squeeze(pad_rois, axis=1)
			return pad_rois

def dummy_input():

	folder_path = "../training_samples"

	input_imgs = []
	box_widths = []
	box_masks = []
	transform_matrixs = []

	fea_h = []
	fea_w = []

	for i in range(2):
		box_num = 0
		img = cv2.imread(os.path.join(folder_path, "img_" + str(i+1) + ".jpg"))
		gt_file = open(os.path.join(folder_path, "img_" + str(i+1) + ".txt"), "rb")
		input_imgs.append(img)
		fea_h.append(img.shape[0])
		fea_w.append(img.shape[1])
		box_mask = []
		for line in gt_file.readlines():
			box_num += 1

			# line = gt_file.readline()

			info = line.split(",")

			x1, y1, x2, y2, x3, y3, x4, y4 = map(eval, info[:8])

			rotated_rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
			box_w, box_h = rotated_rect[1][0], rotated_rect[1][1]

			map_w = img.shape[1]
			map_h = img.shape[0]

			if box_w <= box_h:
				box_w, box_h = box_h, box_w

			mapped_x1, mapped_y1 = (0, 0)
			mapped_x4, mapped_y4 = (0, 8)

			width_box = math.ceil(8 * box_w / box_h)
			width_box = int(min(width_box, map_w)) # not to exceed feature map's width

			mapped_x2, mapped_y2 = (width_box, 0)
			# mapped_x3, mapped_y3 = (width_box, 8)

			# src_pts = np.float32([(x1, y1), (x2, y2),(x3, y3), (x4, y4)])
			src_pts = np.float32([(x1, y1), (x2, y2), (x4, y4)])
			# dst_pts = np.float32([(mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x3, mapped_y3), (mapped_x4, mapped_y4)])
			dst_pts = np.float32([(mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)])

			affine_matrix = cv2.getAffineTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
			affine_matrix = affine_matrix.flatten()
			# project_matrix = cv2.getPerspectiveTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
			# project_matrix = project_matrix.flatten()[:8]

			box_widths.append(width_box)
			box_mask.append(i)
			transform_matrixs.append(affine_matrix)
			# transform_matrixs.append(project_matrix)
		box_masks.append(box_mask)
	input_imgs = np.array(input_imgs)
	fea_h = np.array(fea_h)
	fea_w = np.array(fea_w)
	transform_matrixs = np.array(transform_matrixs)
	# box_masks = np.array(box_masks)
	box_widths = np.array(box_widths)
	return input_imgs, fea_h, fea_w, transform_matrixs, box_masks, box_widths

def check_RoIRotate(RR):
	# RR = RoIRotate()
	input_feature_map = tf.placeholder(tf.float32, shape=[None, None, None, 3])
	input_transform_matrix = tf.placeholder(tf.float32, shape=[None, 6])
	input_feature_height = tf.placeholder(tf.int32, shape=[None])
	input_feature_width = tf.placeholder(tf.int32, shape=[None])
	# input_box_mask = tf.placeholder(tf.int32, shape=[None])
	input_box_masks = []
	input_box_widths = tf.placeholder(tf.int32, shape=[None])
	input_box_nums = tf.placeholder(tf.int32)

	for i in range(2): # Batch size is 2
		input_box_masks.append(tf.placeholder(tf.int32, shape=[None]))

	# pad_rois = RR.roi_rotate_tensor(input_feature_map, input_transform_matrix, input_box_masks, input_box_widths)
	pad_rois = RR.roi_rotate_tensor_pad(input_feature_map, input_transform_matrix, input_box_masks, input_box_widths)

	data = dummy_input()
	for i in range(6):
		if i != 4:
			print(data[i].shape)

	with tf.Session() as sess:
		inp_dict = {input_feature_map: data[0], input_feature_height: data[1], input_feature_width: data[2], input_transform_matrix: data[3], input_box_widths: data[5]}
		for i in range(2):
			inp_dict[input_box_masks[i]] = data[4][i]
		result_rois = sess.run(pad_rois, feed_dict=inp_dict)
		# output_rois = np.squeeze(output_rois)
		# print(len(transforms))
	return result_rois

if __name__ == '__main__':
	RR = RoIRotate()
	output = check_RoIRotate(RR)

	for i in range(output.shape[0]):
		cv2.imwrite("out_" + str(i) + ".jpg", output[i])
