import sys
sys.path.append("..")
import numpy as np
import cv2
import tensorflow as tf
import math
# import icdar
import config
import os

def gt_to_boxes(file_path):
	boxes = []
	with open(file_path) as f:
		for line in f.readlines():
			info = line.split(",")
			label = info[-1]
			x1, y1, x2, y2, x3, y3, x4, y4 = map(eval, info[:8])
			box = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
			boxes.append(box)
	boxes = np.array(boxes)

	return boxes

def param2theta(param, w, h):
	param = np.vstack([param, [0, 0, 1]])
	param = np.linalg.inv(param)

	theta = np.zeros([2, 3])
	theta[0, 0] = param[0, 0]
	theta[0, 1] = param[0, 1] * h / w
	theta[0, 2] = param[0, 2] * 2 / w + theta[0, 0] + theta[0, 1] - 1
	theta[1, 0] = param[1, 0] * w / h
	theta[1, 1] = param[1, 1]
	theta[1, 2] = param[1, 2] * 2 / h + theta[1, 0] + theta[1, 1] - 1
	return theta

class RoIRotate(object):
	def __init__(self, height=8):
		self.height = height

	"""
	def compute_affine_param(self, feature_map, boxes, boxes_mask):
		
		num_boxes = boxes.shape[0]
		
		affine_matrixs = []
		box_widths = []
		max_width = 0

		for box_index in range(num_boxes):
			box = boxes[box_index]
			
			# 512 -> 128
			x1 = box[0][0] / 4
			y1 = box[0][1] / 4
			x2 = box[1][0] / 4
			y2 = box[1][1] / 4
			x3 = box[2][0] / 4
			y3 = box[2][1] / 4
			x4 = box[3][0] / 4
			y4 = box[3][1] / 4
			
			
			# x1 = box[0][0]
			# y1 = box[0][1]
			# x2 = box[1][0]
			# y2 = box[1][1]
			# x3 = box[2][0]
			# y3 = box[2][1]
			# x4 = box[3][0]
			# y4 = box[3][1]
			
	
			feature_map_width = 128
			feature_map_height = 128
			# feature_map_width = feature_map.shape[1]
			# feature_map_height = feature_map.shape[0]

			rotated_rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
			# rotated_rect = cv2.minAreaRect(box)
			box_w, box_h = rotated_rect[1][0], rotated_rect[1][1]

			if box_w <= box_h:
				box_w, box_h = box_h, box_w

			mapped_x1, mapped_y1 = (0, 0)
			mapped_x4, mapped_y4 = (0, self.height)

			width_box = math.ceil(self.height * box_w / box_h)
			width_box = min(width_box, feature_map_width) # not to exceed feature map's width
			max_width = width_box if width_box > max_width else max_width

			mapped_x2, mapped_y2 = (width_box, 0)

			src_pts = np.float32([(x1, y1), (x2, y2),(x4, y4)])
			dst_pts = np.float32([(mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)])

			affine_matrix = cv2.getAffineTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))

			# affine_matrix = np.vstack([affine_matrix, [0, 0, 1]])
			# affine_matrix = 

			affine_matrixs.append(affine_matrix)
			box_widths.append(width_box)

		return affine_matrixs, box_widths, max_width
	"""

	def compute_affine_param(self, feature_map, boxes, boxes_mask, is_debug=False):

		max_width = 0
		boxes_widths = []
		affine_matrixes = []
		feature_maps = []

		for index, box in zip(boxes_mask, boxes):
		# for i in range(boxes.shape[0]):
			# index = boxes_mask[i]
			# box = boxes[i]
			_feature_map = feature_map[index]
			feature_maps.append(_feature_map)

			if is_debug:
				x1, y1, x2, y2, x3, y3, x4, y4 = box 
			else:
				x1, y1, x2, y2, x3, y3, x4, y4 = box / 4 


			rotated_rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
			box_w, box_h = rotated_rect[1][0], rotated_rect[1][1]

			map_w = _feature_map.shape[1]
			map_h = _feature_map.shape[0]

			if box_w <= box_h:
				box_w, box_h = box_h, box_w

			mapped_x1, mapped_y1 = (0, 0)
			mapped_x4, mapped_y4 = (0, self.height)

			width_box = math.ceil(self.height * box_w / box_h)
			width_box = min(width_box, map_w) # not to exceed feature map's width
			max_width = width_box if width_box > max_width else max_width

			mapped_x2, mapped_y2 = (width_box, 0)

			src_pts = np.float32([(x1, y1), (x2, y2),(x4, y4)])
			dst_pts = np.float32([(mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)])

			affine_matrix = cv2.getAffineTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))

			affine_matrixes.append(affine_matrix)
			boxes_widths.append(int(width_box))

		return feature_maps, affine_matrixes, boxes_widths, max_width


	def apply_affine(self, feature_maps, affine_matrixes, boxes_widths, max_width, is_opencv=True):

		affine_maps = []
		boxes = []
		sizes = []
		pad_rois = []


		f_width = feature_maps[0].shape[1]
		f_height = feature_maps[0].shape[0]
		f_channel = feature_maps[0].shape[2]

		for fm, am, width in zip(feature_maps, affine_matrixes, boxes_widths):
			affine_map = cv2.warpAffine(fm, am, (f_width, f_height)) # H * W * C

			if is_opencv:
				roi = affine_map[0:8, 0:width, :]
				# roi = tf.constant(value=roi, dtype=tf.float32)
				pad_roi = cv2.copyMakeBorder(roi, 0, 0, 0, int(max_width - width), cv2.BORDER_CONSTANT, value=0)
			else:
				# Convert to tensor
				affine_map = tf.constant(value=np.array([affine_map]), dtype=tf.float32)
				crop_box = tf.constant(value=[[0, 0, 8.0/fm.shape[1], width/fm.shape[2]]], dtype=tf.float32)
				crop_size = tf.constant(value=[8, width], dtype=tf.int32)

				# Crop and resize
				roi = tf.image.crop_and_resize(affine_map, crop_box, [0], crop_size)
				pad_roi = tf.image.pad_to_bounding_box(roi, 0, 0, 8, int(max_width))

			affine_maps.append(affine_map)
			pad_rois.append(pad_roi)
		

		"""
		for fm, am in zip(feature_maps, affine_matrixs):
			affine_map = cv2.warpAffine(fm, am, (fm.shape[2], fm.shape[1]))
			# affine_map = tf.constant(value=affine_map, dtype=tf.float32)
			affine_maps.append(affine_map)

		crop_map = tf.constant(value=np.array(affine_maps), dtype=tf.float32)
		boxes_mask = np.arange(len(feature_maps))
		crop_box_index = tf.constant(value=boxes_mask, dtype=tf.int32)

		for width in boxes_widths:
			boxes.append([0, 0, 8.0/f_height, width/f_width])
			sizes.append([8, width])

		crop_box = tf.constant(value=boxes, dtype=tf.float32)
		crop_size = tf.constant(value=sizes, dtype=tf.int32)

		rois = tf.image.crop_and_resize(crop_map, crop_box, crop_box_index, crop_size)
		"""

		return pad_rois, affine_maps 

	def roi_rotate(self, feature_map, boxes, boxes_mask):
		feature_maps, affine_matrixes, boxes_widths, max_width = self.compute_affine_param(feature_map, boxes, boxes_mask, is_debug=False)
		pad_rois, _ = self.apply_affine(feature_maps, affine_matrixes, boxes_widths, max_width)

		return pad_rois, boxes_widths

	def roi_rotate_tensor(self, feature_map, transform_matrixs, box_masks, box_widths, box_nums, is_debug=False):
		with tf.variable_scope("RoIrotate"): 
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
				print box_widths
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
			
			"""
			# for i in range(box_nums):
			while i < box_nums:
				index = box_masks[i]
				matrix = transform_matrixs[i]
				_feature_map = feature_map[index]
				# _feature_map = feature_map[i]
				print box_widths
				width_box = box_widths[i]

				# Project transform
				after_transform = tf.contrib.image.transform(_feature_map, matrix, "BILINEAR")
				after_transforms.append(after_transform)
				roi = tf.image.crop_and_resize([after_transform], [[0, 0, 8/720.0, width_box/1280.0]], [0], [8, width_box])
				pad_roi = tf.image.pad_to_bounding_box(roi, 0, 0, 8, max_width)

				pad_rois.append(pad_roi)

				i += 1
			"""
			pad_rois = pad_rois.stack()
			pad_rois = tf.squeeze(pad_rois, axis=1)
			return pad_rois

def dummy_input():

	folder_path = "../training_samples"

	input_imgs = []
	box_widths = []
	box_masks = []
	transform_matrixs = []
	box_num = 0
	fea_h = []
	fea_w = []

	for i in range(2):
		img = cv2.imread(os.path.join(folder_path, "img_" + str(i+1) + ".jpg"))
		gt_file = open(os.path.join(folder_path, "gt_img_" + str(i+1) + ".txt"), "rb")
		input_imgs.append(img)
		fea_h.append(img.shape[0])
		fea_w.append(img.shape[1])

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
			mapped_x3, mapped_y3 = (width_box, 8)

			src_pts = np.float32([(x1, y1), (x2, y2),(x3, y3), (x4, y4)])
			dst_pts = np.float32([(mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x3, mapped_y3), (mapped_x4, mapped_y4)])

			project_matrix = cv2.getPerspectiveTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
			project_matrix = project_matrix.flatten()[:8]

			box_widths.append(width_box)
			box_masks.append(i)
			transform_matrixs.append(project_matrix)

	input_imgs = np.array(input_imgs)
	fea_h = np.array(fea_h)
	fea_w = np.array(fea_w)
	transform_matrixs = np.array(transform_matrixs)
	box_masks = np.array(box_masks)
	box_widths = np.array(box_widths)
	return input_imgs, fea_h, fea_w, transform_matrixs, box_masks, box_widths, box_num

def check_RoIRotate(RR):
	# RR = RoIRotate()
	input_feature_map = tf.placeholder(tf.float32, shape=[None, None, None, 3])
	input_transform_matrix = tf.placeholder(tf.float32, shape=[None, 8])
	input_feature_height = tf.placeholder(tf.int32, shape=[None])
	input_feature_width = tf.placeholder(tf.int32, shape=[None])
	input_box_mask = tf.placeholder(tf.int32, shape=[None])
	input_box_widths = tf.placeholder(tf.int32, shape=[None])
	input_box_nums = tf.placeholder(tf.int32)

	pad_rois = RR.roi_rotate_tensor(input_feature_map, input_transform_matrix, input_box_mask, input_box_widths, input_box_nums)

	data = dummy_input()
	for i in range(6):
		print data[i].shape

	print data[6]
	with tf.Session() as sess:
		output_rois = sess.run([pad_rois], feed_dict={input_feature_map: data[0], input_feature_height: data[1], input_feature_width: data[2], input_transform_matrix: data[3], input_box_mask: data[4], input_box_widths: data[5], input_box_nums: data[6]})
		output_rois = np.squeeze(output_rois)
		# print len(transforms)
	return output_rois

if __name__ == '__main__':
	RR = RoIRotate()

	"""
	test_img = cv2.imread("../training_samples/img_1.jpg")
	width = test_img.shape[1]
	height = test_img.shape[0]
	boxes = gt_to_boxes("../training_samples/img_1.txt")

	affine_matrixs = RR.compute_affine_param(test_img, boxes)
	print "affine_matrixs size: ", len(affine_matrixs)
	print affine_matrixs[0]

	iii = cv2.warpAffine(test_img, affine_matrixs[0], (width, height))
	cv2.imwrite("aff_image.jpg", iii)
	

	image_list = np.array(["../training_samples/img_1.jpg", "../training_samples/img_2.jpg"])

	data = icdar.get_batch_without_aug(image_list=image_list, batch_size=2)

	images = data[0]
	# boxes = np.concatenate(data[5])
	# boxes_mask = np.concatenate(data[6])

	feature_maps, affine_matrixs, boxes_widths, max_width = RR.compute_affine_param(images, data[5], data[6], is_debug=True)

	rois, affine_maps = RR.apply_affine(feature_maps, affine_matrixs, boxes_widths, max_width) 

	with tf.Session() as sess:
		index = 0
		for roi in rois:
			ccc = sess.run(roi)
			print ccc.shape

			# cv2.imwrite(str(index)+".jpg", ccc)
			index += 1
	"""

	output = check_RoIRotate(RR)

	for i in range(output.shape[0]):
		cv2.imwrite("out_" + str(i) + ".jpg", output[i])
