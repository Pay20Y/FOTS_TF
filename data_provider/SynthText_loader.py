import os
import numpy as np
from .data_loader import DataLoader
from .data_utils import label_to_array
import scipy.io as sio

class SynthTextLoader(DataLoader):

	def check_minus(self, polygon):
		for p in polygon:
			if p < 0:
				return True
		return False
	def get_images(self, data_dir):
		gt_dict = {}
		gt_mat_path = os.path.join(data_dir, "gt.mat")
		gt_mat = sio.loadmat(gt_mat_path, gt_dict, squeeze_me=True, struct_as_record=False, variable_names=['imnames', 'wordBB', 'txt'])

		image_list = gt_dict['imnames']
		return image_list

	def load_annotation(self, gt_file):
		try:
			text_polys = []
			text_tags = []
			labels = []
			if not os.path.exists(gt_file):
				return np.array(text_polys, dtype=np.float32)
			with open(gt_file, 'r') as f:
				for line in f.readlines():
					line = line.replace('\xef\xbb\bf', '')
					line = line.replace('\xe2\x80\x8d', '')
					line = line.strip()
					line = line.split(',')
					# Deal with transcription containing ,
					if len(line) > 9:
						label = line[8]
						for i in range(len(line) - 9):
							label = label + "," + line[i+9]
					else:
						label = line[-1]

					temp_line = list(map(eval, line[:8]))
					if self.check_minus(temp_line):
						continue
					x1, y1, x2, y2, x3, y3, x4, y4 = map(float, temp_line)
					
					text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
					text_tags.append(False)
					labels.append(label_to_array(label))
		except Exception as e:
			print(e)
			print(gt_file)

		return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool), labels
