import os
import numpy as np
from .data_loader import DataLoader
from .data_utils import label_to_array

class ICDARLoader(DataLoader):
	def __init__(self, edition='13', shuffle=False):
		super(ICDARLoader, self).__init__()
		self.edition = edition
		self.shuffle = shuffle # shuffle the polygons

	def load_annotation(self, gt_file):
		text_polys = []
		text_tags = []
		labels = []
		if not os.path.exists(gt_file):
			return np.array(text_polys, dtype=np.float32)
		with open(gt_file, 'r', encoding="utf-8-sig") as f:
			for line in f.readlines():
				try:
					line = line.replace('\xef\xbb\bf', '')
					line = line.replace('\xe2\x80\x8d', '')
					line = line.strip()
					line = line.split(',')
					if self.edition == '17':
						line.pop(8) # since icdar17 has script
					# Deal with transcription containing ,
					if len(line) > 9:
						label = line[8]
						for i in range(len(line) - 9):
							label = label + "," + line[i+9]
					else:
						label = line[-1]

					temp_line = list(map(eval, line[:8]))
					x1, y1, x2, y2, x3, y3, x4, y4 = map(float, temp_line)

					text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
					if label == '*' or label == '###' or label == '':
						text_tags.append(True)
						labels.append([-1])
					else:
						labels.append(label_to_array(label))
						text_tags.append(False)
				except Exception as e:
					print(e)
					continue
		text_polys = np.array(text_polys)
		text_tags = np.array(text_tags)

		return text_polys, text_tags, labels
