import cv2
import os

dataset1_images_dir = ''
dataset2_images_dir = ''
dataset3_images_dir = ''

dataset1_gt_dir = ''
dataset2_gt_dir = ''
dataset3_gt_dir = ''

target_images_dir = ''
target_gt_dir = ''

combine_dataset_images_dir = [dataset1_images_dir, dataset2_images_dir, dataset3_images_dir]
combine_dataset_gt_dir = [dataset1_gt_dir, dataset2_gt_dir, dataset3_gt_dir]

count = 0
for i, (dataset, gt) in zip(combine_dataset_images_dir, combine_dataset_gt_dir):
	print("process dataset ", i)
	count_img = 0
	count_gt = 0
	for j, path in enumerate(os.listdir(dataset)):
		img_path = os.path.join(dataset, path)
		# img = cv2.imread(img_path)
		target_img_path = os.path.join(target_images_dir, path.split('.')[0] + '_' + str(i) + ".jpg")
		# cv2.imwrite(os.path.join(target_images_dir, path.split('.')[0] + '_' + str(i) + ".jpg"))
		count_img += 1
		shutil.copyfile(img_path, target_img_path)
	for j, path in enumerate(os.listdir(gt)):
		gt_path = os.path.join(gt, path)
		target_img_path = os.path.join(target_gt_dir, path.split('.')[0] + '_' + str(i) + ".txt")
		shutil.copyfile(gt_path, target_img_path)
		count_gt += 1

	assert count_img == count_gt, "dataset images are not equal to annotations"

	count += count_img

print("Done Total: ", count)



