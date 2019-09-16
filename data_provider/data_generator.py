import time
import os
import random
import numpy as np
import tensorflow as tf
import cv2
from itertools import compress
from data_provider.data_utils import check_and_validate_polys, crop_area, rotate_image, generate_rbox, get_project_matrix_and_width, sparse_tuple_from, crop_area_fix
from data_provider.ICDAR_loader import ICDARLoader
# from data_provider.SynthText_loader import SynthTextLoader
from data_provider.data_enqueuer import GeneratorEnqueuer


def generator(input_images_dir, input_gt_dir, input_size=512, batch_size=12, random_scale=np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2]),):
    # data_loader = SynthTextLoader()
    data_loader = ICDARLoader(edition='13', shuffle=True)
    # image_list = np.array(data_loader.get_images(FLAGS.training_data_dir))
    image_list = np.array(data_loader.get_images(input_images_dir))
    # print('{} training images in {} '.format(image_list.shape[0], FLAGS.training_data_dir))
    index = np.arange(0, image_list.shape[0])
    while True:
        np.random.shuffle(index)
        batch_images = []
        batch_image_fns = []
        batch_score_maps = []
        batch_geo_maps = []
        batch_training_masks = []

        batch_text_polyses = [] 
        batch_text_tagses = []
        batch_boxes_masks = []

        batch_text_labels = []
        count = 0
        for i in index:
            try:
                im_fn = image_list[i]
                # print(im_fn)
                # if im_fn.split(".")[0][-1] == '0' or im_fn.split(".")[0][-1] == '2':
                #     continue
                im = cv2.imread(os.path.join(input_images_dir, im_fn))
                h, w, _ = im.shape
                file_name = "gt_" + im_fn.replace(os.path.basename(im_fn).split('.')[1], 'txt').split('/')[-1]
                # file_name = im_fn.replace(im_fn.split('.')[1], 'txt') # using for synthtext
                # txt_fn = os.path.join(FLAGS.training_gt_data_dir, file_name)
                txt_fn = os.path.join(input_gt_dir, file_name)
                if not os.path.exists(txt_fn):
                    print('text file {} does not exists'.format(txt_fn))
                    continue
                print(txt_fn)
                text_polys, text_tags, text_labels = data_loader.load_annotation(txt_fn) # Change for load text transiption
                
                if text_polys.shape[0] == 0:
                    continue
                
                text_polys, text_tags, text_labels = check_and_validate_polys(text_polys, text_tags, text_labels, (h, w))

                ############################# Data Augmentation ##############################
                # random scale this image
                rd_scale = np.random.choice(random_scale)
                im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
                text_polys *= rd_scale

                # rotate image from [-10, 10]
                angle = random.randint(-10, 10)
                im, text_polys = rotate_image(im, text_polys, angle)

                # 600×600 random samples are cropped.
                im, text_polys, text_tags, selected_poly = crop_area(im, text_polys, text_tags, crop_background=False)
                # im, text_polys, text_tags, selected_poly = crop_area_fix(im, text_polys, text_tags, crop_size=(600, 600))
                text_labels = [text_labels[i] for i in selected_poly]
                if text_polys.shape[0] == 0 or len(text_labels) == 0:
                    continue

                # pad the image to the training input size or the longer side of image
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
                text_polys[:, :, 0] *= resize_ratio_3_x
                text_polys[:, :, 1] *= resize_ratio_3_y
                new_h, new_w, _ = im.shape

                score_map, geo_map, training_mask, rectangles = generate_rbox((new_h, new_w), text_polys, text_tags)

                mask = [not (word == [-1]) for word in text_labels]
                text_labels = list(compress(text_labels, mask))
                rectangles = list(compress(rectangles, mask))

                assert len(text_labels) == len(rectangles), "rotate rectangles' num is not equal to text label"

                if len(text_labels) == 0:
                    continue

                boxes_mask = np.array([count] * len(rectangles))

                count += 1

                batch_images.append(im[:, :, ::-1].astype(np.float32))
                batch_image_fns.append(im_fn)
                batch_score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                batch_geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                batch_training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))

                batch_text_polyses.append(rectangles)
                batch_boxes_masks.append(boxes_mask)
                batch_text_labels.extend(text_labels)
                batch_text_tagses.append(text_tags)

                if len(batch_images) == batch_size:
                    batch_text_polyses = np.concatenate(batch_text_polyses)
                    batch_text_tagses = np.concatenate(batch_text_tagses)
                    batch_transform_matrixes, batch_box_widths = get_project_matrix_and_width(batch_text_polyses, batch_text_tagses)
                    # TODO limit the batch size of recognition 
                    batch_text_labels_sparse = sparse_tuple_from(np.array(batch_text_labels))

                    # yield images, image_fns, score_maps, geo_maps, training_masks
                    yield batch_images, batch_image_fns, batch_score_maps, batch_geo_maps, batch_training_masks, batch_transform_matrixes, batch_boxes_masks, batch_box_widths, batch_text_labels_sparse, batch_text_polyses, batch_text_labels
                    batch_images = []
                    batch_image_fns = []
                    batch_score_maps = []
                    batch_geo_maps = []
                    batch_training_masks = []
                    batch_text_polyses = [] 
                    batch_text_tagses = []
                    batch_boxes_masks = []
                    batch_text_labels = []
                    count = 0
            except Exception as e:
                import traceback
                print(im_fn)
                traceback.print_exc()
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        print('Generator use 10 batches for buffering, this may take a while, you can tune this yourself.')
        enqueuer.start(max_queue_size=10, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


def test():
    font = cv2.FONT_HERSHEY_SIMPLEX
    dg = get_batch(num_workers=1, input_size=512, batch_size=4)
    for iter in range(2000):
        print("iter: ", iter)
        data = next(dg)
        imgs = data[0]
        imgs_name = data[1]
        polygons = data[-2]
        labels = data[-1]
        masks = data[6]
        prev_start_index = 0
        for i, (img, mask, img_name) in enumerate(zip(imgs, masks, imgs_name)):
            # img_name = ''
            im = img.copy()
            poly_start_index = len(masks[i-1])
            poly_end_index = len(masks[i-1]) + len(mask)
            for poly, la,  in zip(polygons[prev_start_index:(prev_start_index+len(mask))], labels[prev_start_index:prev_start_index+len(mask)]):
                cv2.polylines(img, [poly.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
                # trans = ground_truth_to_word(la)
                # img_name = img_name + trans + '_'
            img_name = img_name[:-1] + '.jpg'
            cv2.imwrite("./polygons/" + os.path.basename(img_name), img)

            prev_start_index += len(mask)

if __name__ == '__main__':
    test()
