CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-~`<>'.:;^/|!?$%#@&*()[]{}_+=,\\\""
NUM_CLASSES = len(CHAR_VECTOR) + 1
PRETRAINED_MODEL_PATH = "nets/resnet_v1_50.ckpt"
INPUT_IMAGE_PATH = "training_samples" # The root path of the training images
INPUT_GT_PATH = "training_samples" # The root path of the training ground-truth text file.
# ICDAR15_DATA_PATH = "/data2/data/15ICDAR/ch4_training_images"
# ICDAR15_GT_DATA_PATH = "/data2/data/15ICDAR/ch4_training_localization_transcription_gt_rec"
# Synth800K_DATA_PATH = "/data2/data/SynthText"
# Synth800K_GT_DATA_PATH = "/data2/data/SynthText/ground_truth_txt"
# DEMO_TRAIN_DATA_PATH = "training_samples"
RESTORE=False
SAVE_CHECKPOINT_STEP=500
SAVE_CHECKPOINT_PATH = "checkpoints/"
NUM_READERS=8 # num of threads to load data
INPUT_SIZE=512
BATCH_SIZE=8
MAX_STEPS=100000
LEARNING_RATE=0.0001
min_text_size=5 # if the text size is smaller than this, we ignore it during training
min_crop_side_ratio=0.1 # when doing random crop from input image, the min length of min(H, W')


