CHAR_VECTOR = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

letters = [letter for letter in CHAR_VECTOR]

num_classes = len(letters) + 1

img_w, img_h = 200, 40

# Network parameters
batch_size = 32
val_batch_size = 32

max_text_len = 10
downsample_factor = 4   #depends on number of (2,2) max pooling layers used or factor by which img_w is reduced