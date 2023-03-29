from lpips_tensorflow import learned_perceptual_metric_model
import os 
import argparse
from tqdm import tqdm
import tensorflow as tf
vgg_ckpt_fn = os.path.join( 'vgg', 'exported') 
lin_ckpt_fn = os.path.join('lin', 'exported')
cropped_img_size = 512
import numpy as np
lpips = learned_perceptual_metric_model(cropped_img_size, vgg_ckpt_fn, lin_ckpt_fn)


def benchmark(args):


    DATA_DIR = args.model_dir
    COMARE_DIR = args.generated_dir
    BATCH_SIZE = 8
    IMG_SIZE = 512
    def train_convert(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        return img
    def create_dataaset(directory, batch_size, seed=15):
        img_paths = tf.data.Dataset.list_files(str(directory))
        BUFFER_SIZE = tf.data.experimental.cardinality(img_paths)
        img_paths = img_paths.cache().shuffle(BUFFER_SIZE)
        ds = img_paths.map(train_convert, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
            batch_size, drop_remainder=True, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
            tf.data.experimental.AUTOTUNE)
        print(f'Train dataset size: {BUFFER_SIZE}')
        print(f'Train batches: {tf.data.experimental.cardinality(ds)}')
        return ds

    base_dataset = create_dataaset(DATA_DIR, BATCH_SIZE) # this as 752 images
    compare_dataset = create_dataaset(COMARE_DIR, BATCH_SIZE) # this as 50 images


    mean = []

    for real_batch in tqdm(base_dataset):
        for compare_batch in compare_dataset:

            loss = lpips([real_batch, compare_batch])
            mean.append(tf.reduce_mean(loss))

    print("LPIPS loss: ", tf.reduce_mean(mean))
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir')
    parser.add_argument('--generated_dir')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=512)
    args = parser.parse_args()
    benchmark(args)

    
