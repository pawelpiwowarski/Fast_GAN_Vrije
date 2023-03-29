
from model1 import Discriminator, Generator
import tensorflow as tf
from lpips_tensorflow import learned_perceptual_metric_model
from diffaug import DiffAugment
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from tqdm import tqdm
LATENT_DIM = 256
EPOCHS = 1000
BATCH_SIZE = 8
IMG_SIZE = 512
DATA_DIR = "../data/van_gogh_paintings/oil/*.jpg"
NUMBER_OF_SAMPLES_TO_GENERATE = 8
NUMBER_OF_IMAGES_PER_COLUMN = 4
NUMBER_OF_IMAGES_PER_ROW = 4
GENERATOR_OUTPUT = "output/model_1_gen_output/"
DISCRIMINATOR_OUTPUT = "output/model_1_disc_output/"
CHECKPOINT_DIR = "output/model_1_training_ckeckpoints/"
LOG_DIR = "output/model_1_logs/"
POLICY = "color,translation"

vgg_ckpt_fn = os.path.join( 'vgg', 'exported') 
lin_ckpt_fn = os.path.join('lin', 'exported')
cropped_img_size = 128
lpips = learned_perceptual_metric_model(cropped_img_size, vgg_ckpt_fn, lin_ckpt_fn)

def deprocess(img):
    return img * 127.5 + 127.5

def save_generator_img(model, noise,epoch ,direct):
    generated_img = model(noise)
    predictions = np.clip(deprocess(generated_img), 0, 255).astype(np.uint8)

    for i in range(predictions.shape[0]):
        path = os.path.join(direct, f'{epoch:04d}_{i:04d}.png')
        plt.imsave(path, predictions[i, :, :, :])





  


   
def save_decoder_img(model, img, epoch,direct):
    [pred,part1]  = model(img,label='real', part=1)
    predictions = np.clip(deprocess(part1), 0, 255).astype(np.uint8) 

    fig = plt.figure(figsize=(8, 4))

    for i in range(predictions.shape[0]):
        fig.add_subplot(2, 4, i+1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    path = os.path.join(direct, f'{epoch:04d}.png')
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(path, format='png')
    plt.close()

def crop_image_by_part(image, part):


    hw = image.shape[2]//2

    if part == 0:
        return image[:, :hw, :hw, :]
    if part == 1:
        return image[:, :hw, hw:, :]
    if part == 2:
        return image[:, hw:, :hw, :]
    if part == 3:
        return image[:, hw:, hw:, :]
    
def train_convert(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.image.random_flip_left_right(img)
    img = (img - 127.5) / 127.5 
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



print("***********Loading dataset***********")
dataset = create_dataaset(DATA_DIR, BATCH_SIZE)
print("*Dataset loaded* ✅")
def d_real_loss(logits):
            return tf.reduce_mean(tf.nn.relu(1.0 - logits))

def d_fake_loss(logits):
            return tf.reduce_mean(tf.nn.relu(1.0 + logits))

def discriminator_loss(real_img, fake_img):
            real_loss = d_real_loss(real_img)
            fake_loss = d_fake_loss(fake_img)
            return fake_loss + real_loss

def generator_loss(fake_img):
            return -tf.reduce_mean(fake_img)



class FastGAN(tf.keras.Model):
    def __init__(self,generator, discriminator,  latent_dim):
        super(FastGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.rec_loss =  lpips
        self.latent_dim = latent_dim
        self.g_loss_avg = tf.keras.metrics.Mean()
        self.d_loss_avg = tf.keras.metrics.Mean()
        self.rec_avg = tf.keras.metrics.Mean()
        self.d_total_avg = tf.keras.metrics.Mean()

    def reconstruction_loss(self,decoded_big, real_img):
        real_img = deprocess(real_img)
        big = deprocess(decoded_big)
        scenery_real = tf.image.resize(real_img,[128,128])
        err = tf.reduce_sum(self.rec_loss([big, scenery_real]))
        return err
    
    def create_log(self,ckpt_interval,max_ckpt_to_keep):
        self.writer = tf.summary.create_file_writer(LOG_DIR)

        self.ckpt = tf.train.Checkpoint(g_optimizer=self.g_optimizer,
                                        d_optimizer=self.d_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator,
                                        epoch=tf.Variable(0))
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, CHECKPOINT_DIR, max_to_keep=max_ckpt_to_keep)
        self.ckpt_interval = ckpt_interval

        if self.ckpt_manager.latest_checkpoint:
            last_ckpt = self.ckpt_manager.latest_checkpoint
            self.ckpt.restore(last_ckpt)
            print(f'Checkpoint restored from {last_ckpt} at epoch {int(self.ckpt.epoch)}')
            self.ckpt.epoch.assign_add(1)

    def save_log(self):
        epoch = int(self.ckpt.epoch)
        print(f'Epoch: {epoch}')
        print(f'Generator loss: {self.g_loss_avg.result():.4f}')
        print(f'Discriminator loss: {self.d_loss_avg.result():.4f}')
        print(f'Reconstruction loss: {self.rec_avg.result():.4f}') 
        print(f'Discriminator total loss: {self.d_total_avg.result():.4f}\n') 
        with self.writer.as_default():
            tf.summary.scalar('g_loss', self.g_loss_avg.result(), step=epoch)
            tf.summary.scalar('d_loss', self.d_loss_avg.result(), step=epoch)
            tf.summary.scalar('reconstruction_loss', self.rec_avg.result(), step=epoch)
            tf.summary.scalar('d_total_loss', self.d_total_avg.result(), step=epoch)
            self.writer.flush()

        if epoch % self.ckpt_interval == 0:
                self.ckpt_manager.save(epoch)
                print('Checkpoint saved at epoch {}\n'.format(epoch)) 
                
        self.ckpt.epoch.assign_add(1)

    def compile(self, g_optimizer, d_optimizer,g_loss_fn, d_loss_fn):
        super(FastGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn
    

    @tf.function
    def train_step(self, real_img):
        batch_size = tf.shape(real_img)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as discriminator_tape:
            fake_images = self.generator(noise, training=True)
            real_augmented = DiffAugment(real_img, policy=POLICY)
            fake_augmented = DiffAugment(fake_images, policy=POLICY)
            [real_discriminator_logits,decoded] = self.discriminator(real_augmented, training=True,label='real' )
            fake_discriminator_logits = self.discriminator(fake_augmented, training=True,label='fake')

            d_loss = self.d_loss_fn(real_discriminator_logits, fake_discriminator_logits)

            reconstruction_loss = self.reconstruction_loss(decoded, real_img)

            d_total_loss = d_loss + reconstruction_loss
        
        d_gradients = discriminator_tape.gradient(d_total_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        # Save discriminator metrics
        self.d_loss_avg.update_state(d_loss)
        self.rec_avg.update_state(reconstruction_loss)
        self.d_total_avg.update_state(d_total_loss)

        noise = tf.random.normal([batch_size, self.latent_dim])
        # train generator
        with tf.GradientTape() as generator_tape:
            fake_images = self.generator(noise, training=True)
            fake_augmented = DiffAugment(fake_images, policy=POLICY)
            fake_discriminator_logits = self.discriminator(fake_augmented, training=True,label='fake')
            g_loss = self.g_loss_fn(fake_discriminator_logits)
        
        g_gradients = generator_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        # Save generator metrics
        self.g_loss_avg.update_state(g_loss)


print('\n#########################')
print('Self-Supervised GAN Train')
print('#########################\n')

generator = Generator()
discriminator = Discriminator()


fastgan = FastGAN(generator, discriminator, LATENT_DIM)

dataset = create_dataaset(DATA_DIR, BATCH_SIZE)


fastgan.compile(g_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.99),
                d_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.99),
                g_loss_fn=generator_loss,
                d_loss_fn=discriminator_loss)

fastgan.create_log(ckpt_interval=10,max_ckpt_to_keep=5)

noise_seed = tf.random.normal([NUMBER_OF_SAMPLES_TO_GENERATE, LATENT_DIM], seed=13)

test_batch = next(iter(dataset))
fastgan.ckpt.epoch.assign_add(1)
start_epoch = int(fastgan.ckpt.epoch)




for _ in range(start_epoch, EPOCHS - start_epoch):
    start = time.time()
    for image_batch in tqdm(dataset):
            fastgan.train_step(image_batch)

    print(f'\nTime for epoch is {time.time()-start} sec')

    save_generator_img(fastgan.generator, noise_seed, int(fastgan.ckpt.epoch),GENERATOR_OUTPUT)
    save_decoder_img(fastgan.discriminator, test_batch, int(fastgan.ckpt.epoch),DISCRIMINATOR_OUTPUT)

    fastgan.save_log()

