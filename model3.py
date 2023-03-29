import keras as keras
import keras.layers as L
import tensorflow as tf
import tensorflow_addons as tfa




# This is the implementation of the FastGAN model with more decoders.  
# This model is larger than the original FastGAN model and is used for the 256x256 images
# Comments labeled "ChatGPT" are provided by ChatGPT, checked by us and added to the code
# Comments labeled "Group 18" are provided by us. 


# ChatGPT - This function is useful in machine learning and deep learning applications, 
# where it is often desirable to normalize the inputs to a neural network to
# improve training performance and avoid numerical instability. 
# By normalizing the second moment of the input tensor, 
# the function helps to ensure that the magnitudes of the inputs are approximately the same, 
# which can help to improve the stability and convergence of the training process


def normalize_2nd_moment(x, axis=1, eps=1e-8):
    return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=axis, keepdims=True) + eps)


# ChatGPT - Spectral normalization works by dividing the weight matrix of a layer by its spectral norm, 
# so that the singular values of the matrix are all scaled down to be less than or equal to 1. 
# This scaling operation does not change the overall behavior of the layer, 
# it can make the training process more stable and prevent the layer from
#  becoming too sensitive to small changes in the input.
# Group 18 - we also intialize the weights with the orthogonal initializer for better gradient flow. 
# We also don't use any bias in the convolutional layers. 
def conv2D(*args, **kwargs):
    return tfa.layers.SpectralNormalization(L.Conv2D(*args, **kwargs,kernel_initializer='orthogonal',use_bias=False ))



# GROUP 18 - Simple Gated Linear Unit (GLU) layer.
# ChatGPT - The purpose of this operation is to apply a gating mechanism to the input tensor, 
# where the sigmoid function acts as a gate that controls the flow of information from the second part to the first part.
#  Elements in the second part that have a high sigmoid value (close to 1) are allowed to pass through, while elements with a low sigmoid value (close to 0) are suppressed. 
# This can be useful in various types of neural networks, such as convolutional neural networks and recurrent neural networks,  to control the flow of information and prevent overfitting.
class GLU(L.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        nc = inputs.shape[-1]
        assert nc % 2 == 0, "The channel dimension of the input must be divisible by 2"
        self.nc = nc // 2
        return inputs[:, :, :, :self.nc] * tf.nn.sigmoid(inputs[:, :, :, self.nc:])



    def get_config(self):
        config = super(GLU, self).get_config()
        return config



# Simple Noise Injection layer taken directly from Emilio Morales' implementation
class NoiseInjection(L.Layer):
    def __init__(self, **kwargs):
        super(NoiseInjection, self).__init__(**kwargs)
        self.weight = self.add_weight(
            name='weight', shape=(1), initializer='zeros', 
            dtype=self.dtype, trainable=True) 

    def call(self, feat):
        batch, height, width, _ = feat.shape
        noise = tf.random.normal((batch, height, width, 1), dtype=self.dtype)
        return feat + self.weight * noise

    def get_config(self):
        config = super(NoiseInjection, self).get_config()
        return config


# Group 18 - This is a simple trivial convlution layer
# used in the Generator Network and the Discriminator Network
# It takes as an input a 4d tensor of shape (batch_size, height, width, channels)
# and returns a 4d tensor of shape (batch_size, height*2, width*2, out_channels * 2)

def UpBlock(out_channels):
    return keras.Sequential([
        L.UpSampling2D(2),
        conv2D(out_channels * 2, 3, padding='same'),
        L.BatchNormalization(),
        GLU(),
    ])


# Group 18 - This is a layer similiar to the UpBlock layer
# but it also adds noise to the input tensor. 
def UpBlockComp(out_channels):
    return keras.Sequential([
        L.UpSampling2D(2),
        conv2D(out_channels * 2, 3, padding='same'),
        NoiseInjection(),
        L.BatchNormalization(),
        GLU(),
        conv2D(out_channels * 2, 3, padding='same'),
        NoiseInjection(),
        L.BatchNormalization(),
        GLU(),
    ])



## ChatGPT)  Batch normalization is especially useful because the generator and discriminator networks are trained in parallel, 
# and the distributions of the input data to the networks can change rapidly during training. 
# This can cause the gradients to become unstable, which in turn can lead to the generator or discriminator getting stuck in a local minima. By using batch normalization, 
# the input data to each layer is normalized, making it easier for the gradients to flow through the network and reducing the likelihood of getting stuck in a local minima.
# Another benefit of batch normalization is that it can help to regularize the network,
#  by adding noise to the input data. This can help to prevent overfitting and improve the generalization performance of the network.



# Group 18 - This is a simple skip execution layer 

# ChatGPT) Skip connections, also known as residual connections, 
# are a technique used in neural networks to improve the flow of gradients during training, 
# and improve the performance of deep models. A skip connection allows the output of a 
# layer to be passed directly to a layer further downstream in the network, rather than being transformed by a series of intermediate layers. This can help to preserve important information from the input, and allow the network to learn both shallow and deep features at the same time.
def SEBlock(out_planes):
    return keras.Sequential([
        tfa.layers.AdaptiveAveragePooling2D(4),
        conv2D(out_planes, kernel_size = 4, activation='swish'),
        conv2D(out_planes, kernel_size = 1, activation='sigmoid'),
    ])


# Init layer directly copied from Emilio Morales' implementation

class InitLayer(L.Layer):
    def __init__(self, latent_dim=256, initializer='orthogonal'):
        super(InitLayer, self).__init__()
        self.conv = tf.keras.Sequential([
            keras.layers.Conv2DTranspose(latent_dim * 2, 
                kernel_size=4, 
                use_bias=False, kernel_initializer=initializer),
                L.BatchNormalization(),
                GLU()
        ])
        
    def call(self, x):
        x = tf.expand_dims(x, 1)
        x = tf.expand_dims(x, 1)
        return self.conv(x)



class Generator(keras.models.Model):
    def __init__(self, out_planes=1024):
        super(Generator, self).__init__()
        self.init = InitLayer()
        self.Conv_8 = UpBlockComp(out_planes)
        self.Conv_16 = UpBlock(out_planes // 2)
        self.Conv_32 = UpBlockComp(out_planes // 4)
        self.Conv_64 = UpBlock(out_planes // 8)
        self.Conv_128 = UpBlockComp(out_planes // 16)
        self.Conv_256 = UpBlock(out_planes // 32)
        self.Conv_512 = UpBlock(out_planes // 64)
        self.Skip_64 = SEBlock(out_planes // 8)
        self.Skip_128 = SEBlock(out_planes // 16)
        self.Skip_256 = SEBlock(out_planes // 32)
        self.Skip_512 = SEBlock(out_planes // 64)
        self.chanels_out = conv2D(3, 3, padding='same', activation='tanh')



    def call(self, x):
        x = normalize_2nd_moment(x)
        w_h_4 = self.init(x)
        w_h_8 = self.Conv_8(w_h_4)
        w_h_16 = self.Conv_16(w_h_8)
        w_h_32 = self.Conv_32(w_h_16)
        w_h_64 = self.Conv_64(w_h_32) * self.Skip_64(w_h_4)
        w_h_128 = self.Conv_128(w_h_64) * self.Skip_128(w_h_8)
        w_h_256 = self.Conv_256(w_h_128) * self.Skip_256(w_h_16)
        w_h_512 = self.Conv_512(w_h_256) * self.Skip_512(w_h_32)

        return self.chanels_out(w_h_512)
     

    


# Group 18 - Here lies the Discriminator Network and the most of our improvements

class DownBlockComp(L.Layer):
    def __init__(self, filters):
        super(DownBlockComp, self).__init__()

        self.main = tf.keras.Sequential([
            conv2D(filters, kernel_size=4, padding='same',
                strides=2),
            L.BatchNormalization(),
            L.LeakyReLU(0.2),
            conv2D(filters, kernel_size=3, padding='same'),
            L.BatchNormalization(),
            L.LeakyReLU(0.2),
        ])
        
        self.direct = tf.keras.Sequential([
            L.AveragePooling2D((2, 2)),
            conv2D(filters, kernel_size=1, padding='same'),
            L.BatchNormalization(),
            L.LeakyReLU(0.2),
        ])

    def call(self, x):
        return (self.main(x) + self.direct(x)) / 2



# this decoder is used to decode the so called scenery tensor 
def decoder(filters =128, small=False):

    return tf.keras.Sequential([
        tfa.layers.AdaptiveAveragePooling2D(8),
        UpBlock(filters),   # image size 16
        UpBlock(filters//2), # image size 32
        UpBlock(filters//4), # image size 64
        UpBlock(filters//8), # image size 128
        conv2D(3, kernel_size=3, padding='same', activation='tanh'),
    ])





class Discriminator(keras.models.Model):

    def __init__(self, filters=128, dec_dim=128, img_size=512):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        if self.img_size == 256:
            self.econde_to_latent_space = keras.Sequential([
                conv2D(filters//32, 3, padding='same'),
                L.LeakyReLU(0.2),
            ])

        elif self.img_size == 512:
            self.econde_to_latent_space = keras.Sequential([
                conv2D(filters//32, 4,2, padding='same'),
                L.LeakyReLU(0.2),
            ])


        self.w_h_128 = DownBlockComp(filters//16)
        self.w_h_64 = DownBlockComp(filters//8)
        self.w_h_32 = DownBlockComp(filters//4)
        self.w_h_16 = DownBlockComp(filters//2)
        self.w_h_8 = DownBlockComp(filters)

        self.big_logits = tf.keras.Sequential([
            conv2D(filters, kernel_size=4, padding='valid'),
            L.LeakyReLU(0.2),
            conv2D(1, kernel_size=1, padding='valid'),
            L.Flatten(),
        ])

        self.decoder = decoder(filters)
        self.part_decoder = decoder(filters//2)
        self.center_decoder = decoder(filters//4)
 

        
        # skip connections for better gradient flow in Discriminator
        self.skip_256_32 = SEBlock(filters//4)
        self.skip_128_16 = SEBlock(filters//2)
        self.skip_64_8 = SEBlock(filters)



    def call(self, x, label='fake', part=None):
        w_h_256 = self.econde_to_latent_space(x)
        w_h_128 = self.w_h_128(w_h_256)
        w_h_64 = self.w_h_64(w_h_128)
             
        w_h_32 = self.w_h_32(w_h_64) * self.skip_256_32(w_h_256)

        w_h_16 = self.w_h_16(w_h_32) * self.skip_128_16(w_h_128)
        w_h_8 = self.w_h_8(w_h_16)   * self.skip_64_8(w_h_64)




        if label == 'fake':
            return self.big_logits(w_h_8)

        elif label == 'real':
            assert part is not None
            

            # now also encoding the part of the image
            if part == 0:
                cropped = crop(w_h_16, 0)
            elif part == 1:
                cropped = crop(w_h_16, 1)
            elif part == 2:
                cropped = crop(w_h_16, 2)
            elif part == 3:
                cropped = crop(w_h_16, 3)
            
            center_cropped = center_crop(w_h_32)

            return [self.big_logits(w_h_8), self.decoder(w_h_8), self.part_decoder(cropped), self.center_decoder(center_cropped)]







def center_crop(x):
    return x[:, 8:16, 8:16, :]




def crop(x, part):
    if part == 0:
        return x[:, :8, :8, :]
    elif part == 1:
        return x[:, :8, 8:, :]
    elif part == 2:
        return x[:, 8:, :8, :]
    elif part == 3:
        return x[:, 8:, 8:, :]
    



