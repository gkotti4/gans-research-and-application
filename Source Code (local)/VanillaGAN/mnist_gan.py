# ================================================
#         SIMPLE VANILLA GAN IMPLEMENTATION
#         Dataset: MNIST (Handwritten Digits)
#         Image Size: 28 x 28 (Grayscale)
#         Framework: NumPy (No Deep Learning Libs)
# ================================================

import argparse
import os

import time

import numpy as np
import matplotlib.pyplot as plt


#                        === DEFAULT PARAMETERS ===
MNIST_DATASET_ROOT = "MNIST_CSV"
MNIST_DATASET_FILENAME = "mnist_train.csv"

MODEL_DATA_ROOT = "Model Data/"
MODEL_DATA_FILENAME = "mnist_gan_weights.npz"

DEFAULT_TRAINING_SLICE_SIZE = 1280
DEFAULT_NUM_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE_G = 0.0001
DEFAULT_LEARNING_RATE_D = 0.0001

DEFAULT_LATENT_DIM = 100 


# Set np error flags       === DEBUGGIN ===
np.seterr(all='raise')



#                          === LOAD ARGUMENTS ===
parser = argparse.ArgumentParser(description="Train GAN on MNIST dataset")
parser.add_argument('--train_size', type=int, default=DEFAULT_TRAINING_SLICE_SIZE)
parser.add_argument('--num_epochs', type=int, default=DEFAULT_NUM_EPOCHS)
parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
parser.add_argument('--latent_dim', type=int, default=DEFAULT_LATENT_DIM)
parser.add_argument('--learning_rate_g', type=float, default=DEFAULT_LEARNING_RATE_G)
parser.add_argument('--learning_rate_d', type=float, default=DEFAULT_LEARNING_RATE_D)
parser.add_argument('--show_losses', action='store_true')
parser.add_argument('--show_g_images', action='store_true')
args = parser.parse_args()



#                          === LOAD DATASET ===

# Get the directory this script is in
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build a path to the dataset relative to the script
data_path = os.path.join(script_dir, MNIST_DATASET_ROOT, MNIST_DATASET_FILENAME)

# Load the dataset
data = np.loadtxt(data_path, delimiter=",", skiprows=1)


# Shuffle dataset images
#np.random.shuffle(data)

# Separate inputs and labels and set parameters
dataset_size = len(data)
train_size = min(args.train_size, dataset_size)

labels = data[:, 0]                     # shape: (60000,) or (train_size,)
images = data[:, 1:] / 255.0            # shape: (60000, 784), normalize to [0, 1]
images = images.reshape(-1, 28, 28)     # reshape to (60000, 28, 28)
images = images[:train_size]            # resize (if wanting to use smaller section)

print("Number of MNIST dataset images: ", dataset_size)
#print("Number of training images: ", train_size)


# Clear dataset to free up memory
del data



#                       === SET TRAINING PARAMETERS ===
num_epochs = args.num_epochs
batch_size = args.batch_size
latent_dim_g = args.latent_dim
learning_rate_g = args.learning_rate_g
learning_rate_d = args.learning_rate_d



#                           === DISPLAY RESULTS ===

# === Display images ===
def show_images(images, n=4):
    fig, axes = plt.subplots(n, n, figsize=(5, 5))
    for i in range(n):
        for j in range(n):
            axes[i, j].imshow(images[i*n + j], cmap='gray')
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()

def show_image(img):
    #img = generated_img.reshape(28, 28)
    #if(img.shape != np.shape(28,28)):
        #img.reshape(28,28)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.close()
# Show sample selection of images
#show_images(images, 10) # show small subsection of training images


def show_generated_images(generator, n_rows=4, n_cols=4, latent_dim=latent_dim_g):
    total = n_rows * n_cols
    z = np.random.randn(total, latent_dim)
    generated = generator.forward(z)  # shape: (total, 784)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            image = generated[idx].reshape(28, 28)
            axes[i][j].imshow(image, cmap='gray')
            axes[i][j].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()
# Show trained generator images
#show_generated_images(G)


# === Plot Data ===
def plot_gan_losses(G_losses=[], D_losses=[], D_real_scores=[], D_fake_scores=[], title="GAN Training Progress"):

    def smooth(data, factor=0.9):
        # smoothed = previous smoothed * factor + current raw * (1-factor)
        smoothed=[]
        last = data[0]
        for point in data:
            smoothed_val = last * factor + (1-factor) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    plt.figure(figsize=(10,6))
    plt.title(title)

    # Plot generator and discriminator loss
    if G_losses is not None:
        plt.plot(G_losses, label="Generator Loss", color='blue')
    
    if D_losses is not None:
        #plt.plot(D_losses, label="Discriminator Score on Fake", color='red')
        pass

    # Plot D's confidence on real/fake
    if D_real_scores is not None:
        plt.plot(D_real_scores, label="D(Real) Confidence", linestyle='--', color='green')

    if D_fake_scores is not None:
        plt.plot(D_fake_scores, label="D(Fake) Confidence", linestyle='--', color='orange')

    plt.xlabel("Training Steps (Batches)")
    plt.ylabel("Loss / Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def output_parameters():
    print("\n----------------------GAN Parameters--------------------------")
    print("filename= ", MODEL_DATA_FILENAME)
    print("train_size= ", train_size)
    print("num_epochs= ", num_epochs)
    print("batch_size= ", batch_size)
    print("learning_rate_g", args.learning_rate_g)
    print("learning_rate_d", args.learning_rate_d)
    print("--------------------------------------------------------------", end="\n\n")










#                                      === ACTIVATION FUNCTIONS ===

# ReLU - Rectified Linear Unit - if | x > 0: output x | if x <= 0: output 0 | - max(0,x)
def relu(x):
    return np.maximum(0,x)

# Hyperbolic Tangent 
def tanh(x):
    return np.tanh(x)



#                               === LOSS FUNCTIONs ===
# Binary Cross-Entropy (BCE)

def bce_loss(y_true, y_pred):
    epsilon = 1e-8
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

def bce_grad(y_true, y_pred):
    epsilon = 1e-8
    return (y_pred - y_true) / ((y_pred + epsilon) * (1 - y_pred + epsilon)) / y_true.shape[0]


#                                       === GENERATOR ===

# Takes in random noise (z) → outputs a 28×28 "fake image"
# It's just a function:
# z → (Linear → ReLU → Linear → Tanh) → x_fake (forward)

class Generator:
    #           - Self Variables (Generator) -
    # W1     - weights for noise → hidden layer (shape: 100 x 128)
    # dW1    - gradient of W1
    # W2     - weights for hidden → image layer (shape: 128 x 784)
    # dW2    - gradient of W2

    # b1     - bias for hidden layer (shape: 1 x 128)
    # db1    - gradient of b1
    # b2     - bias for output layer (shape: 1 x 784)
    # db2    - gradient of b2

    # z      - latent vector (random noise) input (shape: 1 x 100)
    # h_raw  - hidden layer input before ReLU
    # h      - hidden layer output after ReLU

    # out_raw - output before tanh (shape: 1 x 784)
    # out     - generated image (after tanh), values in [-1, 1]

    # lr     - learning rate used in generator updates

    # G_losses - average bce losses - G trying to Minimize

    def __init__(self, input_dim=100, hidden_dim=128, output_dim=784, lr=learning_rate_g):
        self.lr = lr

        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1. / input_dim)
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1. / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

        self.G_losses = []

    def forward(self, z):
        self.z = z

        self.h_raw = np.dot(z, self.W1) + self.b1
        self.h = relu(self.h_raw)

        self.out_raw = np.dot(self.h, self.W2) + self.b2
        self.out = tanh(self.out_raw)

        return self.out  # shape: (batch_size, 784)

    def backward(self, d_out):  # d_out = gradient from discriminator
        d_out_raw = d_out * (1 - self.out ** 2)         # tanh derivative

        dW2 = np.dot(self.h.T, d_out_raw)
        db2 = np.sum(d_out_raw, axis=0, keepdims=True)

        d_h = np.dot(d_out_raw, self.W2.T)
        d_h_raw = d_h * (self.h_raw > 0).astype(float)  # relu derivative

        dW1 = np.dot(self.z.T, d_h_raw)
        db1 = np.sum(d_h_raw, axis=0, keepdims=True)

        self.dW1, self.db1 = dW1, db1
        self.dW2, self.db2 = dW2, db2

        # === Debugging ===
        d_out = np.clip(d_out, -5, 5) # prevent exploding 


    def update(self):
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2

        del self.dW1, self.dW2, self.db1, self.db2

    def train_step(self, d_out):
        self.backward(d_out)
        self.update()

        





#                                   === DISCRIMINATOR ===

def leaky_relu(x, alpha=0.2):
    #      max(αx,x)
    return np.where(x>0, x, alpha*x)

def leaky_relu_derivative(x, alpha=0.2):
    return np.where(x > 0, 1, alpha)

def sigmoid(x):
    return 1 / (1+np.exp(-x))

# Takes in an image (real or fake) → outputs probability it's real
# It's a function:
# x → (Linear → LeakyReLU → Linear → Sigmoid) → p_real (forward)

class Discriminator:
    #         - Self Variables (Discriminator) -
    # W1     - weights for input → hidden layer (shape: 784 x 128)
    # dW1    - gradient of W1
    # W2     - weights for hidden → output (real/fake) (shape: 128 x 1)
    # dW2    - gradient of W2

    # b1     - bias for hidden layer (shape: 1 x 128)
    # db1    - gradient of b1
    # b2     - bias for output layer (shape: 1 x 1)
    # db2    - gradient of b2

    # x      - input image (flattened, shape: batch x 784)
    # h_raw  - hidden layer input before activation
    # h      - hidden layer output after leaky ReLU

    # out_raw - output before sigmoid (shape: batch x 1)
    # out     - prediction (after sigmoid), values in [0, 1]

    # lr     - learning rate used in discriminator updates

    # D_losses      - tracked output confidence values on fake images
    # real_scores   - average D confidence on real images (for plotting/debug)
    # fake_scores   - average D confidence on fake images (for plotting/debug)

    def __init__(self, input_dim=784, hidden_dim=128, lr=learning_rate_d):
        # Xavier Initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1. / input_dim)
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(1. / hidden_dim)
        self.b2 = np.zeros((1, 1))

        self.lr = lr

        # Logging metrics
        self.D_losses = []
        self.real_scores = []
        self.fake_scores = [] # same as losses

    def forward(self, x, label=None):
        """
        Forward pass through the Discriminator.
        Input:
            x      - flattened input image (1 x 784 or batch x 784)
            label  - "real" or "fake" (optional, for tracking)
        Returns:
            out    - prediction (probability of being real)
        """
        self.x = x

        # Layer 1: Input → Hidden
        self.h_raw = np.dot(x, self.W1) + self.b1
        self.h = leaky_relu(self.h_raw)

        # Layer 2: Hidden → Output
        self.out_raw = np.dot(self.h, self.W2) + self.b2
        self.out = sigmoid(self.out_raw)

        # Track confidence if label is given
        if label == "real":
            self.real_scores.append(np.mean(self.out))
        elif label == "fake":
            self.fake_scores.append(np.mean(self.out))

        return self.out

    def backward(self, y_true):
        # Backward pass through the Discriminator using BCE loss

        d_out = bce_grad(y_true, self.out)  # ∂Loss/∂out

        # Output layer gradients
        d_out_raw = d_out * self.out * (1 - self.out)
        dW2 = np.dot(self.h.T, d_out_raw)
        db2 = np.sum(d_out_raw, axis=0, keepdims=True)

        # Hidden layer gradients
        d_h = np.dot(d_out_raw, self.W2.T)
        d_h_raw = d_h * leaky_relu_derivative(self.h_raw)
        dW1 = np.dot(self.x.T, d_h_raw)
        db1 = np.sum(d_h_raw, axis=0, keepdims=True)

        self.dW1, self.db1 = dW1, db1
        self.dW2, self.db2 = dW2, db2

    def update(self):
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2

        del self.dW1, self.dW2, self.db1, self.db2

    def train_step(self, y_true):
        self.backward(y_true)
        self.update()





#                       === SAVE | LOAD MODEL DATA ===
def save_model(generator, discriminator, filename=MODEL_DATA_FILENAME, root=MODEL_DATA_ROOT):
    
    save_path = root + filename
    np.savez(
        save_path,
        G_W1=generator.W1, G_b1=generator.b1,
        G_W2=generator.W2, G_b2=generator.b2,
        D_W1=discriminator.W1, D_b1=discriminator.b1,
        D_W2=discriminator.W2, D_b2=discriminator.b2
    )
    print("saved model data", end="\n\n")

def load_model(generator, discriminator, filename=MODEL_DATA_FILENAME, root=MODEL_DATA_ROOT):
    
    load_path = root + filename
    if not os.path.exists(load_path):
        print(f"Model file '{load_path}' not found. Creating new save file...")
        save_model(generator, discriminator, load_path)
        return

    data = np.load(load_path)
    if generator:
        generator.W1 = data["G_W1"]
        generator.b1 = data["G_b1"]
        generator.W2 = data["G_W2"]
        generator.b2 = data["G_b2"]

    if discriminator:
        discriminator.W1 = data["D_W1"]
        discriminator.b1 = data["D_b1"]
        discriminator.W2 = data["D_W2"]
        discriminator.b2 = data["D_b2"]

    print("loaded model data for ", generator, discriminator, end="\n\n")











#                           === INITIALIZE MODEL VALUES ===

G = Generator(latent_dim_g)
D = Discriminator()


#                           === TRAIN MODEL ===
def train():
    # Load model data
    load_model(G, D)

    print("============= GAN TRAINING BEGIN =============")
    output_parameters()

    # Performance Metrics
    epoch_times = []

    for epoch in range(num_epochs):
        print(f"Training epoch {epoch}...")
        start_time_epoch = time.time()
        for i in range(0, train_size, batch_size):
            # Skip last incomplete batch (if size < batch_size)
            if images[i:i+batch_size].shape[0] < batch_size:
                print("Batch skipped at i =", i)
                continue

            # -----------------------------------------------
            # 1. REAL DATA: Get a batch of real MNIST images
            # -----------------------------------------------
            real_images = images[i:i+batch_size].reshape(batch_size, 784)
            real_labels = np.ones((batch_size, 1))  # Label = 1 for real

            # -----------------------------------------------
            # 2. FAKE DATA: Generate fake images from random noise
            # -----------------------------------------------
            z_noise = np.random.randn(batch_size, latent_dim_g)
            fake_images = G.forward(z_noise)
            fake_labels = np.zeros((batch_size, 1))  # Label = 0 for fake

            # -----------------------------------------------
            # 3. TRAIN DISCRIMINATOR
            # -----------------------------------------------
            # Forward pass on real
            D_real_out = D.forward(real_images)
            D.train_step(real_labels)

            # Forward pass on fake
            D_fake_out = D.forward(fake_images)
            D.train_step(fake_labels)

            # Track D outputs (confidence scores)
            D_confidence_real = np.mean(D_real_out)
            D_confidence_fake = np.mean(D_fake_out)

            D.real_scores.append(D_confidence_real)
            D.fake_scores.append(D_confidence_fake)

            # Optional: summarize D’s belief on fake as a proxy for loss
            D.D_losses.append(D_confidence_fake)

            # -----------------------------------------------
            # 4. TRAIN GENERATOR
            # -----------------------------------------------
            z_noise = np.random.randn(batch_size, latent_dim_g)
            fake_images = G.forward(z_noise)

            D_output_on_fake = D.forward(fake_images)

            # Calculate G loss properly (not just gradients)
            g_loss = bce_loss(np.ones((batch_size, 1)), D_output_on_fake)
            G.G_losses.append(g_loss)

            # Backprop through D into G
            d_loss_D_output = bce_grad(np.ones((batch_size, 1)), D_output_on_fake) * D.out * (1 - D.out)
            d_hidden_D = np.dot(d_loss_D_output, D.W2.T)
            d_input_D = d_hidden_D * (D.h_raw > 0).astype(np.float64)
            dG_input = np.dot(d_input_D, D.W1.T)

            G.train_step(dG_input)

        epoch_duration = time.time() - start_time_epoch
        epoch_times.append(epoch_duration)
        print(f"Epoch {epoch} took {epoch_duration:.4f} seconds\n")
        # Epoch End

    print("GAN training finished")

    average_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f"Average epoch time: {average_epoch_time:.4f} seconds")
    print(f"Total training time elapsed: {sum(epoch_times):.4f}")

    # Plot losses + D scores
    if args.show_losses:
        plot_gan_losses(
            G.G_losses,
            D.D_losses,
            D.real_scores,
            D.fake_scores,
        )

    # Plot current generator images
    if(args.show_g_images):
        show_generated_images(G, 5,5)

    # Save model data
    save_model(G, D)






# 			    === MAIN PROGRAM EXECUTION ===

if __name__ == "__main__":
    train()





















# === SAMPLE TESTS ===

# Sample Generator Test
#gen = Generator()

# Sample random noise vector (batch of 1)
#z = np.random.randn(1, 100)

# Forward pass
#generated_img = gen.forward(z)

# Show generated image data
#print("Generated image shape:", generated_img.shape)  # (1, 784)
#show_image(generated_img.reshape(28,28))


# Sample Discriminator Test
#D = Discriminator()

# Use generated image from Generator (already shape [1, 784])
# OR: use a real image from MNIST CSV
#fake_img = gen.forward(np.random.randn(1, 100))  # still flattened
#real_img = images[0].reshape(1, 784)             # one real MNIST sample

#print("Discriminator score (fake):", D.forward(fake_img))
#print("Discriminator score (real):", D.forward(real_img))






























