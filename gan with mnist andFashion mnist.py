
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model

# =====================================================
# 1. CONFIGURATION
# =====================================================

DATA_PATH = r"C:\Users\Tintu\OneDrive\Desktop\NPOL\combined_mnist_fashion"

OUTPUT_PATH = os.path.join(DATA_PATH, "GAN_Results")
os.makedirs(OUTPUT_PATH, exist_ok=True)

N = 784
M = 100

TRAIN_SAMPLES = 50000
TEST_SAMPLES = 10000

EPOCHS = 100
BATCH_SIZE = 64

LEARNING_RATE = 0.0002
ADV_WEIGHT = 0.001

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow version:", tf.__version__)

# =====================================================
# 2. LOAD COMBINED DATASET
# =====================================================

print("\nLoading combined dataset...")

x_train = np.load(
    os.path.join(DATA_PATH, "combined_train.npy")
)

x_test = np.load(
    os.path.join(DATA_PATH, "combined_test.npy")
)

print("Original training shape:", x_train.shape)
print("Original testing shape:", x_test.shape)

# =====================================================
# 3. SELECT TRAINING AND TESTING SAMPLES
# =====================================================

train_indices = np.random.permutation(len(x_train))
test_indices = np.random.permutation(len(x_test))

x_train = x_train[train_indices[:TRAIN_SAMPLES]]
x_test = x_test[test_indices[:TEST_SAMPLES]]

# Ensure correct data type and range

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train = np.clip(x_train, 0.0, 1.0)
x_test = np.clip(x_test, 0.0, 1.0)

print("\nSelected training images:", x_train.shape)
print("Selected testing images:", x_test.shape)

# =====================================================
# 4. GENERATE MEASUREMENT MATRIX
# =====================================================

print("\nGenerating measurement matrix...")

A = np.random.randn(M, N).astype("float32")

A = A / np.sqrt(M)

A_tensor = tf.constant(A, dtype=tf.float32)

# =====================================================
# 5. GENERATE COMPRESSED MEASUREMENTS
# =====================================================

print("\nGenerating compressed measurements...")

y_train = np.matmul(x_train, A.T)
y_test = np.matmul(x_test, A.T)

print("Original image dimension:", N)
print("Compressed measurement dimension:", M)

print("Training measurements:", y_train.shape)
print("Testing measurements:", y_test.shape)

# =====================================================
# 6. BUILD GENERATOR
# =====================================================

def build_generator():

    inputs = layers.Input(shape=(M,))

    x = layers.Dense(256, activation="relu")(inputs)

    x = layers.Dense(512, activation="relu")(x)

    x = layers.Dense(784, activation="sigmoid")(x)

    generator = Model(
        inputs,
        x,
        name="Generator"
    )

    return generator


# =====================================================
# 7. BUILD DISCRIMINATOR
# =====================================================

def build_discriminator():

    inputs = layers.Input(shape=(N,))

    x = layers.Dense(512)(inputs)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    discriminator = Model(
        inputs,
        outputs,
        name="Discriminator"
    )

    return discriminator


# =====================================================
# 8. CREATE MODELS
# =====================================================

generator = build_generator()

discriminator = build_discriminator()

generator.summary()
discriminator.summary()

# =====================================================
# 9. OPTIMIZERS AND LOSS FUNCTIONS
# =====================================================

g_optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    beta_1=0.5
)

d_optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    beta_1=0.5
)

bce = tf.keras.losses.BinaryCrossentropy()

# =====================================================
# 10. GAN TRAINING STEP
# =====================================================

@tf.function
def train_step(real_images, compressed_images):

    # ---------------------------------------------
    # DISCRIMINATOR TRAINING
    # ---------------------------------------------

    with tf.GradientTape() as d_tape:

        generated_images = generator(
            compressed_images,
            training=True
        )

        generated_images = tf.stop_gradient(
            generated_images
        )

        real_output = discriminator(
            real_images,
            training=True
        )

        fake_output = discriminator(
            generated_images,
            training=True
        )

        real_loss = bce(
            tf.ones_like(real_output),
            real_output
        )

        fake_loss = bce(
            tf.zeros_like(fake_output),
            fake_output
        )

        d_loss = (real_loss + fake_loss) / 2.0

    d_gradients = d_tape.gradient(
        d_loss,
        discriminator.trainable_variables
    )

    d_optimizer.apply_gradients(
        zip(
            d_gradients,
            discriminator.trainable_variables
        )
    )

    # ---------------------------------------------
    # GENERATOR TRAINING
    # ---------------------------------------------

    with tf.GradientTape() as g_tape:

        reconstructed_images = generator(
            compressed_images,
            training=True
        )

        fake_output = discriminator(
            reconstructed_images,
            training=False
        )

        reconstruction_loss = tf.reduce_mean(
            tf.square(
                real_images - reconstructed_images
            )
        )

        adversarial_loss = bce(
            tf.ones_like(fake_output),
            fake_output
        )

        g_loss = (
            reconstruction_loss
            + ADV_WEIGHT * adversarial_loss
        )

    g_gradients = g_tape.gradient(
        g_loss,
        generator.trainable_variables
    )

    g_optimizer.apply_gradients(
        zip(
            g_gradients,
            generator.trainable_variables
        )
    )

    return (
        g_loss,
        d_loss,
        reconstruction_loss,
        adversarial_loss
    )


# =====================================================
# 11. PREPARE TRAINING DATA
# =====================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)
)

train_dataset = train_dataset.shuffle(
    TRAIN_SAMPLES,
    seed=SEED
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

train_dataset = train_dataset.prefetch(1)

# =====================================================
# 12. TRAIN GAN
# =====================================================

print("\n========== GAN TRAINING STARTED ==========")

for epoch in range(EPOCHS):

    g_losses = []
    d_losses = []
    r_losses = []
    adv_losses = []

    for real_images, compressed_images in train_dataset:

        g_loss, d_loss, r_loss, adv_loss = train_step(
            real_images,
            compressed_images
        )

        g_losses.append(float(g_loss.numpy()))
        d_losses.append(float(d_loss.numpy()))
        r_losses.append(float(r_loss.numpy()))
        adv_losses.append(float(adv_loss.numpy()))

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"G Loss: {np.mean(g_losses):.6f} | "
        f"D Loss: {np.mean(d_losses):.6f} | "
        f"Reconstruction Loss: {np.mean(r_losses):.6f} | "
        f"Adversarial Loss: {np.mean(adv_losses):.6f}"
    )

print("\nGAN training completed!")

# =====================================================
# 13. RECONSTRUCT TEST IMAGES
# =====================================================

print("\nGenerating reconstructed images...")

reconstructed = generator.predict(
    y_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

reconstructed = np.clip(
    reconstructed,
    0.0,
    1.0
)

# =====================================================
# 14. CALCULATE E, S, L2, MSE AND NMSE
# =====================================================

difference = x_test - reconstructed

# Per-image calculations

E_values = np.sum(
    difference ** 2,
    axis=1
)

S_values = np.sum(
    x_test ** 2,
    axis=1
)

L2_values = np.sqrt(E_values)

MSE_values = np.mean(
    difference ** 2,
    axis=1
)

NMSE_values = E_values / (S_values + 1e-12)

NMSE_dB_values = 10 * np.log10(
    NMSE_values + 1e-12
)

print("\n========== TEST RESULTS ==========")

print("Average E:", np.mean(E_values))
print("Average S:", np.mean(S_values))
print("Average L2:", np.mean(L2_values))
print("Average MSE:", np.mean(MSE_values))
print("Average NMSE:", np.mean(NMSE_values))
print("Average NMSE (dB):", np.mean(NMSE_dB_values))

# =====================================================
# 15. DISPLAY ORIGINAL AND RECONSTRUCTED IMAGES
# =====================================================

number_of_images = 5

plt.figure(figsize=(15, 5))

for i in range(number_of_images):

    plt.subplot(2, number_of_images, i + 1)

    plt.imshow(
        x_test[i].reshape(28, 28),
        cmap="gray",
        vmin=0,
        vmax=1
    )

    plt.title("Original")
    plt.axis("off")

    plt.subplot(
        2,
        number_of_images,
        number_of_images + i + 1
    )

    plt.imshow(
        reconstructed[i].reshape(28, 28),
        cmap="gray",
        vmin=0,
        vmax=1
    )

    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_PATH, "GAN_Reconstruction.png"),
    dpi=200
)

plt.show()

# =====================================================
# 16. SAVE MODEL
# =====================================================

generator.save(
    os.path.join(OUTPUT_PATH, "GAN_Generator.keras")
)

discriminator.save(
    os.path.join(OUTPUT_PATH, "GAN_Discriminator.keras")
)

np.save(
    os.path.join(OUTPUT_PATH, "reconstructed_images.npy"),
    reconstructed
)

np.save(
    os.path.join(OUTPUT_PATH, "E_values.npy"),
    E_values
)

np.save(
    os.path.join(OUTPUT_PATH, "S_values.npy"),
    S_values
)

np.save(
    os.path.join(OUTPUT_PATH, "NMSE_values.npy"),
    NMSE_values
)

print("\n========== ALL RESULTS SAVED ==========")

print("Output folder:", OUTPUT_PATH)
