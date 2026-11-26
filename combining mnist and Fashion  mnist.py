
import numpy as np
import os
import struct
import gzip

# ==========================================
# 1. DATASET LOCATIONS
# ==========================================

BASE_PATH = r"C:\Users\Tintu\OneDrive\Desktop\NPOL\combine MNIST AND fastion MNIST"

MNIST_PATH = os.path.join(BASE_PATH, "MNIST")

FASHION_PATH = os.path.join(BASE_PATH, "fastion")

SAVE_PATH = r"C:\Users\Tintu\OneDrive\Desktop\NPOL\combined_mnist_fashion"

os.makedirs(SAVE_PATH, exist_ok=True)


# ==========================================
# 2. FIND FILES AUTOMATICALLY
# ==========================================

def find_file(folder, filename):

    if not os.path.isdir(folder):
        raise FileNotFoundError("Folder not found: " + folder)

    alternative = filename.replace("-idx", ".idx")

    possible_names = [
        filename,
        filename + ".gz",
        alternative,
        alternative + ".gz"
    ]

    possible_names = [name.lower() for name in possible_names]

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower() in possible_names:

                path = os.path.join(root, file)

                print("Found:", path)

                return path

    print("\nFiles available in:", folder)

    for root, dirs, files in os.walk(folder):
        for file in files:
            print(file)

    raise FileNotFoundError("Missing file: " + filename)


# ==========================================
# 3. OPEN NORMAL OR COMPRESSED FILE
# ==========================================

def open_file(path):

    if path.lower().endswith(".gz"):
        return gzip.open(path, "rb")

    return open(path, "rb")


# ==========================================
# 4. LOAD IMAGES
# ==========================================

def load_images(folder, filename):

    path = find_file(folder, filename)

    with open_file(path) as file:

        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))

        if magic != 2051:
            raise ValueError("Invalid image file: " + path)

        images = np.frombuffer(file.read(), dtype=np.uint8)

        if images.size != count * rows * cols:
            raise ValueError("Image count mismatch: " + path)

        images = images.reshape(count, rows, cols)

    return images


# ==========================================
# 5. LOAD LABELS
# ==========================================

def load_labels(folder, filename):

    path = find_file(folder, filename)

    with open_file(path) as file:

        magic, count = struct.unpack(">II", file.read(8))

        if magic != 2049:
            raise ValueError("Invalid label file: " + path)

        labels = np.frombuffer(file.read(), dtype=np.uint8)

        if len(labels) != count:
            raise ValueError("Label count mismatch: " + path)

    return labels


# ==========================================
# 6. LOAD MNIST
# ==========================================

print("\n========== LOADING MNIST ==========")

x_train_mnist = load_images(MNIST_PATH, "train-images-idx3-ubyte")
y_train_mnist = load_labels(MNIST_PATH, "train-labels-idx1-ubyte")

x_test_mnist = load_images(MNIST_PATH, "t10k-images-idx3-ubyte")
y_test_mnist = load_labels(MNIST_PATH, "t10k-labels-idx1-ubyte")

print("MNIST Training:", x_train_mnist.shape)
print("MNIST Testing:", x_test_mnist.shape)


# ==========================================
# 7. LOAD FASHION-MNIST
# ==========================================

print("\n========== LOADING FASHION-MNIST ==========")

x_train_fashion = load_images(FASHION_PATH, "train-images-idx3-ubyte")
y_train_fashion = load_labels(FASHION_PATH, "train-labels-idx1-ubyte")

x_test_fashion = load_images(FASHION_PATH, "t10k-images-idx3-ubyte")
y_test_fashion = load_labels(FASHION_PATH, "t10k-labels-idx1-ubyte")

print("Fashion Training:", x_train_fashion.shape)
print("Fashion Testing:", x_test_fashion.shape)


# ==========================================
# 8. NORMALIZE
# ==========================================

x_train_mnist = x_train_mnist.astype("float32") / 255.0
x_test_mnist = x_test_mnist.astype("float32") / 255.0

x_train_fashion = x_train_fashion.astype("float32") / 255.0
x_test_fashion = x_test_fashion.astype("float32") / 255.0


# ==========================================
# 9. COMBINE TRAINING DATA
# ==========================================

x_train = np.concatenate(
    (x_train_mnist, x_train_fashion), axis=0
)

y_train = np.concatenate(
    (y_train_mnist, y_train_fashion), axis=0
)


# ==========================================
# 10. COMBINE TESTING DATA
# ==========================================

x_test = np.concatenate(
    (x_test_mnist, x_test_fashion), axis=0
)

y_test = np.concatenate(
    (y_test_mnist, y_test_fashion), axis=0
)


# ==========================================
# 11. SHUFFLE
# ==========================================

np.random.seed(42)

train_indices = np.random.permutation(len(x_train))
test_indices = np.random.permutation(len(x_test))

x_train = x_train[train_indices]
y_train = y_train[train_indices]

x_test = x_test[test_indices]
y_test = y_test[test_indices]


# ==========================================
# 12. FLATTEN IMAGES
# ==========================================

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)


# ==========================================
# 13. SAVE DATASETS
# ==========================================

np.save(os.path.join(SAVE_PATH, "combined_train.npy"), x_train)
np.save(os.path.join(SAVE_PATH, "combined_test.npy"), x_test)

np.save(os.path.join(SAVE_PATH, "combined_train_labels.npy"), y_train)
np.save(os.path.join(SAVE_PATH, "combined_test_labels.npy"), y_test)


# ==========================================
# 14. DISPLAY RESULTS
# ==========================================

print("\n========== COMBINED DATASET ==========")

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)

print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)

print("Minimum pixel:", np.min(x_train))
print("Maximum pixel:", np.max(x_train))

print("\nDATASET SAVED SUCCESSFULLY!")
print("Location:", SAVE_PATH)
