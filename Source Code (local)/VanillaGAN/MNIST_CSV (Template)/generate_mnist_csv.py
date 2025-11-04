# generate_mnist_csv.py
# Utility script for converting MNIST .ubyte files to CSV format.
# Adapted from public domain MNIST conversion examples (various sources, 2016–2019).
# Included here for dataset preparation and reproducibility only.

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

try:
    convert("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
            "mnist_train.csv", 60000)
    convert("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte",
            "mnist_test.csv", 10000)
except FileNotFoundError as e:
    print(f"MNIST file not found: {e.filename}")
except Exception as e:
    print("Unable to convert MNIST ubyte to csv:", e)

