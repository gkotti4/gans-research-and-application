# generate_mnist_csv.py
# Utility script for converting MNIST .ubyte files to CSV format.
# Adapted from public domain MNIST conversion examples.
# Included here for dataset preparation and reproducibility only.

#The format is
#    label, pix-11, pix-12, pix-13, ...

def convert(imgf, labelf, outf, n):
    with open(imgf, "rb") as f, open(labelf, "rb") as l, open(outf, "w") as o:
        f.read(16)
        l.read(8)
        images = []

        for i in range(n):
            label_byte = l.read(1)
            if not label_byte:
                print(f"Reached end of label file at {i} samples.")
                break

            image_bytes = f.read(28*28)
            if len(image_bytes) < 28*28:
                print(f"Reached end of image file at {i} samples.")
                break

            label = ord(label_byte)
            image = [label] + [ord(b) for b in image_bytes]
            images.append(image)

        for image in images:
            o.write(",".join(map(str, image)) + "\n")



# 			=== MAIN ===

# Paths to MNIST ubyte data files
images_path = "train-images.idx3-ubyte"
labels_path = "train-labels.idx1-ubyte"

# Output CSV and number of samples to process
output_csv = "mnist_train.csv"
num_images = 60000

try:
    convert(images_path, labels_path,
            output_csv, num_images)
except FileNotFoundError as e:
    print(f"MNIST file not found: {e.filename}")
except Exception as e:
    print("Unable to convert MNIST ubyte to csv:", e)
    
    
    

# EXAMPLE USAGE:
#     convert("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte",
#             "mnist_test.csv", 10000)



