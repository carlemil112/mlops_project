import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def analyze_dataset(image_dir):
    class_counts = {}
    image_sizes = []

    # Variabler til udregning af Mean og Std. af pixel intensity
    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixels = 0

    for class_name in os.listdir(image_dir):
        class_path = os.path.join(image_dir, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            class_counts[class_name] = len(images)

            for img_name in images:
                img_path = os.path.join(class_path, img_name)

                # Indlæs som greyscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is not None:
                    image_sizes.append(img.shape[:2])  # (height, width)

                    # Normaliser data til float [0, 1] for beregning af statistik
                    img_norm = img.astype(np.float32) / 255.0

                    # Opdater løbende variabler for total til beregning
                    total_sum += np.sum(img_norm)
                    total_sq_sum += np.sum(img_norm**2)
                    total_pixels += img_norm.size

                else:
                    print(f"Warning: Could not read image '{img_path}'")

    # Beregning af Mean og std
    mean_val = total_sum / total_pixels
    std_val = np.sqrt((total_sq_sum / total_pixels) - (mean_val**2))

    return class_counts, image_sizes, mean_val, std_val
    cv2.imwrite(f"augmented_image_{i}.png", img)


if __name__ == "__main__":
    dataset_path = "FER-2013/train_balanced"
    if not os.path.exists(dataset_path):
        dataset_path = "FER-2013/train"

    class_counts, image_sizes, mean_val, std_val = analyze_dataset(dataset_path)

    print("\n" + "*" * 30)
    print("Pixel analysis:")
    print("*" * 30)
    print("\nGreyscale pixel statistics ([0, 1] range):")
    print(f"Mean: {mean_val:.4f}")
    print(f"Std: {std_val:.4f}")
    print("*" * 30)
    print("VIGTIGT: Husk at kopier disse værdier ind i train.py!")
    print("*" * 30)

    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")

    heights, widths = zip(*image_sizes)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(heights, bins=30, color="blue", alpha=0.7)
    plt.title("Image Height Distribution")
    plt.xlabel("Height (pixels)")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(widths, bins=30, color="green", alpha=0.7)
    plt.title("Image Width Distribution")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
