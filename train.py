import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
    Input,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import mlflow
import mlflow.keras

#MLFlow configuration
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)


mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "MLFlow FER tracking"))
# Save model runs for lineage
mlflow.keras.autolog(log_models=True)


# Konfiguration og parametre
TRAIN_DIR = "FER-2013/train_balanced"  # Balanceret data
IMG_SIZE = 48  # Standard størrelse billede
BATCH_SIZE = 64  # Antal billeder per batch
EPOCHS = 50  # Maksimalt antal gennemløb

# Brug de nye mean og std pixel-værdier fra dataset_analysis (placeholder [0.5, 0.25])
DATASET_MEAN = 0.5147
DATASET_STD = 0.2536


def custom_preprocessing(img):
    """
    Denne funktion kører på hvert eneste billede før det rammer modellen.
    Den normaliserer pixel-værdierne baseret på hele datasættet.
    """
    # 1. Skaler fra [0, 255] til [0, 1]
    img = img / 255.0

    # 2. Standardisering: (pixel - gennemsnit) / standardafvigelse
    # Dette centrerer data omkring 0, hvilket hjælper modellen med at lære hurtigere.
    img = (img - DATASET_MEAN) / DATASET_STD
    return img


# Data generators
print("Opsætter data generators...")

# Vi bruger ImageDataGenerator til at streame billeder fra disken
datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing,  # Vores custom normalisering
    validation_split=0.2,  # 20% data til validation
)

# Generator til træningsdata
train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",  # Sikrer at vi kun bruger 1 kanal for sort/hvid billeder
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

# Generator til Valideringsdata
validation_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

with mlflow.start_run() as run:
    mlflow.set_tags({
        "git.commit": os.getenv("GIT_COMMIT", ""),
        "git.branch": os.getenv("GIT_BRANCH", ""),
        "jenkins.job": os.getenv("JOB_NAME", ""),
        "jenkins.build_number": os.getenv("BUILD_NUMBER", ""),
        "jenkins.build_url": os.getenv("BUILD_URL", ""),
        "data.version": os.getenv("DATA_VERSION", ""),
    })

    mlflow.log_params({
        "train_dir": TRAIN_DIR,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_max": EPOCHS,
        "dataset_mean": DATASET_MEAN,
        "dataset_std": DATASET_STD,
    })

  

# Model-arkitektur (CNN)
# Lag på lag (sekventiel)
model = Sequential()

# Input lag (definerer formen: 48x48x1)
model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 1)))

# Conv2D: finder simple features (kanter, linjer)
# BatchNormalization: stabiliserer læringen
# MaxPooling: gør billedet mindre (halverer størrelsen) for at reducere beregninger
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(
    Dropout(0.4)
)  # Slukker tilfældige neuroner for at hjællpe med at forhindre overfitting

# Dybden øges (128 filtre) for at finde mere komplekse mønstre (former, øjne)(samme opbygning)
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

# Dybden øges igen (156 filtre)
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

# Flatten: laver 2D billedet om til en lang liste af tal. Fully connected lag
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Output Lag: 7 neuroner  en for hver følelse - med Softmax (sandsynligheder i % (zero sum))
num_classes = train_generator.num_classes
model.add(Dense(num_classes, activation="softmax"))

# Oversigt over modellen
model.summary()

# Træning
# Callbacks: funktioner der kører under træning i model.fit
# Find og gem bedste model
#OUTPUTS MADE RUN-SPECIFIC
out_dir = os.path.join("outputs", run.info.run_id)
os.makedirs(out_dir, exist_ok=True)

best_model_path = os.path.join(out_dir, "best_emotion_model.keras")

checkpoint = ModelCheckpoint(best_model_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)

# Early stopping (før 50 epochs)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,  # Stop hvis den ikke ser forbedring efter 10 epochs
    restore_best_weights=True,
)

# Kontroller learning_rate dynamisk
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001, verbose=1
)

# Boiler-plate for setup af model før læring (strategi)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Fit modellen på træningsdata
print("Starter træning...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[checkpoint, early_stop, reduce_lr],
)


# Resultater visualisering
def plot_training_history(history, plotpath):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training vs Validation Accuracy")

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training vs Validation Loss")

    plt.savefig(history, plot_path,)

# Plots + best model in MLFLow
plot_path = os.path.join(out_dir, "training_results.png")
plot_training_history(history, plot_path)

mlflow.log_artifact(plot_path, artifact_path="plots")
mlflow.log_artifact(best_model_path, artifact_path="checkpoints")