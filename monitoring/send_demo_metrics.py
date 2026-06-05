import json
import time
import urllib.error
import urllib.request


BASE_URL = "http://localhost:8000"


def post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        response.read()


def main():
    stages = ["pytest", "dvc_pull", "training", "drift_detection"]
    for stage in stages:
        post("/metrics/stage", {"stage": stage, "success": 1})

    training_points = [
        {
            "epoch": 1,
            "train_loss": 1.32,
            "train_acc": 0.38,
            "val_loss": 1.41,
            "val_acc": 0.35,
        },
        {
            "epoch": 2,
            "train_loss": 1.08,
            "train_acc": 0.49,
            "val_loss": 1.19,
            "val_acc": 0.44,
        },
        {
            "epoch": 3,
            "train_loss": 0.91,
            "train_acc": 0.58,
            "val_loss": 1.02,
            "val_acc": 0.53,
        },
        {
            "epoch": 4,
            "train_loss": 0.77,
            "train_acc": 0.66,
            "val_loss": 0.90,
            "val_acc": 0.61,
        },
        {
            "epoch": 5,
            "train_loss": 0.65,
            "train_acc": 0.73,
            "val_loss": 0.82,
            "val_acc": 0.67,
        },
    ]

    for point in training_points:
        post("/metrics/training", point)
        print(f"Sent epoch {point['epoch']} metrics")
        time.sleep(6)

    post(
        "/metrics/drift",
        {
            "scale_drift_score": 0.18,
            "color_drift_score": 0.27,
            "scale_drift_p_value": 0.42,
            "color_drift_p_value": 0.08,
        },
    )
    print("Sent drift metrics")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as error:
        raise SystemExit(
            "Could not reach the monitoring app. Start it with: "
            "cd monitoring && docker-compose up --build"
        ) from error
