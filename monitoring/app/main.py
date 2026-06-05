from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

app = FastAPI(title="MLOps Monitoring Exporter")

train_loss = Gauge("mlops_train_loss", "Latest training loss")
train_accuracy = Gauge("mlops_train_accuracy", "Latest training accuracy")
validation_loss = Gauge("mlops_validation_loss", "Latest validation loss")
validation_accuracy = Gauge("mlops_validation_accuracy", "Latest validation accuracy")
current_epoch = Gauge("mlops_current_epoch", "Latest completed training epoch")

scale_drift_score = Gauge("mlops_scale_drift_score", "Scale drift score")
color_drift_score = Gauge("mlops_color_drift_score", "Color drift score")
scale_drift_p_value = Gauge("mlops_scale_drift_p_value", "Scale drift p-value")
color_drift_p_value = Gauge("mlops_color_drift_p_value", "Color drift p-value")

pipeline_stage_success = Gauge(
    "mlops_pipeline_stage_success",
    "Pipeline stage success status, 1 means success and 0 means failure",
    ["stage"],
)


class TrainingMetrics(BaseModel):
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


class DriftMetrics(BaseModel):
    scale_drift_score: float
    color_drift_score: float
    scale_drift_p_value: float
    color_drift_p_value: float


class StageMetrics(BaseModel):
    stage: str
    success: int


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/metrics/training")
def update_training_metrics(metrics: TrainingMetrics):
    current_epoch.set(metrics.epoch)
    train_loss.set(metrics.train_loss)
    train_accuracy.set(metrics.train_acc)
    validation_loss.set(metrics.val_loss)
    validation_accuracy.set(metrics.val_acc)
    return {"status": "updated"}


@app.post("/metrics/drift")
def update_drift_metrics(metrics: DriftMetrics):
    scale_drift_score.set(metrics.scale_drift_score)
    color_drift_score.set(metrics.color_drift_score)
    scale_drift_p_value.set(metrics.scale_drift_p_value)
    color_drift_p_value.set(metrics.color_drift_p_value)
    return {"status": "updated"}


@app.post("/metrics/stage")
def update_stage_metrics(metrics: StageMetrics):
    pipeline_stage_success.labels(stage=metrics.stage).set(metrics.success)
    return {"status": "updated"}
