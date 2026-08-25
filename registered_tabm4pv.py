import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pv_curve_registration
import rtdl_num_embeddings
import sklearn.metrics
import sklearn.preprocessing
import tabm
import torch
import torch.nn as nn
import torch.optim
from torch import Tensor


seed = 0
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)


# >>> Dataset.
DATA_ROOT_PATH = "/data/hjs/your_parquet_directory/"
DATA_FILE_GLOB = "station=*.parquet"

TIMESTAMP_COL = "timestamp_win"
STATION_COL = "station"
PV_COL = "observe_power"
TARGET_COL = "observe_power_future"

FU_COV_COLUMNS = [
    "GHI_SOLARGIS_predict",
    "TEMP_SOLARGIS_predict",
    "WS_SOLARGIS_predict",
    "WD_SOLARGIS_predict",
]

TARGET_STATION = "replace_with_target_station"
SOURCE_STATIONS = None  # None means all stations except TARGET_STATION.
INCLUDE_TARGET_IN_TRAIN = False

STATION_CAPACITY = {
    # "source_station_1": 465.0,
    # "source_station_2": 520.0,
    # "replace_with_target_station": 480.0,
}

TRAIN_START = pd.Timestamp("2024-01-01 00:00:00")
TRAIN_END = pd.Timestamp("2024-12-31 23:59:59")
TEST_START = pd.Timestamp("2025-01-01 00:00:00")
TEST_END = pd.Timestamp("2025-12-31 23:59:59")
MAPPING_START = TRAIN_START
MAPPING_END = TRAIN_END

HISTORY_WINDOW_HOURS = 24
POINT_PER_HOUR = 4
TARGET_FUTURE_HOUR = 4
INPUT_LEN = HISTORY_WINDOW_HOURS * POINT_PER_HOUR
TARGET_INDEX = TARGET_FUTURE_HOUR * POINT_PER_HOUR - 1

# Crop the nested arrays immediately after reading each parquet file. TabM
# finally uses only 96 historical points, but two days are retained to leave
# enough interpolation margin for the time warp.
HISTORY_KEEP_DAYS = 2
HISTORY_KEEP_POINTS = HISTORY_KEEP_DAYS * 24 * POINT_PER_HOUR
FUTURE_KEEP_POINTS = TARGET_INDEX + 1

# Set to -15 if observe_power[-1] is timestamp_win - 15 minutes.
HISTORY_LAST_OFFSET_MINUTES = 0

PREDICTION_CLIP_LOWER = 0.0
PREDICTION_CLIP_UPPER_RATIO = 1.2
SCORE_CAPACITY = 465.0


def array_target(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) <= TARGET_INDEX:
        return np.nan
    return float(values[TARGET_INDEX])


def crop_history(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) < INPUT_LEN:
        raise ValueError(
            f"History length {len(values)} is smaller than "
            f"INPUT_LEN={INPUT_LEN}"
        )
    return values[-HISTORY_KEEP_POINTS:].copy()


def crop_future(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) < FUTURE_KEEP_POINTS:
        raise ValueError(
            f"Future length {len(values)} is smaller than "
            f"FUTURE_KEEP_POINTS={FUTURE_KEEP_POINTS}"
        )
    return values[:FUTURE_KEEP_POINTS].copy()


def process_single_file(df, station, station_warp):
    if df.empty:
        raise ValueError(f"No selected samples for station={station}")

    processed_dfs = []
    timestamp = pd.to_datetime(df[TIMESTAMP_COL])
    capacity = float(STATION_CAPACITY[station])

    processed_dfs.append(
        pd.DataFrame(
            {
                TIMESTAMP_COL: timestamp,
                STATION_COL: station,
                "capacity": capacity,
            },
            index=df.index,
        )
    )

    registered_history = np.stack(
        [
            pv_curve_registration.register_history(
                history,
                ts,
                capacity,
                station_warp,
                input_len=INPUT_LEN,
                history_last_offset_minutes=(
                    HISTORY_LAST_OFFSET_MINUTES
                ),
            )
            for history, ts in zip(df[PV_COL], timestamp)
        ]
    )
    history_columns = [
        f"{PV_COL}_lag_{i}"
        for i in range(INPUT_LEN, 0, -1)
    ]
    processed_dfs.append(
        pd.DataFrame(
            registered_history,
            columns=history_columns,
            index=df.index,
        )
    )

    for col_fut in FU_COV_COLUMNS:
        target_values = df[col_fut].map(array_target)
        processed_dfs.append(
            pd.DataFrame(
                {
                    f"{col_fut}_target": (
                        target_values.to_numpy()
                    )
                },
                index=df.index,
            )
        )

    target_power = df[TARGET_COL].map(array_target)
    processed_dfs.append(
        pd.DataFrame(
            {
                f"{TARGET_COL}_target": (
                    target_power.to_numpy() / capacity
                )
            },
            index=df.index,
        )
    )

    target_timestamp = timestamp + pd.to_timedelta(
        TARGET_FUTURE_HOUR,
        unit="h",
    )
    processed_dfs.append(
        pd.DataFrame(
            {
                "predict_hour": [
                    pv_curve_registration.physical_to_canonical_hour(
                        ts,
                        station_warp,
                    )
                    for ts in target_timestamp
                ],
                "predict_month": target_timestamp.dt.month,
            },
            index=df.index,
        )
    )

    return pd.concat(processed_dfs, axis=1)


station_frames = {}
for file in sorted(Path(DATA_ROOT_PATH).glob(DATA_FILE_GLOB)):
    df = pd.read_parquet(file)

    # Original rows contain seven days of history and two days of future
    # values. Keep only the part used by curve registration and TabM.
    df[PV_COL] = df[PV_COL].map(crop_history)
    for column in [TARGET_COL] + FU_COV_COLUMNS:
        df[column] = df[column].map(crop_future)

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    station_values = df[STATION_COL].dropna().astype(str).unique()
    if len(station_values) != 1:
        raise ValueError(f"{file.name}: station column is not constant")
    station = station_values[0]
    station_frames[station] = df.sort_values(TIMESTAMP_COL)

if not station_frames:
    raise FileNotFoundError(
        f"No {DATA_FILE_GLOB} files under {DATA_ROOT_PATH}"
    )
if TARGET_STATION not in station_frames:
    raise KeyError(f"TARGET_STATION={TARGET_STATION} is missing")

if SOURCE_STATIONS is None:
    source_stations = [
        station
        for station in station_frames
        if station != TARGET_STATION
    ]
else:
    source_stations = list(SOURCE_STATIONS)

if not source_stations:
    raise ValueError("At least one source station is required")
if TARGET_STATION in source_stations:
    raise ValueError(
        "TARGET_STATION must not be included in SOURCE_STATIONS"
    )
unknown_stations = [
    station
    for station in source_stations
    if station not in station_frames
]
if unknown_stations:
    raise KeyError(f"Unknown source stations: {unknown_stations}")

required_stations = source_stations + [TARGET_STATION]
missing_capacity = [
    station
    for station in required_stations
    if station not in STATION_CAPACITY
]
if missing_capacity:
    raise KeyError(
        f"STATION_CAPACITY is missing: {missing_capacity}"
    )
for station in required_stations:
    if float(STATION_CAPACITY[station]) <= 0:
        raise ValueError(f"Invalid capacity for station={station}")

station_warps = pv_curve_registration.fit_station_warps(
    station_frames,
    source_stations,
    TARGET_STATION,
    STATION_CAPACITY,
    MAPPING_START,
    MAPPING_END,
    timestamp_col=TIMESTAMP_COL,
    power_history_col=PV_COL,
    history_last_offset_minutes=HISTORY_LAST_OFFSET_MINUTES,
)

train_transform_list = []
for station in source_stations:
    df = station_frames[station]
    df = df[df[TIMESTAMP_COL].between(TRAIN_START, TRAIN_END)]
    train_transform_list.append(
        process_single_file(
            df,
            station,
            station_warps[station],
        )
    )

if INCLUDE_TARGET_IN_TRAIN:
    df = station_frames[TARGET_STATION]
    df = df[df[TIMESTAMP_COL].between(TRAIN_START, TRAIN_END)]
    train_transform_list.append(
        process_single_file(
            df,
            TARGET_STATION,
            station_warps[TARGET_STATION],
        )
    )

train_dataset = pd.concat(
    train_transform_list,
    ignore_index=True,
)

df = station_frames[TARGET_STATION]
df = df[df[TIMESTAMP_COL].between(TEST_START, TEST_END)]
test_dataset = process_single_file(
    df,
    TARGET_STATION,
    station_warps[TARGET_STATION],
).reset_index(drop=True)

x_columns = []
for col in FU_COV_COLUMNS:
    x_columns.append(f"{col}_target")
x_columns += [
    f"{PV_COL}_lag_{i}"
    for i in range(INPUT_LEN, 0, -1)
]
x_columns += ["predict_hour", "predict_month"]
y_columns = f"{TARGET_COL}_target"

train_dataset = (
    train_dataset
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=x_columns + [y_columns])
    .sort_values(TIMESTAMP_COL)
)
test_dataset = (
    test_dataset
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=x_columns + [y_columns])
    .sort_values(TIMESTAMP_COL)
)


def get_seasonal_split(df, val_days=5, date_col=TIMESTAMP_COL):
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    days_remaining = (
        data[date_col].dt.days_in_month
        - data[date_col].dt.day
    )
    val_mask = days_remaining < val_days
    return data[~val_mask], data[val_mask]


# Regression.
train_ds, valid_ds = get_seasonal_split(
    train_dataset,
    val_days=5,
)
if train_ds.empty or valid_ds.empty or test_dataset.empty:
    raise ValueError("Train, validation, and test data must all be non-empty")

X_num_train = train_ds[x_columns].values.astype(np.float32)
Y_train = train_ds[y_columns].values.astype(np.float32)
X_num_valid = valid_ds[x_columns].values.astype(np.float32)
Y_valid = valid_ds[y_columns].values.astype(np.float32)
X_num_test = test_dataset[x_columns].values.astype(np.float32)
Y_test = test_dataset[y_columns].values.astype(np.float32)
n_num_features = X_num_test.shape[1]

data_numpy = {
    "train": {"x_num": X_num_train, "y": Y_train},
    "val": {"x_num": X_num_valid, "y": Y_valid},
    "test": {"x_num": X_num_test, "y": Y_test},
}

for part, part_data in data_numpy.items():
    for key, value in part_data.items():
        print(
            f"{part:<5}    {key:<5}    "
            f"{value.shape!r:<10}    {value.dtype}"
        )

noise = (
    np.random.default_rng(seed)
    .normal(0.0, 1e-5, X_num_train.shape)
    .astype(X_num_train.dtype)
)
preprocessing = sklearn.preprocessing.QuantileTransformer(
    n_quantiles=max(
        min(X_num_train.shape[0] // 30, 1000),
        10,
    ),
    output_distribution="normal",
    subsample=10**9,
).fit(X_num_train + noise)

for part in data_numpy:
    data_numpy[part]["x_num"] = (
        preprocessing.transform(
            data_numpy[part]["x_num"]
        ).astype(np.float32)
    )

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)
data = {
    part: {
        key: torch.as_tensor(value, device=device)
        for key, value in part_data.items()
    }
    for part, part_data in data_numpy.items()
}
for part in data:
    data[part]["y"] = data[part]["y"].float()

amp_dtype = (
    torch.bfloat16
    if torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
    else torch.float16
    if torch.cuda.is_available()
    else None
)
amp_enabled = False and amp_dtype is not None
grad_scaler = (
    torch.cuda.amp.GradScaler()
    if amp_dtype is torch.float16
    else None
)

print(f"Device:        {device.type.upper()}")
print(f"AMP:           {amp_enabled}")

# Keep the same numerical embeddings as tabm4pv.py.
num_embeddings = (
    rtdl_num_embeddings.LinearReLUEmbeddings(
        n_num_features
    )
)
model = tabm.TabM.make(
    n_num_features=n_num_features,
    cat_cardinalities=[],
    d_out=1,
    num_embeddings=num_embeddings,
).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-3,
    weight_decay=3e-4,
)
gradient_clipping_norm: Optional[float] = 1.0
evaluation_mode = torch.inference_mode
share_training_batches = True


@torch.autocast(
    device.type,
    enabled=amp_enabled,
    dtype=amp_dtype,
)
def apply_model(part: str, idx: Tensor) -> Tensor:
    return (
        model(data[part]["x_num"][idx], None)
        .squeeze(-1)
        .float()
    )


def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
    y_pred = y_pred.flatten(0, 1)
    y_true = y_true.repeat_interleave(
        model.backbone.k
    )
    return nn.functional.mse_loss(y_pred, y_true)


@evaluation_mode()
def inference(part: str):
    model.eval()
    y_pred = (
        torch.cat(
            [
                apply_model(part, idx)
                for idx in torch.arange(
                    len(data[part]["y"]),
                    device=device,
                ).split(512)
            ]
        )
        .cpu()
        .numpy()
        .mean(1)
    )
    y_true = data[part]["y"].cpu().numpy()
    score = -(
        sklearn.metrics.mean_squared_error(
            y_true,
            y_pred,
        )
        ** 0.5
    )
    return float(score), y_true, y_pred


def evaluate(part: str):
    return inference(part)[0]


print(f'Test score before training: {evaluate("test"):.4f}')

n_epochs = 200
train_size = X_num_train.shape[0]
batch_size = 512
epoch = -1
metrics = {"val": -math.inf, "test": -math.inf}


def make_checkpoint() -> dict[str, Any]:
    return deepcopy(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        }
    )


best_checkpoint = make_checkpoint()
patience = 10
remaining_patience = patience

for epoch in range(n_epochs):
    batches = torch.randperm(
        train_size,
        device=device,
    ).split(batch_size)

    for batch_idx in batches:
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(
            apply_model("train", batch_idx),
            data["train"]["y"][batch_idx],
        )

        if grad_scaler is None:
            loss.backward()
        else:
            grad_scaler.scale(loss).backward()

        if gradient_clipping_norm is not None:
            if grad_scaler is not None:
                grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(),
                gradient_clipping_norm,
            )

        if grad_scaler is None:
            optimizer.step()
        else:
            grad_scaler.step(optimizer)
            grad_scaler.update()

    metrics = {
        part: evaluate(part)
        for part in ["val", "test"]
    }
    val_score_improved = (
        metrics["val"]
        > best_checkpoint["metrics"]["val"]
    )

    print(
        f'{"*" if val_score_improved else " "}'
        f" [epoch] {epoch:<3}"
        f' [val] {metrics["val"]:.3f}'
        f' [test] {metrics["test"]:.3f}'
    )

    if val_score_improved:
        best_checkpoint = make_checkpoint()
        remaining_patience = patience
    else:
        remaining_patience -= 1

    if remaining_patience < 0:
        break

model.load_state_dict(best_checkpoint["model"])

print("\n[Summary]")
print(f'best epoch:  {best_checkpoint["epoch"]}')
print(f'val score:  {best_checkpoint["metrics"]["val"]}')
print(f'test score: {best_checkpoint["metrics"]["test"]}')

_, groundtruth_ratio, prediction_ratio = inference("test")
prediction_ratio = np.clip(
    prediction_ratio,
    PREDICTION_CLIP_LOWER,
    PREDICTION_CLIP_UPPER_RATIO,
)

capacity = test_dataset["capacity"].to_numpy()
test_dataset = test_dataset.copy()
test_dataset["groundtruth"] = groundtruth_ratio * capacity
test_dataset["prediction"] = prediction_ratio * capacity
test_dataset.to_parquet(
    "best_prediction_registered_tabm.parquet",
    index=False,
)


def rmse(y_true, y_pred):
    return np.sqrt(
        ((y_true - y_pred) ** 2).mean()
    )


result = (
    test_dataset.groupby(
        test_dataset[TIMESTAMP_COL].dt.date
    )
    .apply(
        lambda g: pd.Series(
            {
                "rmse_final_pred": rmse(
                    g["prediction"],
                    g["groundtruth"],
                )
            }
        )
    )
    .reset_index()
    .rename(columns={TIMESTAMP_COL: "date"})
)

result["date"] = pd.to_datetime(result["date"])
score_list = []
for i in range(12):
    sub = result[result["date"].dt.month == (i + 1)]
    s_post = (
        1
        - sub["rmse_final_pred"].mean()
        / SCORE_CAPACITY
    )
    print(
        f"{i + 1} month score post process: "
        f"{s_post: .4f}"
    )
    score_list.append(s_post)

print(f"score mean: {sum(score_list) / len(score_list)}")
