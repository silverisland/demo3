"""ShapeEncoder + TabM for four-hour-ahead PV power forecasting.

This version intentionally follows the style of the original ``tabm4pv.py``:
all experiment settings are fixed in the constants section below, with no
command-line arguments.
"""

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import rtdl_num_embeddings
import sklearn.metrics
import sklearn.preprocessing
import tabm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# =============================================================================
# Experiment configuration
# =============================================================================

SEED = 0

# Data files.
DATA_ROOT_PATH = "/data/hjs/1219_report_onv8/"
DATA_FILE_SUFFIX = "_v1.parquet"
DATA_FILE_PREFIX = ""  # Example: "mkv82"; empty means all stations.

# Array-valued parquet columns.
TIMESTAMP_COL = "timestamp_win"
POWER_HISTORY_COL = "Power"
POWER_FUTURE_COL = "Power_predict"
FUTURE_COVARIATE_COLUMNS = [
    "GHI_SOLARGIS_predict",
    "TEMP_SOLARGIS_predict",
    "WS_SOLARGIS_predict",
    "WD_SOLARGIS_predict",
]

# 15-minute data: use the latest day and predict the 16th future point.
POINTS_PER_HOUR = 4
HISTORY_LENGTH = 24 * POINTS_PER_HOUR
TARGET_FUTURE_HOUR = 4
TARGET_INDEX = TARGET_FUTURE_HOUR * POINTS_PER_HOUR - 1
MINUTES_PER_POINT = 15

# Capacity normalization is important for cross-station OOD.
# Set CAPACITY_COL to the real scalar capacity column when it exists.
CAPACITY_COL: Optional[str] = None
DEFAULT_CAPACITY = 500.0
SCORE_CAPACITY = 465.0
HISTORY_RATIO_CLIP = (0.0, 1.2)
PREDICTION_RATIO_CLIP = (0.0, 1.05)

# Split strategy.
# "time": reproduce the old single-station time split.
# "station": hold out complete station files by filename prefix.
SPLIT_MODE = "time"

TRAIN_YEAR = 2024
TRAIN_START_MONTH = 9
TEST_YEAR = 2025
VALIDATION_LAST_DAYS = 5

# Used only when SPLIT_MODE == "station".
# Example:
# OOD_VALIDATION_FILE_PREFIXES = ("mkv81",)
# OOD_TEST_FILE_PREFIXES = ("mkv82",)
OOD_VALIDATION_FILE_PREFIXES: tuple[str, ...] = ()
OOD_TEST_FILE_PREFIXES: tuple[str, ...] = ()

# Shape encoder.
ENCODER_CHANNELS = (16, 32, 64)
ENCODER_LATENT_DIM = 32

# TabM.
TABM_K = 32
TABM_N_BLOCKS = 3
TABM_D_BLOCK = 512
TABM_DROPOUT = 0.1

# Keep the numerical embeddings used by the original tabm4pv.py.
USE_RTDL_NUM_EMBEDDINGS = True

# Training.
N_EPOCHS = 200
BATCH_SIZE = 512
ENCODER_LEARNING_RATE = 3e-4
TABM_LEARNING_RATE = 2e-3
WEIGHT_DECAY = 3e-4
GRADIENT_CLIPPING_NORM = 1.0
EARLY_STOPPING_PATIENCE = 10
BALANCE_STATIONS = True

# Optional representation-invariance regularization.
# Start with 0.0 to measure the effect of Encoder + TabM alone.
INVARIANCE_LOSS_WEIGHT = 0.0
INVARIANCE_GAIN_RANGE = (0.7, 1.3)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("./outputs/encoder_tabm")


random.seed(SEED)
np.random.seed(SEED + 1)
torch.manual_seed(SEED + 2)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED + 3)


# =============================================================================
# Model
# =============================================================================


class ShapeEncoder(nn.Module):
    """Encode a normalized 24-hour PV curve into a compact representation."""

    def __init__(
        self,
        history_length: int,
        latent_dim: int,
        channels: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.history_length = history_length
        c1, c2, c3 = channels

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(c1, c2, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(c2, c3, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            # Four bins preserve coarse morning/noon/evening position.
            nn.AdaptiveAvgPool1d(4),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

    def forward(self, power_history: Tensor) -> Tensor:
        if power_history.ndim != 2:
            raise ValueError(
                "power_history must have shape (batch, history_length)"
            )
        if power_history.shape[1] != self.history_length:
            raise ValueError(
                f"Expected {self.history_length} history points, "
                f"got {power_history.shape[1]}"
            )
        x = power_history.unsqueeze(1)
        return self.projection(self.feature_extractor(x))


class EncoderTabM(nn.Module):
    """Jointly trained ShapeEncoder and TabM regressor."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = ShapeEncoder(
            history_length=HISTORY_LENGTH,
            latent_dim=ENCODER_LATENT_DIM,
            channels=ENCODER_CHANNELS,
        )

        # Encoder latent + future NWP + cyclic hour/day-of-year features.
        n_tabm_features = ENCODER_LATENT_DIM + len(FUTURE_COVARIATE_COLUMNS) + 4

        num_embeddings = (
            rtdl_num_embeddings.LinearReLUEmbeddings(n_tabm_features)
            if USE_RTDL_NUM_EMBEDDINGS
            else None
        )

        self.tabm = tabm.TabM.make(
            n_num_features=n_tabm_features,
            cat_cardinalities=[],
            d_out=1,
            num_embeddings=num_embeddings,
            k=TABM_K,
            n_blocks=TABM_N_BLOCKS,
            d_block=TABM_D_BLOCK,
            dropout=TABM_DROPOUT,
        )

    def encode(self, power_history: Tensor) -> Tensor:
        return self.encoder(power_history)

    def forward(
        self,
        power_history: Tensor,
        future_covariates: Tensor,
        time_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        z_shape = self.encode(power_history)
        x_num = torch.cat(
            [z_shape, future_covariates, time_features],
            dim=1,
        )
        # (batch, k, 1) -> (batch, k)
        member_predictions = self.tabm(x_num, None).squeeze(-1)
        return member_predictions, z_shape


# =============================================================================
# Data
# =============================================================================


class PVDataset(Dataset):
    def __init__(
        self,
        power_history: np.ndarray,
        future_covariates: np.ndarray,
        time_features: np.ndarray,
        target_ratio: np.ndarray,
        capacity: np.ndarray,
        station_index: np.ndarray,
    ) -> None:
        self.power_history = torch.as_tensor(power_history, dtype=torch.float32)
        self.future_covariates = torch.as_tensor(
            future_covariates, dtype=torch.float32
        )
        self.time_features = torch.as_tensor(time_features, dtype=torch.float32)
        self.target_ratio = torch.as_tensor(target_ratio, dtype=torch.float32)
        self.capacity = torch.as_tensor(capacity, dtype=torch.float32)
        self.station_index = np.asarray(station_index, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.target_ratio)

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        return (
            self.power_history[index],
            self.future_covariates[index],
            self.time_features[index],
            self.target_ratio[index],
            self.capacity[index],
        )


def stack_array_column(series: pd.Series, column: str) -> np.ndarray:
    try:
        values = np.stack(
            series.map(lambda value: np.asarray(value, dtype=np.float32))
        )
    except ValueError as error:
        raise ValueError(
            f"Column {column!r} contains arrays with inconsistent lengths"
        ) from error

    if values.ndim != 2:
        raise ValueError(
            f"Column {column!r} must contain one-dimensional arrays"
        )
    return values


def load_samples() -> pd.DataFrame:
    paths = sorted(Path(DATA_ROOT_PATH).glob(f"*{DATA_FILE_SUFFIX}"))
    if DATA_FILE_PREFIX:
        paths = [
            path for path in paths if path.name.startswith(DATA_FILE_PREFIX)
        ]
    if not paths:
        raise FileNotFoundError(
            f"No '*{DATA_FILE_SUFFIX}' files found under {DATA_ROOT_PATH}"
        )

    required_columns = {
        TIMESTAMP_COL,
        POWER_HISTORY_COL,
        POWER_FUTURE_COL,
        *FUTURE_COVARIATE_COLUMNS,
    }
    frames = []

    for path in paths:
        frame = pd.read_parquet(path)
        missing = required_columns.difference(frame.columns)
        if missing:
            raise KeyError(
                f"{path.name} is missing columns: {sorted(missing)}"
            )

        history = stack_array_column(
            frame[POWER_HISTORY_COL], POWER_HISTORY_COL
        )
        if history.shape[1] < HISTORY_LENGTH:
            raise ValueError(
                f"{path.name}: only {history.shape[1]} history points"
            )
        history = history[:, -HISTORY_LENGTH:]

        future_values = {}
        for column in [*FUTURE_COVARIATE_COLUMNS, POWER_FUTURE_COL]:
            values = stack_array_column(frame[column], column)
            if values.shape[1] <= TARGET_INDEX:
                raise ValueError(
                    f"{path.name}: {column!r} does not contain target "
                    f"index {TARGET_INDEX}"
                )
            future_values[column] = values[:, TARGET_INDEX]

        if CAPACITY_COL is None:
            capacity = np.full(
                len(frame), DEFAULT_CAPACITY, dtype=np.float32
            )
        else:
            if CAPACITY_COL not in frame.columns:
                raise KeyError(
                    f"{path.name} is missing capacity column {CAPACITY_COL!r}"
                )
            capacity = pd.to_numeric(
                frame[CAPACITY_COL], errors="coerce"
            ).to_numpy(dtype=np.float32)

        if not np.isfinite(capacity).all() or (capacity <= 0).any():
            raise ValueError(
                f"{path.name}: capacity must be finite and positive"
            )

        timestamp = pd.to_datetime(frame[TIMESTAMP_COL], errors="coerce")
        if timestamp.isna().any():
            raise ValueError(f"{path.name}: invalid timestamp")

        transformed = pd.DataFrame(
            {
                "timestamp": timestamp.to_numpy(),
                "station": path.stem,
                "source_file": path.name,
                "capacity": capacity,
                "target_power": future_values[POWER_FUTURE_COL],
            }
        )
        transformed["power_history"] = list(history)
        transformed["future_covariates"] = list(
            np.column_stack(
                [
                    future_values[column]
                    for column in FUTURE_COVARIATE_COLUMNS
                ]
            )
        )
        frames.append(transformed)

    samples = pd.concat(frames, ignore_index=True)
    samples = samples.sort_values(
        ["timestamp", "station"]
    ).reset_index(drop=True)
    print(
        f"Loaded {len(samples):,} samples from "
        f"{samples['station'].nunique()} station files"
    )
    return samples


def matches_prefix(filename: str, prefixes: tuple[str, ...]) -> bool:
    return any(filename.startswith(prefix) for prefix in prefixes)


def split_samples(
    samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if SPLIT_MODE == "time":
        timestamp = samples["timestamp"]
        train_pool = samples[
            (timestamp.dt.year == TRAIN_YEAR)
            & (timestamp.dt.month >= TRAIN_START_MONTH)
        ].copy()
        test = samples[timestamp.dt.year == TEST_YEAR].copy()

        days_remaining = (
            train_pool["timestamp"].dt.days_in_month
            - train_pool["timestamp"].dt.day
        )
        validation_mask = days_remaining < VALIDATION_LAST_DAYS
        train = train_pool[~validation_mask].copy()
        validation = train_pool[validation_mask].copy()

    elif SPLIT_MODE == "station":
        if not OOD_VALIDATION_FILE_PREFIXES:
            raise ValueError(
                "Set OOD_VALIDATION_FILE_PREFIXES for station split"
            )
        if not OOD_TEST_FILE_PREFIXES:
            raise ValueError(
                "Set OOD_TEST_FILE_PREFIXES for station split"
            )

        test_mask = samples["source_file"].map(
            lambda name: matches_prefix(name, OOD_TEST_FILE_PREFIXES)
        )
        validation_mask = samples["source_file"].map(
            lambda name: matches_prefix(
                name, OOD_VALIDATION_FILE_PREFIXES
            )
        )
        if (test_mask & validation_mask).any():
            raise ValueError(
                "Validation and test station prefixes overlap"
            )

        train = samples[~test_mask & ~validation_mask].copy()
        validation = samples[validation_mask].copy()
        test = samples[test_mask].copy()
    else:
        raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE!r}")

    for name, frame in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        if frame.empty:
            raise ValueError(f"{name} split is empty")
        print(
            f"{name:<10} {len(frame):>8,} samples, "
            f"{frame['station'].nunique():>3} stations"
        )

    return train, validation, test


def make_time_features(timestamp: pd.Series) -> np.ndarray:
    # TARGET_INDEX=15 means 16 future intervals, i.e. four hours.
    forecast_time = timestamp + pd.to_timedelta(
        (TARGET_INDEX + 1) * MINUTES_PER_POINT,
        unit="minute",
    )
    hour = (
        forecast_time.dt.hour.to_numpy()
        + forecast_time.dt.minute.to_numpy() / 60.0
    )
    day_of_year = forecast_time.dt.dayofyear.to_numpy()

    return np.column_stack(
        [
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            np.sin(2.0 * np.pi * day_of_year / 365.25),
            np.cos(2.0 * np.pi * day_of_year / 365.25),
        ]
    ).astype(np.float32)


def prepare_dataset(
    frame: pd.DataFrame,
    covariate_transformer: sklearn.preprocessing.QuantileTransformer,
) -> PVDataset:
    capacity = frame["capacity"].to_numpy(dtype=np.float32)

    power_history = np.stack(frame["power_history"]).astype(np.float32)
    power_history = power_history / capacity[:, None]
    power_history = np.clip(power_history, *HISTORY_RATIO_CLIP)

    target_power = frame["target_power"].to_numpy(dtype=np.float32)
    target_ratio = target_power / capacity

    future_covariates = np.stack(
        frame["future_covariates"]
    ).astype(np.float32)
    future_covariates = covariate_transformer.transform(
        future_covariates
    ).astype(np.float32)

    time_features = make_time_features(frame["timestamp"])
    station_index, _ = pd.factorize(frame["station"], sort=True)

    return PVDataset(
        power_history=power_history,
        future_covariates=future_covariates,
        time_features=time_features,
        target_ratio=target_ratio,
        capacity=capacity,
        station_index=station_index,
    )


def make_train_loader(dataset: PVDataset) -> DataLoader:
    if not BALANCE_STATIONS:
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

    station_counts = np.bincount(dataset.station_index)
    sample_weights = 1.0 / station_counts[dataset.station_index]
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
    )


# =============================================================================
# Training and evaluation
# =============================================================================


def move_batch(
    batch: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    return tuple(value.to(DEVICE) for value in batch)


def tabm_member_loss(
    member_predictions: Tensor,
    target: Tensor,
) -> Tensor:
    """Train every TabM member independently, as required by TabM."""
    target_for_members = target.unsqueeze(1).expand_as(member_predictions)
    return F.mse_loss(member_predictions, target_for_members)


def train_epoch(
    model: EncoderTabM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0

    for batch in loader:
        (
            power_history,
            future_covariates,
            time_features,
            target,
            _,
        ) = move_batch(batch)

        optimizer.zero_grad(set_to_none=True)
        member_predictions, z_shape = model(
            power_history,
            future_covariates,
            time_features,
        )
        forecast_loss = tabm_member_loss(member_predictions, target)
        loss = forecast_loss

        if INVARIANCE_LOSS_WEIGHT > 0.0:
            gain = torch.empty(
                (len(power_history), 1),
                device=DEVICE,
            ).uniform_(*INVARIANCE_GAIN_RANGE)
            z_augmented = model.encode(power_history * gain)
            invariance_loss = F.mse_loss(z_augmented, z_shape)
            loss = (
                loss
                + INVARIANCE_LOSS_WEIGHT * invariance_loss
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIPPING_NORM,
        )
        optimizer.step()

        loss_sum += forecast_loss.item() * len(target)
        sample_count += len(target)

    return loss_sum / sample_count


@torch.inference_mode()
def predict(
    model: EncoderTabM,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    targets = []
    predictions = []
    capacities = []

    for batch in loader:
        (
            power_history,
            future_covariates,
            time_features,
            target,
            capacity,
        ) = move_batch(batch)

        member_predictions, _ = model(
            power_history,
            future_covariates,
            time_features,
        )
        prediction = member_predictions.mean(dim=1)
        prediction = prediction.clamp(*PREDICTION_RATIO_CLIP)

        targets.append(target.cpu().numpy())
        predictions.append(prediction.cpu().numpy())
        capacities.append(capacity.cpu().numpy())

    return (
        np.concatenate(targets),
        np.concatenate(predictions),
        np.concatenate(capacities),
    )


def calculate_metrics(
    target_ratio: np.ndarray,
    prediction_ratio: np.ndarray,
    capacity: np.ndarray,
) -> dict[str, float]:
    target_power = target_ratio * capacity
    prediction_power = prediction_ratio * capacity

    return {
        "rmse_ratio": float(
            np.sqrt(
                sklearn.metrics.mean_squared_error(
                    target_ratio,
                    prediction_ratio,
                )
            )
        ),
        "mae_ratio": float(
            sklearn.metrics.mean_absolute_error(
                target_ratio,
                prediction_ratio,
            )
        ),
        "rmse_power": float(
            np.sqrt(
                sklearn.metrics.mean_squared_error(
                    target_power,
                    prediction_power,
                )
            )
        ),
        "mae_power": float(
            sklearn.metrics.mean_absolute_error(
                target_power,
                prediction_power,
            )
        ),
    }


def calculate_monthly_score(
    prediction_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Reproduce the daily-RMSE monthly score from the original script."""
    data = prediction_frame.copy()
    data["date"] = pd.to_datetime(data["timestamp"]).dt.normalize()
    data["squared_error"] = (
        data["prediction_power"] - data["target_power"]
    ) ** 2

    daily_result = (
        data.groupby("date", as_index=False)["squared_error"]
        .mean()
        .rename(columns={"squared_error": "mse_final_pred"})
    )
    daily_result["rmse_final_pred"] = np.sqrt(
        daily_result["mse_final_pred"]
    )
    daily_result["month"] = daily_result["date"].dt.month

    monthly_rows = []
    score_list = []

    for month in range(1, 13):
        month_data = daily_result[daily_result["month"] == month]
        if month_data.empty:
            print(f"{month} month score post process: no data")
            continue

        mean_daily_rmse = float(
            month_data["rmse_final_pred"].mean()
        )
        score = 1.0 - mean_daily_rmse / SCORE_CAPACITY
        print(f"{month} month score post process: {score:.4f}")

        monthly_rows.append(
            {
                "month": month,
                "mean_daily_rmse": mean_daily_rmse,
                "score": score,
            }
        )
        score_list.append(score)

    score_mean = (
        float(np.mean(score_list))
        if score_list
        else float("nan")
    )
    print(f"score mean: {score_mean:.4f}")

    return pd.DataFrame(monthly_rows), score_mean


def evaluate(
    model: EncoderTabM,
    loader: DataLoader,
) -> dict[str, float]:
    target, prediction, capacity = predict(model, loader)
    return calculate_metrics(target, prediction, capacity)


def make_checkpoint(
    model: EncoderTabM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_metrics: dict[str, float],
) -> dict[str, Any]:
    return deepcopy(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "validation_metrics": validation_metrics,
        }
    )


# =============================================================================
# Main experiment
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_samples()
    train_frame, validation_frame, test_frame = split_samples(samples)

    if CAPACITY_COL is None and samples["station"].nunique() > 1:
        print(
            "Warning: all stations use DEFAULT_CAPACITY. Set CAPACITY_COL "
            "to the real capacity column for a meaningful OOD experiment."
        )

    train_covariates = np.stack(
        train_frame["future_covariates"]
    ).astype(np.float32)
    noise = np.random.default_rng(SEED).normal(
        0.0,
        1e-5,
        train_covariates.shape,
    ).astype(np.float32)
    covariate_transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(
            min(len(train_frame) // 30, 1000),
            10,
        ),
        output_distribution="normal",
        subsample=10**9,
        random_state=SEED,
    ).fit(train_covariates + noise)

    train_dataset = prepare_dataset(
        train_frame,
        covariate_transformer,
    )
    validation_dataset = prepare_dataset(
        validation_frame,
        covariate_transformer,
    )
    test_dataset = prepare_dataset(
        test_frame,
        covariate_transformer,
    )

    train_loader = make_train_loader(train_dataset)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = EncoderTabM().to(DEVICE)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": ENCODER_LEARNING_RATE,
            },
            {
                "params": model.tabm.parameters(),
                "lr": TABM_LEARNING_RATE,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    print(f"Device: {DEVICE}")
    print(
        "Model parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters()):,}"
    )

    initial_metrics = evaluate(
        model,
        validation_loader,
    )
    best_checkpoint = make_checkpoint(
        model,
        optimizer,
        epoch=-1,
        validation_metrics=initial_metrics,
    )
    best_validation_rmse = initial_metrics["rmse_ratio"]
    remaining_patience = EARLY_STOPPING_PATIENCE

    for epoch in range(N_EPOCHS):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
        )
        improved = (
            validation_metrics["rmse_ratio"]
            < best_validation_rmse
        )

        print(
            f"{'*' if improved else ' '} "
            f"epoch={epoch:03d} "
            f"train_mse={train_loss:.6f} "
            f"val_rmse_ratio="
            f"{validation_metrics['rmse_ratio']:.6f} "
            f"val_rmse_power="
            f"{validation_metrics['rmse_power']:.3f}"
        )

        if improved:
            best_validation_rmse = validation_metrics["rmse_ratio"]
            best_checkpoint = make_checkpoint(
                model,
                optimizer,
                epoch,
                validation_metrics,
            )
            remaining_patience = EARLY_STOPPING_PATIENCE
        else:
            remaining_patience -= 1
            if remaining_patience < 0:
                break

    model.load_state_dict(best_checkpoint["model"])
    target_ratio, prediction_ratio, capacity = predict(
        model,
        test_loader,
    )
    test_metrics = calculate_metrics(
        target_ratio,
        prediction_ratio,
        capacity,
    )

    prediction_frame = test_frame[
        [
            "timestamp",
            "station",
            "source_file",
            "capacity",
        ]
    ].reset_index(drop=True)
    prediction_frame["target_ratio"] = target_ratio
    prediction_frame["prediction_ratio"] = prediction_ratio
    prediction_frame["target_power"] = target_ratio * capacity
    prediction_frame["prediction_power"] = (
        prediction_ratio * capacity
    )
    prediction_frame.to_parquet(
        OUTPUT_DIR / "test_predictions.parquet",
        index=False,
    )

    monthly_scores, monthly_score_mean = calculate_monthly_score(
        prediction_frame
    )
    monthly_scores.to_csv(
        OUTPUT_DIR / "monthly_scores.csv",
        index=False,
    )

    station_metric_rows = []
    for station, group in prediction_frame.groupby(
        "station",
        sort=True,
    ):
        station_metric_rows.append(
            {
                "station": station,
                **calculate_metrics(
                    group["target_ratio"].to_numpy(),
                    group["prediction_ratio"].to_numpy(),
                    group["capacity"].to_numpy(),
                ),
            }
        )
    pd.DataFrame(station_metric_rows).to_csv(
        OUTPUT_DIR / "test_metrics_by_station.csv",
        index=False,
    )

    summary = {
        "best_epoch": best_checkpoint["epoch"],
        "validation": best_checkpoint["validation_metrics"],
        "test": test_metrics,
        "monthly_score_mean": monthly_score_mean,
        "score_capacity": SCORE_CAPACITY,
    }
    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    torch.save(
        {
            "model_state_dict": best_checkpoint["model"],
            "history_length": HISTORY_LENGTH,
            "target_index": TARGET_INDEX,
            "future_covariate_columns": FUTURE_COVARIATE_COLUMNS,
            "capacity_col": CAPACITY_COL,
            "default_capacity": DEFAULT_CAPACITY,
        },
        OUTPUT_DIR / "best_model.pt",
    )
    joblib.dump(
        covariate_transformer,
        OUTPUT_DIR / "future_covariate_transformer.joblib",
    )

    print("\nSummary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Artifacts written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
