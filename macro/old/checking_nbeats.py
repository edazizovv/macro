
#
import pandas
from statsmodels.tsa.seasonal import MSTL
from matplotlib import pyplot

#
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import pandas as pd
import torch

from pytorch_forecasting import Baseline, NBeats, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import SMAPE

#


# driver = 'JTU5300QUL'
# driver = 'QUSR628BIS'
# driver = 'RAILFRTCARLOADSD11'
# driver = 'FRGSHPUSM649NCIS'
# driver = 'TSIFRGHT'
# driver = 'AIRRTMFMD11'
driver = 'CUUR0000SEHA'
series = pandas.read_csv('./data/{0}.csv'.format(driver))
# series = data[[driver]].copy()
# series = series.reset_index().rename(columns={'jx': 'ds', driver: 'y'})
series = series.rename(columns={'DATE': 'ds', driver: 'y'})
series['ds'] = pandas.to_datetime(series['ds'])
series['y'] = series['y'].apply(func=lambda x: pandas.NA if x == '.' else x)
series['y'] = series['y'].ffill()
series['y'] = pandas.to_numeric(series['y'])
series = series[series['ds'] > '1970-01-01']
# series['y'] = series['y'].pct_change()
# series = series[series['ds'] > '2000-01-01']
series


# create dataset and dataloaders
max_encoder_length = 60
max_prediction_length = 5

# training_cutoff = (series.shape[0] - 1) - max_prediction_length
training_cutoff = int(series.shape[0] / 2)
series['time_idx'] = list(range(series.shape[0]))
series['group'] = 0

context_length = max_encoder_length
prediction_length = max_prediction_length

training = TimeSeriesDataSet(
    series.iloc[:training_cutoff, :],
    time_idx="time_idx",
    target="y",
    # categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
    group_ids=["group"],
    # only unknown variable is "value" - and N-Beats can also not take any additional variables
    time_varying_unknown_reals=["y"],
    max_encoder_length=context_length,
    max_prediction_length=prediction_length,
)

validation = TimeSeriesDataSet.from_dataset(training, series, min_prediction_idx=training_cutoff)
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
trainer = pl.Trainer(
    max_epochs=500,
    accelerator="auto",
    enable_model_summary=True,
    gradient_clip_val=0.01,
    callbacks=[early_stop_callback],
    limit_train_batches=150,
    # logger=tensorboard
)


net = NBeats.from_dataset(
    training,
    learning_rate=1e-3,
    log_interval=10,
    log_val_interval=1,
    weight_decay=1e-2,
    stack_types = ['trend', 'seasonality'],
    widths=[32, 512],
    backcast_loss_ratio=1.0,
    # logger=tensorboard
)

trainer.fit(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    # logger=tensorboard
)

best_model_path = trainer.checkpoint_callback.best_model_path
best_model = NBeats.load_from_checkpoint(best_model_path)

actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_model.predict(val_dataloader)
(actuals - predictions).abs().mean()

raw_predictions = best_model.predict(val_dataloader, mode="raw", return_x=True, return_y=False)

best_model.plot_prediction(raw_predictions.x, raw_predictions.output, idx=-100, add_loss_to_title=True)

best_model.plot_interpretation(raw_predictions.x, raw_predictions.output, idx=100)

raw_predictions
