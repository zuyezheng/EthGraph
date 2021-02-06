import os
from enum import Enum

import numpy

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint


from src.TimeSpan import TimeSpan
from src.TransactionsStatsData import TransactionsStatsData
from src.PricesData import PricesData


class ArimaInputType(Enum):
    PRICE = 0
    ALL = 1

class HistoricalArima:
    """
    Simple historical arima model using a dense neural net with single layer, single latent dim being the classic.

    Input at each time step could be a single price or a combination of aggregate stats computed from all transactions.

    @author zuye.zheng
    """

    def __init__(self, input_type: ArimaInputType, time_span: TimeSpan, steps: int, layers: int = 1, dims: int = 1):
        self.input_type = input_type
        self.time_span = time_span
        self.steps = steps
        self.layers = layers
        self.dims = dims

    def observations(self, prices: PricesData, transactions: TransactionsStatsData):
        num_observations = len(prices.prices_agg.index) - self.steps

        xs = numpy.zeros((num_observations, self.steps, 1))
        ys = numpy.zeros((num_observations, 1))

        for obs_i in range(0, num_observations):
            obs =  prices.prices_agg.iloc[obs_i + self.steps]
            ys[obs_i, 0] = obs['mean_price_change_s']

            for step_i in range(0, self.steps):
                xs[obs_i, step_i, 0] = prices.prices_agg.iloc[obs_i + step_i]['mean_price_change_s']

        return xs[1:], ys[1:]

    def train(self, prices: PricesData, transactions: TransactionsStatsData):
        xs, ys = self.observations(prices, transactions)

        historical = Input(shape=self.steps)

        hidden = historical
        for i in range(0, self.layers):
            hidden = Dense(self.dims)(hidden)

        output = Dense(1)(hidden)

        model = Model(inputs=[historical], outputs=output)
        model.summary()

        model.compile(optimizer='adam', loss='mse', metrics=["mape", "mae"])

        model.fit(
            xs, ys, batch_size=16, epochs=500,
            callbacks=[
                ModelCheckpoint(
                    monitor='loss',
                    mode='min',
                    save_best_only=True,
                    filepath=os.path.join(f'models/arima_new_only_price.hdf5'),
                    verbose=1
                )
            ]
        )

