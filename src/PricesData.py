import math

import numpy
import pandas
from pandas import DataFrame

from src.TimeSpan import TimeSpan


class PricesData:
    """
    Load prices extracted by CryptoPrices.py with aggregation and processing such as standardization.

    @author zuye.zheng
    """

    def __init__(
        self,
        # location of the csv with prices
        loc: str,
        # timespan to aggregate by
        span: TimeSpan
    ):
        self.span = span
        self.prices = pandas.read_csv(loc)

        # compute the aggregates
        self.prices_agg = PricesData.aggregate_by_time(self.span, self.prices)
        
        # index and sort the prices so they can be searched through efficiently by timestamp
        self.prices = self.prices.set_index('timestamp')
        self.prices = self.prices[~self.prices.index.duplicated(keep='first')].sort_index()

        # standardize the prices
        price_mean = self.prices['price'].mean()
        price_std = self.prices['price'].std()
        self.prices['price_s'] = (self.prices['price'] - price_mean) / price_std

    def price(self, timestamp: int, standardized: bool = False):
        """ Return the last price of or before the timestamp. """
        # right will always return the first value greater than the timestamp so we can always go back 1 to find a price
        # without looking into the future (less or equal to the timestamp),
        # left will not as it will give us the exact index or the first value greater when there is no exact match
        field = 'price_s' if standardized else 'price'
        return self.prices.iloc[numpy.searchsorted(self.prices.index, timestamp, side='right') - 1][field]

    def volume(self, timestamp: int):
        """ Return the last volume of or before the timestamp. """
        return self.prices.iloc[numpy.searchsorted(self.prices.index, timestamp, side='right') - 1]['volume']

    def aggs(self, time_step: int):
        """ Return the aggregate series for the given time step which should be in the unit of TimeSpan since epoch. """
        return self.prices_agg.iloc[numpy.searchsorted(self.prices_agg.index, time_step, side='right') - 1]

    def as_data_frame(self, timespan: int):
        """ Return the prices as a pandas dataframe that backfills any missing timesteps. """
        prices_by_step = self.prices.reset_index()

        # compute the step by taking the floor of the timestamp of span
        prices_by_step['time_step'] = (prices_by_step['timestamp']/timespan).apply(numpy.floor)
        prices_by_step = prices_by_step.set_index('time_step')

        # remove any duplicates
        prices_by_step = prices_by_step[~prices_by_step.index.duplicated()]

        # generate a full index with any possibly missed steps
        prices_index = pandas.Index(
            numpy.arange(prices_by_step.iloc[0].name, prices_by_step.iloc[-1].name + 1, 1),
            name='time_step'
        )

        # reindex and backfill any prices
        prices_by_step = prices_by_step.reindex(prices_index, method='backfill')

        return prices_by_step

    @staticmethod
    def load_prices(loc: str):
        # load prices and index by timestamp
        prices = pandas.read_csv(loc).set_index('timestamp')

        # remove any duplicate prices
        prices = prices[~prices.index.duplicated(keep='first')]

        # return sorted
        return prices.sort_index()

    @staticmethod
    def aggregate_by_time(span: TimeSpan, prices: DataFrame):
        # create a copy of prices to compute aggregates
        prices_agg = prices.copy()

        # compute the time step desired and aggregate by it
        prices_agg['time_step'] = (prices_agg['timestamp'] / span.seconds)\
            .apply(lambda d: math.floor(d))

        aggs = dict()
        agg_columns = []
        for column in ['price', 'volume']:
            for agg in ['last', 'mean', 'std']:
                agg_column = agg + '_' + column
                agg_columns.append(agg_column)
                aggs[agg_column] = pandas.NamedAgg(column=column, aggfunc=agg)
        prices_agg = prices_agg.groupby('time_step').agg(**aggs)

        # compute the aggregate changes for last and mean from the last time span
        agg_previous = prices_agg.shift(1)

        for agg in agg_columns:
            col_name = agg + '_change'
            prices_agg[col_name] = (prices_agg[agg] - agg_previous[agg]) / agg_previous[agg]

            # standardize the value for gradient descent
            agg_mean = prices_agg[col_name].mean()
            agg_std = prices_agg[col_name].std()

            prices_agg[col_name + '_s'] = prices_agg.apply(
                lambda r: (r[col_name] - agg_mean) / agg_std, axis=1
            )

        # return the aggregates sorted
        return prices_agg.sort_index()
