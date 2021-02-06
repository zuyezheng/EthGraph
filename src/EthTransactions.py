from enum import Enum
from typing import Callable

import math
import re

from pyspark.sql import SparkSession, DataFrame, DataFrameWriter, Window
import pyspark.sql.functions as F
import pyspark.sql.types as Types

from src.PricesData import PricesData
from src.TimeSpan import TimeSpan


SIGS = math.pow(10, 18)
# we have prices data at 5 min intervals
PRICES_TIME_SPAN = 300

class DataFrameFormat(Enum):
    CSV = lambda writer, path: writer.csv(path),
    PARQUET = lambda writer, path: writer.option('compression', 'snappy').parquet(path),

    def __init__(self, _write: Callable[[DataFrameWriter, str], None], unused=None):
        self._write = _write

    def write(self, df: DataFrame, path: str):
        self._write(df.write, path)


class EthTransactions:
    """
    Process transactions from ethereum-etl using spark.

    @author zuye.zheng
    """

    METRICS = [
        'value',
        'gas', 'gas_price', 'gas_used', 'gas_max_wei', 'gas_used_wei',
        'value_usd', 'gas_max_usd', 'gas_used_usd'
    ]

    def __init__(self, spark_session: SparkSession, transactions_path: str, receipts_path: str, prices: PricesData):
        # read the transactions
        self.transactions = spark_session.read.csv(transactions_path, header=True)

        # join it with receipts so we know the true gas used
        receipts = spark_session.read.csv(receipts_path, header=True)
        receipts = receipts.select('transaction_hash', 'gas_used')

        self.transactions = self.transactions.join(
            receipts, self.transactions.hash == receipts.transaction_hash, how='left'
        )

        # start with some type conversions
        self.transactions = self.transactions\
            .withColumn('value', F.col('value').cast(Types.DoubleType()))\
            .withColumn('gas', F.col('gas').cast(Types.DoubleType()))\
            .withColumn('gas_price', F.col('gas_price').cast(Types.DoubleType()))\
            .withColumn('gas_used', F.col('gas_used').cast(Types.DoubleType()))\
            .withColumn('gas_max_wei', F.col('gas') * F.col('gas_price'))\
            .withColumn('gas_used_wei', F.col('gas_used') * F.col('gas_price'))

        # compute a time_step for every 15 minutes to join to prices data
        self.transactions = self.transactions.withColumn(
            'prices_time_step',
            # compute the floored time span index
            F.floor(F.col('block_timestamp') / PRICES_TIME_SPAN)
        )

        # drop some columns
        self.transactions = self.transactions.drop(
            'transaction_hash',
            'nonce', 'transaction_index', 'input'
        )

        # join with the prices data frame to compute values by usd
        prices_df = spark_session.createDataFrame(prices.as_data_frame(PRICES_TIME_SPAN).reset_index(drop=False))
        prices_df = prices_df.drop('timestamp', 'volume', 'marketcap', 'price_s')
        self.transactions = self.transactions.join(
            prices_df, self.transactions.prices_time_step == prices_df.time_step, how='left'
        )

        # compute the prices in usd
        self.transactions = self.transactions\
            .withColumn('value_usd', F.col('value') * F.col('price') / SIGS)\
            .withColumn('gas_max_usd', F.col('gas_max_wei') * F.col('price') / SIGS) \
            .withColumn('gas_used_usd', F.col('gas_used_wei') * F.col('price') / SIGS)

        # create a temp view and cache it for the later operations
        temp_view_name = re.sub(r'[/.]', '_', transactions_path)
        self.transactions.createOrReplaceTempView(temp_view_name)
        spark_session.table(temp_view_name).cache()

        self.transactions = spark_session.table(temp_view_name)

    def top_transactions(self,
        top_n: int,
        span: TimeSpan,
        path: str,
        df_format: DataFrameFormat,
        # optional span offset, e.g. 1 when TimeSpan.DAY, would mean to offsset the buckets by 1 hour before aggregation
        span_offset: int = 0
    ):
        """ Find the top N transactions per time span. """

        # add a field for day to later aggregate by
        transactions = self.transactions.withColumn(
            'time_step',
            # compute the floored time span index, optionally offset it which can be used to produce more observastions
            F.floor((F.col('block_timestamp') - (span_offset * span.offset_seconds)) / span.seconds)
        )

        # use a window function to find the top per window
        window_spec = Window \
            .partitionBy('time_step') \
            .orderBy(F.col('value').desc())

        top_transactions = transactions.withColumn("rank", F.rank().over(window_spec)) \
            .filter(F.col('rank') <= top_n)

        # write to file
        df_format.write(top_transactions, path)

    def stats(self, span: TimeSpan, path: str, df_format: DataFrameFormat):
        """ Aggregate stats by a given time span. """

        # create a new dataframe with a column for the timestamp divided by the time span
        agg_transactions = self.transactions.withColumn(
            'time_step',
            F.floor(F.col('block_timestamp') / span.seconds)
        )

        agg_fields = [
            F.count(F.lit(1)).alias('num_transactions'),
            F.countDistinct('block_hash').alias('num_blocks'),
            F.countDistinct('from_address').alias('num_froms'),
            F.countDistinct('to_address').alias('num_tos')
        ]

        for field in EthTransactions.METRICS:
            agg_fields.extend([
                F.sum(field).alias('sum_' + field),
                F.stddev(field).alias('stdev_' + field),
                F.skewness(field).alias('skewness_' + field),
                F.kurtosis(field).alias('kurtosis_' + field),
            ])

        daily_agg = agg_transactions.groupBy('time_step').agg(*agg_fields)

        df_format.write(daily_agg, path)
