from enum import Enum
from typing import Callable

from pyspark.sql import SparkSession, DataFrame, DataFrameWriter, Window
import pyspark.sql.functions as F
import pyspark.sql.types as Types

from src.TimeSpan import TimeSpan


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

    def __init__(self, spark_session: SparkSession, transactions_path: str):
        self.spark_session = spark_session

        # read the transactions, drop some less interesting columns to save some memory
        self.transactions = spark_session.read.csv(transactions_path, header=True) \
            .drop('nonce', 'transaction_index', 'input')

        self.transactions = self.transactions \
            .withColumn('value_d', F.col('value').cast(Types.DoubleType())) \
            .withColumn('gas_d', F.col('gas').cast(Types.DoubleType()))

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
            'time_span',
            # compute the floored time span index, optionally offset it which can be used to produce more observastions
            F.floor((F.col('block_timestamp') - (span_offset * span.offset_seconds)) / span.seconds)
        )

        # use a window function to find the top per window
        window_spec = Window \
            .partitionBy('time_span') \
            .orderBy(F.col('value_d').desc())

        top_transactions = transactions.withColumn("rank", F.rank().over(window_spec)) \
            .filter(F.col('rank') <= top_n)

        # write to file
        df_format.write(top_transactions, path)

    def stats(self, span: TimeSpan, path: str, df_format: DataFrameFormat):
        """ Aggregate stats by a given time span. """

        # create a new dataframe with a column for the timestamp divided by the time span
        agg_transactions = self.transactions.withColumn(
            'time_span',
            F.floor(F.col('block_timestamp') / span.seconds)
        )

        daily_agg = agg_transactions.groupBy('time_span').agg(
            F.sum('value_d').alias('sum_value'),
            F.stddev('value_d').alias('stdev_value'),
            F.skewness('value_d').alias('skewness_value'),
            F.kurtosis('value_d').alias('kurtosis_value'),

            F.sum('gas_d').alias('sum_gas'),
            F.stddev('gas_d').alias('stdev_gas'),
            F.skewness('gas_d').alias('skewness_gas'),
            F.kurtosis('gas_d').alias('kurtosis_gas'),

            F.countDistinct('block_hash').alias('num_blocks'),
            F.countDistinct('from_address').alias('num_froms'),
            F.countDistinct('to_address').alias('num_tos'),

            F.count(F.lit(1)).alias('num_transactions')
        )

        df_format.write(daily_agg, path)
