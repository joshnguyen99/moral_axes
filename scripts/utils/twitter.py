"""
Twitter helpers.
Source: https://github.com/avalanchesiqi/twitter-sampling/blob/master/utils/helper.py
"""

import datetime
# twitter's snowflake parameters
twepoch = 1288834974657
datacenter_id_bits = 5
worker_id_bits = 5
sequence_id_bits = 12
max_datacenter_id = 1 << datacenter_id_bits
max_worker_id = 1 << worker_id_bits
max_sequence_id = 1 << sequence_id_bits
max_timestamp = 1 << (64 - datacenter_id_bits - worker_id_bits - sequence_id_bits)


def make_snowflake(timestamp_ms, datacenter_id, worker_id, sequence_id, twepoch=twepoch):
    """generate a twitter-snowflake id, based on
    https://github.com/twitter/snowflake/blob/master/src/main/scala/com/twitter/service/snowflake/IdWorker.scala
    :param: timestamp_ms time since UNIX epoch in milliseconds"""
    timestamp_ms = int(timestamp_ms)
    sid = ((timestamp_ms - twepoch) % max_timestamp) << datacenter_id_bits << worker_id_bits << sequence_id_bits
    sid += (datacenter_id % max_datacenter_id) << worker_id_bits << sequence_id_bits
    sid += (worker_id % max_worker_id) << sequence_id_bits
    sid += sequence_id % max_sequence_id
    return sid


def melt_snowflake(snowflake_id, twepoch=twepoch):
    """inversely transform a snowflake id back to its components."""
    snowflake_id = int(snowflake_id)
    sequence_id = snowflake_id & (max_sequence_id - 1)
    worker_id = (snowflake_id >> sequence_id_bits) & (max_worker_id - 1)
    datacenter_id = (snowflake_id >> sequence_id_bits >> worker_id_bits) & (max_datacenter_id - 1)
    timestamp_ms = snowflake_id >> sequence_id_bits >> worker_id_bits >> datacenter_id_bits
    timestamp_ms += twepoch
    return timestamp_ms, datacenter_id, worker_id, sequence_id


def tweetid2time(tweet_id):
    time = datetime.datetime.fromtimestamp(melt_snowflake(tweet_id)[0]/1000.0)
    return (time.year, time.month, time.day)
