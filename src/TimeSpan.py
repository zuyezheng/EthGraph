from enum import Enum


class TimeSpan(Enum):
    """ Temporal grain to process data at. """

    HOUR = (3600, 60)
    DAY = (86400, 3600)

    def __init__(self, seconds, offset_seconds):
        self.seconds = seconds
        self.offset_seconds = offset_seconds
