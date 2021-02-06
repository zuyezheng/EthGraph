import requests
import argparse
import csv
import time
import math
import numpy

from collections import deque


class ExceptionResponse:

    def __init__(self):
        self.status_code = -1
        self.text = ''


def track(
    url: str,
    interval: int,
    max_duration: float,
    destination: str,
    keep_body: bool,
    window: int = 60
):
    start_time = time.time()

    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
    }

    with open(destination, 'a', 1) as csv_file:
        csv_writer = csv.writer(csv_file)

        past_durations = deque()

        adaptive_interval = interval

        while True:
            ping_start_time = time.time()
            try:
                response = requests.get(url, headers=headers)
            except:
                # in case server is dead or ping timed out
                response = ExceptionResponse()
            ping_duration = time.time() - ping_start_time

            response_body = ''
            if keep_body:
                try:
                    response_body = response.text
                except:
                    print('Missing expected body.')

            row = [ping_start_time, ping_duration, response.status_code, response_body]
            print(row)
            csv_writer.writerow(row)

            # adapt the interval based on how the duration compares to the window of durations
            if len(past_durations) > 10:
                std_duration = numpy.std(past_durations)
                mean_duration = numpy.mean(past_durations)
                if (ping_duration - mean_duration) > std_duration:
                    adaptive_interval = max(interval / ((ping_duration - mean_duration) / std_duration), 10)
                    print(f'Using adaptive interval of {adaptive_interval}, std {std_duration}, mean {mean_duration}.')

            # accumulate some aggregate stats to adjust intervals as necessary
            past_durations.append(ping_duration)
            if len(past_durations) > window:
                past_durations.popleft()

            current_time = time.time()
            if current_time - start_time > max_duration:
                # stop tracking if we hit a max
                break
            elif current_time - ping_start_time < adaptive_interval:
                # wait if we need to
                time.sleep(adaptive_interval - (current_time - ping_start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Track ping times.')
    parser.add_argument('url', help='What to ping.')
    parser.add_argument('--interval', type=float, help='Min interval between pings.')
    parser.add_argument('--max_duration', type=float, help='Stop once max duration is hit, defaults to infinity.', default=math.inf)
    parser.add_argument('--destination', help='Destination to write results.')
    parser.add_argument('--keep_body', type=bool, help='Keep the body of the response.')

    args = parser.parse_args()

    track(args.url, args.interval, args.max_duration, args.destination, args.keep_body)

