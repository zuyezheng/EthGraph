import requests
import argparse
import csv

from datetime import datetime, timedelta

INPUT_DATE_FORMAT = '%Y-%m-%d'
API_DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'

def query(coin_id: int, start: datetime, end: datetime, interval: str):
    start_q = '{:.0f}'.format(start.timestamp())
    end_q = '{:.0f}'.format(end.timestamp())
    url = 'https://web-api.coinmarketcap.com/v1.1/cryptocurrency/quotes/historical?convert=USD&format=chart_crypto_details' \
            + f'&id={coin_id}' \
            + f'&interval={interval}' \
            + f'&time_start={start_q}' \
            + f'&time_end={end_q}'

    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()['data']
        print(f'Retrieved {len(data)} prices between {start} to {end}.')
        return data
    else:
        print(f'Error {resp.status_code} when retrieving history between {start} to {end}.')
        return []


def write_prices(csv_writer, coin_id: int, start: datetime, end: datetime, batch_interval: timedelta, price_interval: str):
    current = start
    while current + batch_interval <= end:
        current_end = current + batch_interval
        data = query(coin_id, current, current_end, price_interval)

        if not ('quotes' in data and len(data['quotes']) == 0):
            for date_key in data:
                try:
                    date = datetime.strptime(date_key, API_DATE_FORMAT)
                    csv_writer.writerow(['{:.0f}'.format(date.timestamp())] + data[date_key]['USD'])
                except ValueError as err:
                    print(f"Error processing price {date_key} between {current} - {current_end}: {err}")

        current = current_end


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract prices to file.')
    parser.add_argument('file_loc', help='Destination csv.')

    # btc = 1, eth = 1027
    parser.add_argument('--token_id', help=f'Token id.', type=int, default=1)

    parser.add_argument('--start', help=f'Start date as {INPUT_DATE_FORMAT}.')
    parser.add_argument('--end', help=f'End date as {INPUT_DATE_FORMAT}.')
    parser.add_argument('--interval', help=f'Interval string.', default='10m')

    args = parser.parse_args()

    with open(args.file_loc, 'w') as csv_file:
        print(f'Writing prices to {args.file_loc}.')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp", "price", "volume", "marketcap"])

        write_prices(
            csv_writer,
            args.token_id,
            datetime.strptime(args.start, INPUT_DATE_FORMAT),
            datetime.strptime(args.end, INPUT_DATE_FORMAT),
            timedelta(days=1),
            args.interval
        )
