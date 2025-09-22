import os
import pickle

import pendulum as pdl


def get_measurements(station, start_date, end_date, mnt_path):
    from correction.helpers.load_stations import get_station_data

    start_date = pdl.parse(start_date)
    end_date = pdl.parse(end_date)
    meteo_period = pdl.interval(start_date, end_date)
    print('loading the station data...')
    measurements_file = f'./stations/borey/raw/station_{station.tslist}_' \
                        f'{start_date.format("YYYY-MM-DD")}_{end_date.format("YYYY-MM-DD")}.pkl'
    if os.path.isfile(measurements_file):
        print(f'File {measurements_file} already exists')
        return

    measurements = {'Station': get_station_data(station, meteo_period),
                    'Coords': (station.lat, station.lon),
                    'Name': station.tslist}
    print('loading the station runs...')
    # measurements.update(get_station_tslists(station, meteo_period, mnt_path))

    print('saving the measurements...')

    with open(measurements_file, 'wb') as f:
        pickle.dump(measurements, f)
    return measurements_file


def send_measurements(measurements_file, station):
    from correction.helpers.load_stations import plot_station_label

    send_measurements.labels = ['TC', 'PSFC', 'WSPD10']

    print('loading the measurements...')
    with open(measurements_file, 'rb') as f:
        measurements = pickle.load(f)
    print('plotting the data...')
    for label in send_measurements.labels:
        buf = plot_station_label(measurements, label, station.title)
    # os.remove(measurements_file)
