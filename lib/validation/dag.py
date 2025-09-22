import os

import pandas as pd
import pendulum as pdl

borey_home = os.environ['BOREY']

scheduled_date = '{{ data_interval_end }}'

meteostations = pd.read_csv(f'{borey_home}/files/wrf/meteostations.csv', comment='#')
for _, station in meteostations.iterrows():
    get_measurements_op = get_measurements.override(task_id=f'get_measurements_{station.tslist}')(
        station, scheduled_date, mnt_path=f'{borey_home}/mnt'
    )
    last_task >> get_measurements_op
    last_task = send_measurements.override(task_id=f'send_measurements_{station.tslist}')(
        measurements_file=get_measurements_op,
        station=station,
    )

