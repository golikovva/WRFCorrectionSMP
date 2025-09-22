import pandas as pd
import sys
sys.path.insert(0,'../WRFCorrection')
from correction.config.config import cfg
from correction.validation.tasks import (
    get_measurements
)
borey_home = cfg.GLOBAL.BASE_DIR


start_date = '2015-01-01T00:00:00'
end_date = '2025-01-01T00:00:00'

meteostations = pd.read_csv(f'{borey_home}/metadata/meteostations_borey.csv', comment='#')
for _, station in meteostations.iterrows():
    get_measurements_op = get_measurements(
        station, start_date, end_date, mnt_path=f'{borey_home}/mnt'
    )
