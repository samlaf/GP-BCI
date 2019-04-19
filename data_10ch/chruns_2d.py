from load_matlab import Trains
from gp_full_2d import run_ch_stats_exps
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--uid', type=str, default=0, help='uid for job number (alphanumerical)')
parser.add_argument('--emg', type=int, default=2, choices=range(7), help='emg. between 0-6')
parser.add_argument('--dt', type=int, default=0, choices=(0,10,20,40,60,80,100), help='dt. one of (0,10,20,40,60,80,100)')
parser.add_argument('--complicated', action='store_true')

if __name__ == "__main__":
    args = parser.parse_args()
    trainsC = Trains(emg=args.emg)
    trains = trainsC.trains
    queriedchs, maxchs = run_ch_stats_exps(trains, args, complicated=args.complicated)
