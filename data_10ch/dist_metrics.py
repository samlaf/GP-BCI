from gp_full_2d import run_dist_exps
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--uid', type=int, default=0, help='uid for job number')
parser.add_argument('--dt', type=int, default=0, choices=(0,10,20,40,60,80,100), help='dt. one of (0,10,20,40,60,80,100)')
parser.add_argument('--emg', type=int, default=2, choices=range(7), help='emg. between 0-6')

if __name__ == "__main__":
    args = parser.parse_args()
    run_dist_exps(args)
