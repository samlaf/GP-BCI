from load_matlab import Trains
from gp_full_2d import run_ch_stats_exps
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--uid', type=str, default='', help='uid for job number (alphanumerical)')
parser.add_argument('--emg', type=int, default=2, choices=range(7), help='emg. between 0-6')
parser.add_argument('--repeat', type=int, default=25, help='Number of time to repeat loops (default: 25)')
parser.add_argument('--dt', type=int, default=0, choices=(0,10,20,40,60,80,100), help='dt. one of (0,10,20,40,60,80,100)')
parser.add_argument('--dtprior', action='store_true')
parser.add_argument('--ntotal', type=int, default=100, help='Total # of query pts to use (default=100)')
parser.add_argument('--nrnd', type=int, nargs='+', default=[15,76,10], help='range of rnd query pts to try (default: [15,76,10]')

if __name__ == "__main__":
    args = parser.parse_args()
    print("Starting job with uid = {}".format(args.uid))
    print("emg = {}".format(args.emg))
    print("dt = {}".format(args.dt))
    print("Repeat = {}".format(args.repeat))
    print("Use dtprior: {}".format(args.dtprior))
    print("ntotal: {}".format(args.ntotal))
    print("nrnd: {}".format(args.nrnd))
    trainsC = Trains(emg=args.emg)
    queriedchs, maxchs = run_ch_stats_exps(trainsC, emg=args.emg, dt=args.dt, uid=args.uid, repeat=args.repeat, dtprior=args.dtprior, ntotal=args.ntotal, nrnd=args.nrnd)
