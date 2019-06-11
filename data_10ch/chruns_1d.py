from load_matlab import Trains
from gp_full_1d import run_ch_stats_exps
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--uid', type=int, default=0, help='uid for job number')
parser.add_argument('--k', type=int, default=2, help='k param for ucb. (default=2)')
parser.add_argument('--ARD', default=True, action='store_true')
parser.add_argument('--repeat', default=25, help='Number of repeats to calculate run stats (default=25)')
parser.add_argument('--ntotal', type=int, default=150, help='ntotal pts. (default=150)')
parser.add_argument('--nrnd', default=[5,75,10])
parser.add_argument('--jobid', type=str, default='', help='sbatch jobid. Used to ask info about job.')

if __name__ == "__main__":
    args = parser.parse_args()
    print("Starting job with uid = {}".format(args.uid))
    print("Repeat = {}".format(args.repeat))
    print("ntotal: {}".format(args.ntotal))
    print("nrnd: {}".format(args.nrnd))
    print("k={}".format(args.k))
    trainsC = Trains(emg=args.emg)
    D = run_ch_stats_exps(trainsC, emgs=[0,4], repeat=args.repeat, uid=args.uid, jobid=args.jobid, continue_opt=True, k=args.k, ntotal=args.ntotal, nrnd=args.nrnd, ARD=args.ARD)
