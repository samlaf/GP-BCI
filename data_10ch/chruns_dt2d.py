from load_matlab import Trains
from gp_full_dt2d import run_ch_stats_exps
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--uid', type=str, default='', help='alphanumerical uid for job number (default: '' will sample randint)')
parser.add_argument('--jobid', type=str, default='', help='sbatch jobid #')
parser.add_argument('--syn', type=int, nargs='+', default=(0,4), help='syn. default=(0,4)')
parser.add_argument('--repeat', type=int, default=25, help='Number of time to repeat loops (default: 25)')
parser.add_argument('--dts', type=int, nargs='+', default=[40,60], help='dts. default=[40,60]')
parser.add_argument('--dtprior', action='store_true')
parser.add_argument('--ntotal', type=int, default=100, help='Total # of query pts to use (default=100)')
parser.add_argument('--nrnd', type=int, nargs='+', default=[25,76,10], help='range of rnd query pts to try (default: [25,76,10]')
parser.add_argument('--sa', action='store_true')
parser.add_argument('--T', type=float, default=0.001, help='temperature for sa. (default=0.001)')
parser.add_argument('--symkern', action='store_true')
parser.add_argument('--multkern', action='store_true')
parser.add_argument('--ardkern', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--constrain', action='store_true')
parser.add_argument('--n_prior_queries', type=int, default=3, help='number of initial queries driven by prior (instead of being random)')
parser.add_argument('--k', type=float, default=2, help='k param in UCB (default=2)')

if __name__ == "__main__":
    args = parser.parse_args()
    print("Starting job with uid = {}".format(args.uid))
    print("syn = {}".format(args.syn))
    print("dts = {}".format(args.dts))
    print("Repeat = {}".format(args.repeat))
    print("Use dtprior: {}".format(args.dtprior))
    print("ntotal: {}".format(args.ntotal))
    print("nrnd: {}".format(args.nrnd))
    print("Use simulated annealing: {}".format(args.sa))
    print("Constrained: {}".format(args.constrain))
    print("k={}".format(args.k))
    trainsC = Trains()
    if args.test:
        # We just want to test the whole setup, so run with minimal
        # configs to end quickly
        D = run_ch_stats_exps(trainsC, syn=args.syn, dts=args.dts, uid=args.uid, jobid=args.jobid, repeat=1, ntotal=50, nrnd=[25,45,10], sa=args.sa, symkern=args.symkern, multkern=args.multkern, ARD=args.ardkern, T=args.T, constrain=args.constrain, n_prior_queries=args.n_prior_queries, k=args.k)
    else:
        # Run the real things
        D = run_ch_stats_exps(trainsC, syn=args.syn, dts=args.dts, uid=args.uid,
                              jobid=args.jobid, repeat=args.repeat, dtprior=args.dtprior,
                              ntotal=args.ntotal, nrnd=args.nrnd, sa=args.sa, T=args.T,
                              symkern=args.symkern, multkern=args.multkern, ARD=args.ardkern,
                              constrain=args.constrain, n_prior_queries=args.n_prior_queries,
                              k=args.k)
