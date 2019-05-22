from load_matlab import Trains
from gp_full_2d import run_ch_stats_exps
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--uid', type=str, default='', help='alphanumerical uid for job number (default: '' will sample randint)')
parser.add_argument('--jobid', type=str, default='', help='sbatch jobid. Used to ask info about job.')
parser.add_argument('--emg', type=int, default=4, choices=range(7), help='emg. between 0-6')
parser.add_argument('--repeat', type=int, default=25, help='Number of time to repeat loops (default: 25)')
parser.add_argument('--dt', type=int, default=60, choices=(0,10,20,40,60,80,100), help='dt. one of (0,10,20,40,60,80,100)')
parser.add_argument('--dtprior', action='store_true')
parser.add_argument('--ntotal', type=int, default=100, help='Total # of query pts to use (default=100)')
parser.add_argument('--nrnd', type=int, nargs='+', default=[15,76,10], help='range of rnd query pts to try (default: [15,76,10]')
parser.add_argument('--sa', action='store_true')
parser.add_argument('--T', type=float, default=0.001, help='temperature for sa. (default=0.001)')
parser.add_argument('--symkern', action='store_true')
parser.add_argument('--multkern', action='store_true')
parser.add_argument('--ardkern', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--constrain', action='store_true')

if __name__ == "__main__":
    args = parser.parse_args()
    print("Starting job with uid = {}".format(args.uid))
    print("emg = {}".format(args.emg))
    print("dt = {}".format(args.dt))
    print("Repeat = {}".format(args.repeat))
    print("Use dtprior: {}".format(args.dtprior))
    print("ntotal: {}".format(args.ntotal))
    print("nrnd: {}".format(args.nrnd))
    print("Use simulated annealing: {}".format(args.sa))
    trainsC = Trains(emg=args.emg)
    if args.test:
        # We just want to test the whole setup, so run with minimal
        # configs to end quickly
        D = run_ch_stats_exps(trainsC, emg=args.emg, dt=args.dt, uid=args.uid, repeat=1, ntotal=50, nrnd=[15,35,10], sa=args.sa, symkern=args.symkern, multkern=args.multkern, ARD=args.ardkern, T=args.T)
    else:
        # Run the real things
        D = run_ch_stats_exps(trainsC, emg=args.emg, dt=args.dt, uid=args.uid,
                              repeat=args.repeat, dtprior=args.dtprior, ntotal=args.ntotal,
                              nrnd=args.nrnd, sa=args.sa, symkern=args.symkern,
                              multkern=args.multkern, ARD=args.ardkern, T=args.T,
                              constrain=args.constrain)
