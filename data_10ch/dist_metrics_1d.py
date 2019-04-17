from gp_full_1d import run_dist_exps

parser = argparse.ArgumentParser()
parser.add_argument('--uid', type=int, default=0, help='uid for job number')
parser.add_argument('--emg', type=int, default=2, choices=range(7), help='emg. between 0-6')

if __name__ == "__main__":
    args = parser.parse_args()
    run_dist_exps(args)
