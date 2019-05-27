uid=color
sbatch --job-name color460 chruns2d_script.sh 4 60 $uid "--multkern --ardkern --constrain --ntotal 300 --n_prior_queries 0"
sbatch --job-name color010 chruns2d_script.sh 0 10 $uid "--multkern --ardkern --constrain --ntotal 300 --n_prior_queries 0"

sbatch --job-name color00 chruns2d_script.sh 0 0 $uid "--symkern --multkern --ardkern --constrain --ntotal 300 --n_prior_queries 0"
sbatch --job-name color40 chruns2d_script.sh 4 0 $uid "--symkern --multkern --ardkern --constrain --ntotal 300 --n_prior_queries 0"
