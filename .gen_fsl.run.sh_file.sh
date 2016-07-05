# vim: filetype=sh

# DEFAULTS
walltime='12:00:00'
gpus=1
cpus=6
mem=512
jobname='genscript-example'
commands='--bypass_sql --test'

# Parse command line parameters
for i in "$@"
do
  case $i in
    -wt=*|--walltime=*)
        walltime="${i#*=}";  shift ;; # TODO format checking...
    --nCPU=*|--cpus=*)
        cpus="${i#*=}"
        if ! [[ $cpus =~ ^[0-9]+$ ]] ; then
          echo "error: cpus is not a number, try again" >&2; exit 2
        fi
        shift ;;
    --nGPU=*|--gpus=*)
        gpus="${i#*=}"
        if ! [[ $gpus =~ ^[0-9]+$ ]] ; then
          echo "error: gpus is not a number, try again" >&2; exit 2
        fi
        shift ;;
    -mem=*|--mem-per-cpu=*)
        mem="${i#*=}"
        if ! [[ $mem =~ ^[0-9]+$ ]] ; then
          echo "error: mem is not a number (in MB), try again" >&2; exit 2
        fi
	shift ;;
    -n=*|--jobname=*)
        jobname="${i#*=}";  shift ;;

    -c=*|--runCommand=*)
        commands="${i#*=}";  shift ;;
  esac
done


# BUILD THE FLAGS FOR `PYTHON MAIN.PY {FLAGS}`
M="M" # I didn't want to spend time looking...
mem=$mem$M

SBATCH___GRES_OPT="#SBATCH --gres=gpu:$gpus"
if [ $gpus -eq 0 ]; then
  SBATCH___GRES_OPT="# Not going to use a gpu"
fi

###############################################################################

echo "#!/bin/bash"

echo "#SBATCH --time=$walltime"     # walltime
echo "#SBATCH --ntasks=$cpus"       # number of processor cores (i.e. tasks)
echo "#SBATCH --nodes=1"            # number of nodes
echo $SBATCH___GRES_OPT             # gpu reqs (empty if gpus=0)
echo "#SBATCH --mem-per-cpu=$mem"   # memory per CPU core # 6G ram
echo "#SBATCH -J '$jobname'"        # job name
echo "#SBATCH --gid=fslg_pccl"      # give group access
echo ""
echo "export PBS_NODEFILE=\`/fslapps/fslutils/generate_pbs_nodefile\`"
echo "export PBS_JOBID=\$SLURM_JOB_ID"
echo "export PBS_O_WORKDIR='\$SLURM_SUBMIT_DIR'"
echo "export PBS_QUEUE=batch"
echo ""
echo "module purge"
echo "module load cuda/7.5.18"
echo "module load cudnn/6.5"
echo "module load tensorflow/0/8" # TODO fix this once we have modules installed
echo ""
echo $commands

