# vim: filetype=sh

function help {
    echo "Usage: jedi.sh    {run_all | run_local | dashboard | use_fsl} [options]"
    echo "               [--fsl_sbatch_arguments]  # Passed into the sbatch options (gpu, mem, nodes, name, etc.) when using FSL"
    echo "               [--fsl_python_arguments]  # Passed into the `python main [args] --rom="blah"` call when using FSL"
    echo "               [--fsl_username]          # username used when ssh-ing into the FSL"
}


function gen_fsl_sbatch_script {
    # DEFAULTS
    walltime='48:00:00'
    nodes=1
    cpus=10
    gpus=2
    mem=12
    mem_unit='G' 
    jobname='genscript-example'
    
    # OPTIONS:
    function sbatch_help {
        echo "[-wt|--walltime]         Default: '48:00:00'"
        echo "[-n|--nodes|--nNodes]    Default: 1"
        echo "[--nCPU|--cpus]          Default: 2"
        echo "[--nGPU|--gpus]          Default: 10"
        echo "[--mem|--total_memory]   Default: 12"
        echo "[--mem_unit]             Default: G (M is also allowed)"
        echo "[-job|--jobname]         Default: 'genscript-example'"
        echo ""
        echo "Example: --gpus=2 -job=testing -wt=01:00:00 -n=1 --mem=24 --mem_unit=G"
    }
        
    # Parse command line parameters
    for i in "$@"
    do
      case $i in
        -wt=*|--walltime=*)
            walltime="${i#*=}";  shift ;; # TODO format checking...
        
        -n=*|--nodes=*|--nNodes=*) nodes="${i#*=}";  shift ;; 
        
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
        
        --mem=*|--total_memory=*) # NOTE: Memory is total memory for that setup, in GBs!
            mem="${i#*=}"
            if ! [[ $mem =~ ^[0-9]+$ ]] ; then
              echo "error: mem is not a number (in GB), try again" >&2; exit 2
            fi
    	    shift ;;
            
        --mem_unit=G)  mem_unit="G";  shift ;;
        --mem_unit=M)  mem_unit="M";  shift ;;
        
        -n=*|--jobname=*) jobname=${i#*=}; shift ;;
            
        *) "Gen SBATCH Function: Error: unrecognized flags."; sbatch_help; echo "BAD_VALUE_WAS: >>>$i<<<"; exit 6 ;;
      esac
    done
    
    SBATCH___GRES_OPT="#SBATCH --gres=gpu:$gpus"
    if [ $gpus -eq 0 ]; then
      SBATCH___GRES_OPT="# Not going to use a gpu"
    fi
    
    ###############################################################################
    echo "#!/bin/bash"                 >> sbatch.tmp
    echo ""                            >> sbatch.tmp
    echo "#SBATCH --time=$walltime"    >> sbatch.tmp  # walltime
    echo "#SBATCH --ntasks=$cpus"      >> sbatch.tmp  # number of processor cores (i.e. tasks)
    echo "#SBATCH --nodes=$nodes"      >> sbatch.tmp  # number of nodes
    echo $SBATCH___GRES_OPT            >> sbatch.tmp  # gpu reqs (empty if gpus=0)
    echo "#SBATCH --qos=dw87"          >> sbatch.tmp  # Runs us on the m8g2 nodes
    echo "#SBATCH --mem=$mem$mem_unit" >> sbatch.tmp  # total memory requested (Note this is different than --mem-per-cpu option. # G or M available.
    echo "#SBATCH -J '$jobname'"       >> sbatch.tmp  # job name
    echo "#SBATCH --gid=fslg_pccl"     >> sbatch.tmp  # give group access
    echo ""                            >> sbatch.tmp
    echo "source $HOME/fsl_groups/fslg_pccl/configs/group_bashrc" >> sbatch.tmp # Modules, etc.
    echo "module list"  >> sbatch.tmp # Debugging
    echo "hostname"     >> sbatch.tmp # Debugging
    echo "pwd"          >> sbatch.tmp # Debugging
}

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

#Defaults for named params:
COMMAND=run_local
FSL_SBATCH_ARGS="--nodes=1"
FSL_PY_ARGS="--bypass_sql --threads=12"
FSL_USERNAME="jacobj66"

###############################################################################

# Parse command line parameters
for i in "$@"
do
  case $i in
    dashboard) echo "db"; COMMAND="dashboard"; shift ;;

    run_all)   echo "al"; COMMAND="run_all"; shift ;;
    run_local) echo "rl"; COMMAND="run_local"; shift ;;

    --fsl_sbatch_arguments=*) FSL_SBATCH_ARGS="${i#*=}";  shift ;;
    --fsl_python_arguments=*) FSL_PY_ARGS="${i#*=}";  shift ;;
    --fsl_username=*) FSL_USERNAME="${i#*=}";  shift ;;
    run_fsl)   echo "fs"; COMMAND="run_fsl"; shift ;;
    run_fsl_local)   echo "fs"; COMMAND="run_fsl_local"; shift ;;
    fsl_copy) 
        echo "Starting copy";
        # Copy
        scp /mnt/pccfs/projects/jedi/*  jacobj66@ssh.fsl.byu.edu:/fslhome/jacobj66/fsl_groups/fslg_pccl/projects/jedi
        exit 0; ;;
    *) echo "Error: unrecognized flags."; help; exit 2
  esac
done

if [ "$#" -lt 1 ]; then
  echo "Using defaults!  I hope that's ok with you."
fi

echo "FSL_SBATCH_ARGS was >>>$FSL_SBATCH_ARGS<<<"
echo "FSL_PY_ARGS was     >>>$FSL_PY_ARGS<<<"
source $HOME/fsl_groups/fslg_pccl/configs/group_bashrc
echo "NOW USING $COMMAND"

###############################################################################

if [ "$COMMAND" = dashboard ]; then
    echo "TOO FAR!!!"
    exit 13
    export JEDI_REMOTE_COMMAND="cd /mnt/pccfs/projects/jedi ; bash --rcfile /mnt/pccfs/downloads/term_customization.bashrc -c \"$RUN_COMMAND\" ; bash --rcfile /mnt/pccfs/downloads/term_customization.bashrc"
    screen -c ./.screenrc

###############################################################################

elif [ "$COMMAND" = run_all ]; then

    export JEDI_REMOTE_COMMAND="cd /mnt/pccfs/projects/jedi ; ./jedi.sh run_local ; bash"
    screen -c ./.screenrc


elif [ "$COMMAND" = run_local ]; then

    echo "$(tput setaf 1)#####"
    echo "$HOSTNAME"
    echo "#####$(tput sgr0)"

    sleep 3

    if [ "$HOSTNAME" = reaper ]; then
        python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=baseline --rom=Breakout

        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --agent_type=qexplorer --rom=Pong &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --agent_type=qexplorer --rom=Breakout &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --agent_type=qexplorer --rom=SpaceInvaders  &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --agent_type=qexplorer --rom=Seaquest &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --agent_type=qexplorer --rom=Zaxxon &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --agent_type=qexplorer --rom=Freeway &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --agent_type=qexplorer --rom=Assault &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=maximummargin &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --batch_size=36 --dqn_type=optimistic

    elif [ "$HOSTNAME" = santaka ]; then
        python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=baselinedouble --rom=Breakout

        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --agent_type=qexplorer --negative_reward_on_death --rom=Pong &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --agent_type=qexplorer --negative_reward_on_death --rom=Breakout &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --agent_type=qexplorer --negative_reward_on_death --rom=SpaceInvaders &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --agent_type=qexplorer --negative_reward_on_death --rom=Seaquest &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --agent_type=qexplorer --negative_reward_on_death --rom=Zaxxon &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --agent_type=qexplorer --negative_reward_on_death --rom=Freeway &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --agent_type=qexplorer --negative_reward_on_death --rom=Assault

    elif [ "$HOSTNAME" = ghost ]; then
        python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=baselinedoubleduel --rom=Breakout

        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=constrained --agent_type=lookahead --lookahead=5 --rom=Pong &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=constrained --agent_type=lookahead --lookahead=5 --rom=Breakout &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=constrained --agent_type=lookahead --lookahead=5 --rom=SpaceInvaders &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=constrained --agent_type=lookahead --lookahead=5 --rom=Seaquest &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=constrained --agent_type=lookahead --lookahead=5 --rom=Zaxxon &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=constrained --agent_type=lookahead --lookahead=5 --rom=Freeway &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=constrained --agent_type=lookahead --lookahead=5 --rom=Assault

    elif [ "$HOSTNAME" = naga-All-Series ]; then
        python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=baselineduel --rom=Breakout

        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --rom=Pong &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --rom=Breakout &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --rom=SpaceInvaders &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --rom=Seaquest &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --rom=Zaxxon &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --rom=Freeway &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --rom=Assault

    elif [ "$HOSTNAME" = morita.cs.byu.edu ]; then
        echo "Noop"

        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=baselinedoubleduel --rom=Pong &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=baselinedoubleduel --rom=Breakout &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=baselinedoubleduel --rom=SpaceInvaders &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=baselinedoubleduel --rom=Seaquest &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=baselinedoubleduel --rom=Zaxxon &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=baselinedoubleduel --rom=Freeway &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=baselinedoubleduel --rom=Assault

    elif [ "$HOSTNAME" = infinity ]; then
        echo "Noop"

        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=baselineduel --rom=Pong &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=baselineduel --rom=Breakout &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=baselineduel --rom=SpaceInvaders &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=baselineduel --rom=Seaquest &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=baselineduel --rom=Zaxxon &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=baselineduel --rom=Freeway &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --network_type=baselineduel --rom=Assault

    elif [ "$HOSTNAME" = hatch ]; then
        echo "Noop"

        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --negative_reward_on_death --rom=Pong &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --negative_reward_on_death --rom=Breakout &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --negative_reward_on_death --rom=SpaceInvaders &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --negative_reward_on_death --rom=Seaquest &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --negative_reward_on_death --rom=Zaxxon &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --negative_reward_on_death --rom=Freeway &&
        # python main.py --total_ticks=$MAX_TICKS --name=$NAME --negative_reward_on_death --rom=Assault

    elif [ "$HOSTNAME" = doctor ]; then
       echo "Noop"


    elif [ "$HOSTNAME" = potts ]; then
       echo "Noop"

    fi
    

###############################################################################


elif [ "$COMMAND" = run_fsl ]; then
  echo "Hi!  While I really want this feature to work, I am sad to report that it is not working yet"
  echo "The job will be scheduled, but unfortunatly, it will probably break.  There is an issue with"
  echo "the `module` command not working.  I have a ticket in place right now to ask about that."
  echo ""
  echo "For now, please ssh into [username]@ssh.fsl.byu.edu and then run it directly.  Thanks."
  echo "i.e:  `./jedi.sh run_fsl_local --fsl_sbatch_arguments=\"stuff\" --fsl_python_arguments=\"stuff\"`"
  echo ""
  if [ "$FSL_USERNAME" = "" ]; then
    echo "Please provide an fsl username to ssh with"
    exit 4
  fi
  
  ssh jacobj66@ssh.fsl.byu.edu -t "cd \$HOME/fsl_groups/fslg_pccl/projects/jedi/ && pwd &&  ./jedi.sh run_fsl_local --fsl_sbatch_arguments=\"$FSL_SBATCH_ARGS\" --fsl_python_arguments=\"$FSL_PY_ARGS\""
  
###############################################################################


 elif [ "$COMMAND" = run_fsl_local ]; then
   ROM_LIST=('Breakout' ) #'WizardOfWor') #'Robotank' 'Boxing' 'StarGunner' 'Pooyan' 'Seaquest' 'Tennis' 'Enduro' 'Gopher' 'Bowling' 'VideoPinball' 'Qbert' 'MontezumaRevenge' 'Phoenix' 'Krull' 'KungFuMaster' 'Pitfall' 'DoubleDunk' 'FishingDerby' 'Riverraid' 'Carnival' 'UpNDown' 'BattleZone' 'Asteroids' 'Atlantis' 'ChopperCommand' 'Skiing' 'PrivateEye' 'Zaxxon' 'AirRaid' 'Venture' 'YarsRevenge' 'ElevatorAction' 'Frostbite' 'DemonAttack' 'Centipede' 'NameThisGame' 'Gravitar' 'Pong' 'Freeway' 'Asterix' 'Amidar' 'Jamesbond' 'BankHeist' 'Tutankham' 'SpaceInvaders' 'Alien' 'Solaris' 'TimePilot' 'Berzerk' 'JourneyEscape' 'IceHockey' 'Assault' 'RoadRunner' 'BeamRider' 'Kangaroo' 'MsPacman' 'CrazyClimber')
   echo "Welcome.  About to start up the batches"

   for i in "${!ROM_LIST[@]}"; do
     romname=${ROM_LIST[$i]}
     gen_fsl_sbatch_script $FSL_SBATCH_ARGS "--jobname=$romname"
     echo "python main.py $FSL_PY_ARGS --rom=$romname" >> sbatch.tmp 
     sbatch sbatch.tmp
     rm sbatch.tmp
  done

fi