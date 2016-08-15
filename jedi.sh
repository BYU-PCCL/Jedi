# vim: filetype=sh

function explain_usage {
    echo "Usage: jedi.sh    {run_all | run_local | dashboard | use_fsl} [options]"
    echo "  for more information, use jedi.sh --help"   
}

function explain_dashboard {
    echo ".jedi.sh dashboard"
    echo ""
    echo "If you are using 'jedi dashboard', you can add a command line parameter in"
    echo "   quotes (\"[cmd]\") that will be send to each screen to be run there."
    echo ""
    echo "           Example: jedi dashboard \"top\""
    echo "           Example: jedi dashboard \"cd /mnt/pccfs/projects/jedi && top\""
    echo ""
}
function explain_runall {
    echo ".jedi.sh run_all"
    echo ""
    echo "Run all will run commands on the PCCS's local Lab Machines."
    echo "  These commands will be run in a screen environment."
    echo "  Screen allows users to keep sessions alive after disconnectiong from the terminal"
    echo "      to detach:                 Ctrl-a Ctrl-c"
    echo "      to reconnect:              screen -r"
    echo "      to quit on all computers:  Ctrl-a ':quit'"
    echo "  (internally, this will ssh and executes 'jedi run_local' on each computer)"
    echo ""
    echo "            Example: jedi run_all"
    echo ""
}
function explain_runfsl {
    echo ""
    echo "            Example: jedi run_fsl --fsl_sbatch_arguments=\"stuff\" --fsl_python_arguments=\"stuff\" "
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
        echo "[--jobname]         Default: 'genscript-example'"
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
        
        --jobname=*) jobname=${i#*=}; shift ;;
            
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
# FSL_SBATCH_ARGS="--nodes=1"
# FSL_PY_ARGS="--bypass_sql --threads=12"

###############################################################################


###############################################################################
if [ "$1" = dashboard ]; then

    export JEDI_REMOTE_COMMAND="cd /mnt/pccfs/projects/jedi ; bash --rcfile /mnt/pccfs/downloads/term_customization.bashrc -c \"${@:2}\" ; bash --rcfile /mnt/pccfs/downloads/term_customization.bashrc"
    screen -c ./.screenrc

###############################################################################

elif [ "$1" = run_all ]; then

    export JEDI_REMOTE_COMMAND="cd /mnt/pccfs/projects/jedi ; ./jedi.sh run_local ; bash"
    screen -c ./.screenrc


elif [ "$1" = run_local ]; then

    # Prints out the hostname variable of the remote machines
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

elif [ "$1" = run_fsl ]; then
    shift ;
    echo $@
    FSL_RUN_ALL_ROMS=""
    for i in "$@"
    do
        case $i in
            --fsl_sbatch_arguments=*) echo "fsl sbatch  args"; FSL_SBATCH_ARGS="${i#*=}"; shift ;;
            --fsl_python_arguments=*) echo "fsl main.py args"; FSL_PY_ARGS="${i#*=}";   shift ;;
            --fsl_username=*)         FSL_USERNAME="${i#*=}"; shift ;;
            --run_all_roms)           FSL_RUN_ALL_ROMS="--run_all_roms"; shift ;;
        esac
    done
    
    if [ "$FSL_USERNAME" = "" ]; then
        echo "Please provide an fsl username to ssh with"
        exit 4
    fi

    rsync -ru /mnt/pccfs/projects/jedi/ $FSL_USERNAME@ssh.fsl.byu.edu:/fslhome/$FSL_USERNAME/fsl_groups/fslg_pccl/projects/jedi/ --exclude '.git' --exclude '*.pyc' --exclude '__pycache__'
    ssh $FSL_USERNAME@ssh.fsl.byu.edu -t "cd \$HOME/fsl_groups/fslg_pccl/projects/jedi/ && pwd &&  ./jedi.sh run_fsl_local --fsl_sbatch_arguments=\"$FSL_SBATCH_ARGS\" --fsl_python_arguments=\"$FSL_PY_ARGS\" $FSL_RUN_ALL_ROMS"
  
###############################################################################

#-DONE- fsl usename needs to be a paramter
#-DONE- jedi dashboard "command" is not working
#-Prob- add --run_all_roms flag to determine if ROM_LIST is used at all
# TODO: clean up code (remove all debugging stuff)
# TODO: improve help for jedi.sh

elif [ "$1" = run_fsl_local ]; then

    FSL_RUN_ALL_ROMS=false
    for i in "$@"
    do
        case $i in
            --fsl_sbatch_arguments=*) FSL_SBATCH_ARGS="${i#*=}";;
            --fsl_python_arguments=*) FSL_PY_ARGS="${i#*=}";;
            --run_all_roms)           FSL_RUN_ALL_ROMS=true;;
        esac
    done

    source $HOME/fsl_groups/fslg_pccl/configs/group_bashrc
    
    if [ "$FSL_RUN_ALL_ROMS" = true ]; then
        ROM_LIST=('Breakout') # 'WizardOfWor' 'Robotank' 'Boxing' 'StarGunner' 'Pooyan' 'Seaquest' 'Tennis' 'Enduro' 'Gopher' 'Bowling' 'VideoPinball' 'Qbert' 'MontezumaRevenge' 'Phoenix' 'Krull' 'KungFuMaster' 'Pitfall' 'DoubleDunk' 'FishingDerby' 'Riverraid' 'Carnival' 'UpNDown' 'BattleZone' 'Asteroids' 'Atlantis' 'ChopperCommand' 'Skiing' 'PrivateEye' 'Zaxxon' 'AirRaid' 'Venture' 'YarsRevenge' 'ElevatorAction' 'Frostbite' 'DemonAttack' 'Centipede' 'NameThisGame' 'Gravitar' 'Pong' 'Freeway' 'Asterix' 'Amidar' 'Jamesbond' 'BankHeist' 'Tutankham' 'SpaceInvaders' 'Alien' 'Solaris' 'TimePilot' 'Berzerk' 'JourneyEscape' 'IceHockey' 'Assault' 'RoadRunner' 'BeamRider' 'Kangaroo' 'MsPacman' 'CrazyClimber')

        for i in "${!ROM_LIST[@]}"; do
            romname=${ROM_LIST[$i]}
            gen_fsl_sbatch_script "--jobname=$romname" $FSL_SBATCH_ARGS 
            echo "python main.py $FSL_PY_ARGS --rom=$romname" >> sbatch.tmp 
            sbatch sbatch.tmp
            rm sbatch.tmp
        done
    else
        gen_fsl_sbatch_script "--jobname=single_command" $FSL_SBATCH_ARGS
        echo "python main.py $FSL_PY_ARGS"  >> sbatch.tmp
         sbatch sbatch.tmp
         rm sbatch.tmp
    fi

fi