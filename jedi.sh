#!/usr/bin/env bash
# vim: filetype=sh

if [ "$1" = help ]; then
    echo "Usage: jedi.sh    {run_all | run_local | dashboard | run_fsl | fsl_pull} [options]"

    echo "Command: jedi run_all"
    echo ""
    echo "Run all will run commands on the PCCS's local Lab Machines."
    echo "  These commands will be run in a screen environment."
    echo "  Screen allows users to keep sessions alive after disconnectiong from the terminal"
    echo "      to detach:                 Ctrl-a Ctrl-c"
    echo "      to reconnect:              screen -r"
    echo "      to quit on all computers:  Ctrl-a ':quit'"
    echo "  Internally, this will ssh and executes 'jedi run_local' on each computer"
    echo ""
    echo "Example: jedi run_all"
    echo ""


    echo "Command: jedi run_fsl"
    echo ""
    echo ""
    echo ""
    echo "Example: jedi.sh run_fsl --fsl_username=el3ment --fsl_python_arguments=\"--name=custom_agent_name --total_ticks=1000\""

    echo "[-wt|--walltime]         Default: '48:00:00'"
    echo "[-n|--nodes|--nNodes]    Default: 1"
    echo "[--nCPU|--cpus]          Default: 2"
    echo "[--nGPU|--gpus]          Default: 10"
    echo "[--mem|--total_memory]   Default: 12"
    echo "[--mem_unit]             Default: G (M is also allowed)"
    echo "[--jobname]         Default: 'genscript-example'"
    echo ""
    echo "Example: --gpus=2 -job=testing -wt=01:00:00 -n=1 --mem=24 --mem_unit=G"


# Execute a command on all lab machines
elif [ "$1" = dashboard ]; then

    export JEDI_REMOTE_COMMAND="cd /mnt/pccfs/projects/jedi ; bash --rcfile /mnt/pccfs/downloads/term_customization.bashrc -c \"${@:2}\" ; bash --rcfile /mnt/pccfs/downloads/term_customization.bashrc"
    screen -c ./.screenrc


# Ask all machines to run_local
elif [ "$1" = run_all ]; then

    export JEDI_REMOTE_COMMAND="cd /mnt/pccfs/projects/jedi ; ./jedi.sh run_local ; bash"
    screen -c ./.screenrc


# Run the predetermined commands for a given local machine
elif [ "$1" = run_local ]; then

    # Prints out the hostname variable of the remote machines
    echo "$(tput setaf 1)#####"
    echo "$HOSTNAME"
    echo "#####$(tput sgr0)"

    sleep 3

    if [ "$HOSTNAME" = reaper ]; then
        echo "Specify command to be run in jedi.sh"

    elif [ "$HOSTNAME" = santaka ]; then
        echo "Specify command to be run in jedi.sh"

    elif [ "$HOSTNAME" = ghost ]; then
        echo "Specify command to be run in jedi.sh"

    elif [ "$HOSTNAME" = naga ]; then
        echo "Specify command to be run in jedi.sh"

    elif [ "$HOSTNAME" = morita.cs.byu.edu ]; then
        echo "Specify command to be run in jedi.sh"

    elif [ "$HOSTNAME" = infinity ]; then
        echo "Specify command to be run in jedi.sh"

    elif [ "$HOSTNAME" = hatch ]; then
        echo "Specify command to be run in jedi.sh"

    elif [ "$HOSTNAME" = doctor ]; then
       echo "Specify command to be run in jedi.sh"

    elif [ "$HOSTNAME" = potts ]; then
       echo "Specify command to be run in jedi.sh"

    fi
    

# Copy database file from FSL to local machine
elif [ "$1" = fsl_pull ]; then

    for i in "$@"
    do
        case $i in
            --fsl_username=*)         FSL_USERNAME="${i#*=}"; shift ;;
        esac
    done

    if [ "$FSL_USERNAME" = "" ]; then
        echo "Please provide an fsl username to ssh with --fsl_username"
        exit 4
    fi

    echo "Beginning transfer of jedi.sqlite"
    rsync -ru --progress $FSL_USERNAME@ssh.fsl.byu.edu:/fslhome/$FSL_USERNAME/fsl_groups/fslg_pccl/projects/jedi/jedi.sqlite /mnt/pccfs/projects/jedi/fsl_jedi.sqlite


elif [ "$1" = fsl_status ]; then

    for i in "$@"
    do
        case $i in
            --fsl_username=*)         FSL_USERNAME="${i#*=}"; shift ;;
        esac
    done

    if [ "$FSL_USERNAME" = "" ]; then
        echo "Please provide an fsl username to ssh with --fsl_username"
        exit 4
    fi

    ssh $FSL_USERNAME@ssh.fsl.byu.edu -t "cd \$HOME/fsl_groups/fslg_pccl/projects/jedi/ && paste -d '\n' <(ls *.out) <(cat *.out | grep 'Name : ' | sed 's/^ *//;s/ *$//') <(tail -n1 -q *.out) <(echo '') && squeue --long -u $FSL_USERNAME"

elif [ "$1" = fsl_kill ]; then

    for i in "$@"
    do
        case $i in
            --fsl_username=*)         FSL_USERNAME="${i#*=}"; shift ;;
        esac
    done

    if [ "$FSL_USERNAME" = "" ]; then
        echo "Please provide an fsl username to ssh with --fsl_username"
        exit 4
    fi

    ssh $FSL_USERNAME@ssh.fsl.byu.edu -t "squeue --long -u $FSL_USERNAME"
    echo ""
    echo ""
    read -p "Are you sure you wish to cancel all these jobs belonging to $FSL_USERNAME? [y/n]" -n 1 -r
    echo    # (optional) move to a new line
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        ssh $FSL_USERNAME@ssh.fsl.byu.edu -t "scancel -u $FSL_USERNAME"
        echo "Jobs killed"
    fi


elif [ "$1" = fsl_clean ]; then

    for i in "$@"
    do
        case $i in
            --fsl_username=*)         FSL_USERNAME="${i#*=}"; shift ;;
        esac
    done

    if [ "$FSL_USERNAME" = "" ]; then
        echo "Please provide an fsl username to ssh with --fsl_username"
        exit 4
    fi

    ssh $FSL_USERNAME@ssh.fsl.byu.edu -t "squeue --long -u $FSL_USERNAME && ls -al *.out"
    echo ""
    echo ""
    read -p "Are you sure you wish to delete all the jedi output? [y/n]" -n 1 -r
    echo    # (optional) move to a new line
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        ssh $FSL_USERNAME@ssh.fsl.byu.edu -t "cd \$HOME/fsl_groups/fslg_pccl/projects/jedi/ && rm -r * && echo '' && ls -al"
        echo "Output cleaned."
    fi


# Command to push code up to the FSL and run jobs
elif [ "$1" = run_fsl ]; then


    shift ; # Remove "run_fsl" from command line parameters
    FSL_RUN_ALL_ROMS=""
    for i in "$@"
    do
        case $i in
            --fsl_sbatch_arguments=*) FSL_SBATCH_ARGS="${i#*=}"; shift ;;
            --fsl_python_arguments=*) FSL_PY_ARGS="${i#*=}";   shift ;;
            --fsl_username=*)         FSL_USERNAME="${i#*=}"; shift ;;
            --run_all_roms)           FSL_RUN_ALL_ROMS="--run_all_roms"; shift ;;
        esac
    done
    
    if [ "$FSL_USERNAME" = "" ]; then
        echo "Please provide an fsl username to ssh with --fsl_username"
        exit 4
    fi

    rsync -ru --progress /mnt/pccfs/projects/jedi/ $FSL_USERNAME@ssh.fsl.byu.edu:/fslhome/$FSL_USERNAME/fsl_groups/fslg_pccl/projects/jedi/ --exclude '.git' --exclude '*.pyc' --exclude '__pycache__' --exclude '.idea' --exclude 'fsl_jedi.sqlite'

    ssh $FSL_USERNAME@ssh.fsl.byu.edu -t "cd \$HOME/fsl_groups/fslg_pccl/projects/jedi/ &&  ./jedi.sh __run_fsl_local__ --fsl_sbatch_arguments=\"$FSL_SBATCH_ARGS\" --fsl_python_arguments=\"$FSL_PY_ARGS\" $FSL_RUN_ALL_ROMS"

# Hidden command only run by the fsl ssh node
elif [ "$1" = __run_fsl_local__ ]; then

    FSL_RUN_ALL_ROMS=false
    for i in "$@"
    do
        case $i in
            --fsl_sbatch_arguments=*) FSL_SBATCH_ARGS="${i#*=}";;
            --fsl_python_arguments=*) FSL_PY_ARGS="${i#*=}";;
            --run_all_roms)           FSL_RUN_ALL_ROMS=true;;
        esac
    done

    function gen_fsl_sbatch_script {
        # DEFAULTS
        walltime='48:00:00'
        nodes=1
        cpus=10
        gpus=2
        mem=12
        mem_unit='G'
        jobname='genscript-example'

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

            *) "Gen SBATCH Function: Error: unrecognized flags."; echo "BAD_VALUE_WAS: >>>$i<<<"; exit 6 ;;
          esac
        done

        SBATCH___GRES_OPT="#SBATCH --gres=gpu:$gpus"
        if [ $gpus -eq 0 ]; then
          SBATCH___GRES_OPT="# Not going to use a gpu"
        fi

        echo "#!/bin/bash"                 >> sbatch.$$.tmp
        echo ""                            >> sbatch.$$.tmp
        echo "#SBATCH --time=$walltime"    >> sbatch.$$.tmp  # walltime
        echo "#SBATCH --ntasks=$cpus"      >> sbatch.$$.tmp  # number of processor cores (i.e. tasks)
        echo "#SBATCH --nodes=$nodes"      >> sbatch.$$.tmp  # number of nodes
        echo $SBATCH___GRES_OPT            >> sbatch.$$.tmp  # gpu reqs (empty if gpus=0)
        echo "#SBATCH --qos=dw87"          >> sbatch.$$.tmp  # Runs us on the m8g2 nodes
        echo "#SBATCH --mem=$mem$mem_unit" >> sbatch.$$.tmp  # total memory requested (Note this is different than --mem-per-cpu option. # G or M available.
        echo "#SBATCH -J '$jobname'"       >> sbatch.$$.tmp  # job name
        echo "#SBATCH --gid=fslg_pccl"     >> sbatch.$$.tmp  # give group access
        echo ""                            >> sbatch.$$.tmp
        echo "source $HOME/fsl_groups/fslg_pccl/configs/group_bashrc" >> sbatch.$$.tmp # Modules, etc.
    }

    source $HOME/fsl_groups/fslg_pccl/configs/group_bashrc
    
    if [ "$FSL_RUN_ALL_ROMS" = true ]; then
        ROM_LIST=('Breakout' 'WizardOfWor' 'Robotank' 'Boxing' 'StarGunner' 'Pooyan' 'Seaquest' 'Tennis' 'Enduro' 'Gopher' 'Bowling' 'VideoPinball' 'Qbert' 'MontezumaRevenge' 'Phoenix' 'Krull' 'KungFuMaster' 'Pitfall' 'DoubleDunk' 'FishingDerby' 'Riverraid' 'Carnival' 'UpNDown' 'BattleZone' 'Asteroids' 'Atlantis' 'ChopperCommand' 'Skiing' 'PrivateEye' 'Zaxxon' 'AirRaid' 'Venture' 'YarsRevenge' 'ElevatorAction' 'Frostbite' 'DemonAttack' 'Centipede' 'NameThisGame' 'Gravitar' 'Pong' 'Freeway' 'Asterix' 'Amidar' 'Jamesbond' 'BankHeist' 'Tutankham' 'SpaceInvaders' 'Alien' 'Solaris' 'TimePilot' 'Berzerk' 'JourneyEscape' 'IceHockey' 'Assault' 'RoadRunner' 'BeamRider' 'Kangaroo' 'MsPacman' 'CrazyClimber')

        for i in "${!ROM_LIST[@]}"; do
            romname=${ROM_LIST[$i]}
            gen_fsl_sbatch_script "--jobname=$romname" $FSL_SBATCH_ARGS 
            echo "python main.py $FSL_PY_ARGS --rom=$romname" >> sbatch.$$.tmp
            sbatch sbatch.$$.tmp
            rm sbatch.$$.tmp
        done
    else
        gen_fsl_sbatch_script "--jobname=single_command" $FSL_SBATCH_ARGS
        echo "python main.py $FSL_PY_ARGS"  >> sbatch.$$.tmp

        sbatch sbatch.$$.tmp
        rm sbatch.$$.tmp
    fi

fi