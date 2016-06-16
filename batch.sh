if [ "$HOSTNAME" = reaper ]; then
    echo "Reaper..."
    sleep 1
    python main.py --total_ticks=5000000 --name=fiesta --agent_type=qexplorer --use_prioritization --rom=Breakout
    python main.py --total_ticks=5000000 --name=fiesta --agent_type=qexplorer --use_prioritization --rom=SpaceInvaders
    python main.py --total_ticks=5000000 --name=fiesta --agent_type=qexplorer --use_prioritization --rom=Seaquest
    python main.py --total_ticks=5000000 --name=fiesta --agent_type=qexplorer --use_prioritization --rom=Pong
    python main.py --total_ticks=5000000 --name=fiesta --agent_type=qexplorer --use_prioritization --rom=Zaxxon
    python main.py --total_ticks=5000000 --name=fiesta --agent_type=qexplorer --use_prioritization --rom=Freeway
    python main.py --total_ticks=5000000 --name=fiesta --agent_type=qexplorer --use_prioritization --rom=Assault
elif [ "$HOSTNAME" = naga-All-Series ]; then
    echo "Naga..."
    sleep 1
    python main.py --total_ticks=5000000 --name=fiesta --rom=Breakout
    python main.py --total_ticks=5000000 --name=fiesta --rom=SpaceInvaders
    python main.py --total_ticks=5000000 --name=fiesta --rom=Seaquest
    python main.py --total_ticks=5000000 --name=fiesta --rom=Pong
    python main.py --total_ticks=5000000 --name=fiesta --rom=Zaxxon
    python main.py --total_ticks=5000000 --name=fiesta --rom=Freeway
    python main.py --total_ticks=5000000 --name=fiesta --rom=Assault
elif [ "$HOSTNAME" = santaka ]; then
    echo "Santaka..."
    sleep 1
    python main.py --total_ticks=3000000 --name=pinata
    python main.py --total_ticks=3000000 --name=pinata --use_prioritization
    python main.py --total_ticks=3000000 --name=pinata --agent_type=density --network_type=density
    python main.py --total_ticks=3000000 --name=pinata --network_type=causal
    python main.py --total_ticks=3000000 --name=pinata --agent_type=lookahead --network_type=constrained
    python main.py --total_ticks=3000000 --name=pinata --agent_type=qexplorer
    python main.py --total_ticks=3000000 --name=pinata --agent_type=qexplorer --use_prioritization
    python main.py --total_ticks=3000000 --name=pinata --negative_reward_on_death
    python main.py --total_ticks=3000000 --name=pinata --agent_type=qexplorer --negative_reward_on_death
elif [ "$HOSTNAME" = hatch ]; then
    echo "Hatch..."
    sleep 1
    python main.py --total_ticks=5000000 --name=fiesta --use_prioritization --rom=Breakout
    python main.py --total_ticks=5000000 --name=fiesta --use_prioritization --rom=SpaceInvaders
    python main.py --total_ticks=5000000 --name=fiesta --use_prioritization --rom=Seaquest
    python main.py --total_ticks=5000000 --name=fiesta --use_prioritization --rom=Pong
    python main.py --total_ticks=5000000 --name=fiesta --use_prioritization --rom=Zaxxon
    python main.py --total_ticks=5000000 --name=fiesta --use_prioritization --rom=Freeway
    python main.py --total_ticks=5000000 --name=fiesta --use_prioritization --rom=Assault
fi
