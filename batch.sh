if [ "$HOSTNAME" = reaper ]; then
    echo "Reaper..."
    sleep 1
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --use_prioritization --rom=Breakout
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --use_prioritization --rom=SpaceInvaders
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --use_prioritization --rom=Seaquest
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --use_prioritization --rom=Pong
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --use_prioritization --rom=Zaxxon
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --use_prioritization --rom=Freeway
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --use_prioritization --rom=Assault

    python main.py --total_ticks=3000000 --name=muse --network_type=maximummargin
    python main.py --total_ticks=3000000 --name=muse --batch_size=36 --dqn_type=optimistic

elif [ "$HOSTNAME" = naga-All-Series ]; then
    echo "Naga..."
    sleep 1
    python main.py --total_ticks=3000000 --name=partybash --rom=Breakout
    python main.py --total_ticks=3000000 --name=partybash --rom=SpaceInvaders
    python main.py --total_ticks=3000000 --name=partybash --rom=Seaquest
    python main.py --total_ticks=3000000 --name=partybash --rom=Pong
    python main.py --total_ticks=3000000 --name=partybash --rom=Zaxxon
    python main.py --total_ticks=3000000 --name=partybash --rom=Freeway
    python main.py --total_ticks=3000000 --name=partybash --rom=Assault
elif [ "$HOSTNAME" = santaka ]; then
    echo "Santaka..."
    sleep 1
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --negative_reward_on_death --use_prioritization --rom=Breakout
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --negative_reward_on_death --use_prioritization --rom=SpaceInvaders
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --negative_reward_on_death --use_prioritization --rom=Seaquest
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --negative_reward_on_death --use_prioritization --rom=Pong
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --negative_reward_on_death --use_prioritization --rom=Zaxxon
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --negative_reward_on_death --use_prioritization --rom=Freeway
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --negative_reward_on_death --use_prioritization --rom=Assault
elif [ "$HOSTNAME" = hatch ]; then
    echo "Hatch..."
    sleep 1
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --network_type=baselinedoubleduel --negative_reward_on_death --use_prioritization --rom=Breakout
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --network_type=baselinedoubleduel --negative_reward_on_death --use_prioritization --rom=SpaceInvaders
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --network_type=baselinedoubleduel --negative_reward_on_death --use_prioritization --rom=Seaquest
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --network_type=baselinedoubleduel --negative_reward_on_death --use_prioritization --rom=Pong
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --network_type=baselinedoubleduel --negative_reward_on_death --use_prioritization --rom=Zaxxon
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --network_type=baselinedoubleduel --negative_reward_on_death --use_prioritization --rom=Freeway
    python main.py --total_ticks=3000000 --name=partybash --agent_type=qexplorer --network_type=baselinedoubleduel --negative_reward_on_death --use_prioritization --rom=Assault
elif [ "$HOSTNAME" = morita.cs.byu.edu ]; then
    echo "Morita..."
    sleep 1
    python main.py --total_ticks=3000000 --name=partybash --network_type=baselinedoubleduel --rom=Breakout
    python main.py --total_ticks=3000000 --name=partybash --network_type=baselinedoubleduel --rom=SpaceInvaders
    python main.py --total_ticks=3000000 --name=partybash --network_type=baselinedoubleduel --rom=Seaquest
    python main.py --total_ticks=3000000 --name=partybash --network_type=baselinedoubleduel --rom=Pong
    python main.py --total_ticks=3000000 --name=partybash --network_type=baselinedoubleduel --rom=Zaxxon
    python main.py --total_ticks=3000000 --name=partybash --network_type=baselinedoubleduel --rom=Freeway
    python main.py --total_ticks=3000000 --name=partybash --network_type=baselinedoubleduel --rom=Assault
fi