for agent_dir in agents/agent_TD3_*; do
    streamline_dir="$agent_dir/eval_bg/streamlines"

    if [ -d "$streamline_dir" ]; then
        echo "Organizing streamlines in $streamline_dir..."
        cd "$streamline_dir" || continue

        for file in streamline_*.png streamline_*_trajectories.pkl; do
            type=$(echo "$file" | sed -E 's/.*_(circle|curve_minus|curve_plus|line|ondulating)(_trajectories\.pkl|\.png)?/\1/')
            mkdir -p "$type"
            mv "$file" "$type/"
        done

        cd - > /dev/null
    fi
done