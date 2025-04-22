for agent_dir in agents/agent_TD3_*; do
    eval_dir="$agent_dir/eval_bg"
    if [ -d "$eval_dir" ]; then
        echo "Organizing eval_bg in $agent_dir..."
        cd "$eval_dir"

        for file in eval_with_*.png; do
            type=$(echo "$file" | sed -E 's/.*_((circle|curve(_(minus|plus))?|line|ondulating))\.png/\1/')
            mkdir -p "$type"
            mv "$file" "$type/"
        done

        cd - > /dev/null
    fi
done


