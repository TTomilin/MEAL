DEFAULTVALUE=0
device="${1:-$DEFAULTVALUE}"
layout_name="${2:-"cramped_room"}"

CUDA_VISIBLE_DEVICES=${device} LD_LIBRARY_PATH="" nice -n 5 python -m eval_agents_generation.run_br \
    --layout_name ${layout_name}