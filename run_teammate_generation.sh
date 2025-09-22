DEFAULTVALUE=0
device="${1:-$DEFAULTVALUE}"
dfficulty="${2:-"easy"}"
# Are exclusive. String will always overwrite int if both given and name is not empty.
layout_idx="${3:-0}"
layout_name="${4:-""}"

CUDA_VISIBLE_DEVICES=${device} LD_LIBRARY_PATH="" nice -n 5 python -m eval_agents_generation.run \
    --layout_difficulty ${dfficulty} \
    --layout_idx ${layout_idx} \
    --layout_name ${layout_name}