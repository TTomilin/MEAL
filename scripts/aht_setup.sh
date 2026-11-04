#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

AHT_DIR=third_party/jax-aht
AHT_COMMIT=890a2826100f86fab3b37da6ec88720e27ac17ef  # LARG/jax-aht v1.1.0 + README update (2026)

if [[ ! -f "${AHT_DIR}/teammate_generation/CoMeDi.py" ]]; then
    if [[ -e .git ]]; then
        git submodule update --init --recursive "${AHT_DIR}"
    else
        echo "${AHT_DIR} is empty and this is not a git checkout (send.sh syncs without .git):" >&2
        echo "run 'git submodule update --init ${AHT_DIR}' locally and re-send." >&2
        exit 1
    fi
fi
if [[ -e .git ]] && command -v git >/dev/null; then
    head=$(git -C "${AHT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)
    [[ "${head}" == "${AHT_COMMIT}" ]] || echo "warning: ${AHT_DIR} is at ${head}, expected ${AHT_COMMIT}" >&2
fi

if [[ -f .venv/bin/activate && -z "${VIRTUAL_ENV:-}" ]]; then source .venv/bin/activate; fi
if python -m pip --version >/dev/null 2>&1; then
    PIP=(python -m pip install --quiet)
elif command -v uv >/dev/null; then
    PIP=(uv pip install --quiet --python "$(command -v python)")
else
    echo "neither pip nor uv is available for $(command -v python)" >&2
    exit 1
fi

# --no-deps: jax-aht pins jax 0.5.3 / flax 0.10.2; keep MEAL's JAX stack
if [[ "${PIP[0]}" == "uv" ]]; then
    "${PIP[@]}" --no-deps -e "${AHT_DIR}" "jaxmarl==0.0.7"
else
    "${PIP[@]}" --no-deps --ignore-requires-python -e "${AHT_DIR}" "jaxmarl==0.0.7"
fi
"${PIP[@]}" -e ".[aht]"

python - <<'PY'
import importlib.metadata as md
import experiments.partner_adaptation.jax_aht_bridge as bridge
from experiments.partner_adaptation.jax_aht_bridge import compat
from experiments.partner_adaptation.jax_aht_bridge.env_adapter import MealOvercookedAHT
import common.save_load_utils, agents.mlp_actor_critic, agents.agent_interface  # noqa: F401
env = MealOvercookedAHT("coord_ring")
print("jax-aht root:", bridge.JAX_AHT_ROOT)
print("obs/action tables identity:", compat.OBS_IS_IDENTITY, compat.ACTIONS_ARE_IDENTITY)
print({p: md.version(p) for p in ["jax", "flax", "orbax-checkpoint", "jaxmarl", "jax-aht"]})
PY
echo "aht setup ok"
