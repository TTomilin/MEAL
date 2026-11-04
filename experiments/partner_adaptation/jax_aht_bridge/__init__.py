from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMODULE_PATH = REPO_ROOT / "third_party" / "jax-aht"
INSTALL_HINT = (
    "jax-aht is not available. Run `bash scripts/aht_setup.sh` (initialises the third_party/jax-aht submodule and "
    "installs it with `pip install --no-deps -e`, plus the `aht` extra of pyproject.toml)."
)

JAXAHT_OVERCOOKED = "overcooked-v1"
MEAL_OVERCOOKED = "meal-overcooked"


class JaxAHTUnavailable(ImportError):
    pass


def _jax_aht_root() -> Path:
    spec = importlib.util.find_spec("teammate_generation")
    if spec is not None and spec.submodule_search_locations:
        return Path(list(spec.submodule_search_locations)[0]).parent
    if (SUBMODULE_PATH / "teammate_generation").is_dir():
        sys.path.insert(0, str(SUBMODULE_PATH))
        importlib.invalidate_caches()
        return SUBMODULE_PATH
    raise JaxAHTUnavailable(INSTALL_HINT)


def _package_shell(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__file__ = str(path / "__init__.py")
    module.__package__ = name
    module.__jax_aht_bridge_shell__ = True
    sys.modules[name] = module
    if "." in name:
        parent, child = name.rsplit(".", 1)
        setattr(sys.modules[parent], child, module)
    return module


def _install_jaxmarl_shell() -> None:
    if "jaxmarl" in sys.modules:
        return
    spec = importlib.util.find_spec("jaxmarl")
    if spec is None or not spec.submodule_search_locations:
        raise JaxAHTUnavailable(f"jaxmarl is not installed. {INSTALL_HINT}")
    root = Path(list(spec.submodule_search_locations)[0])
    _package_shell("jaxmarl", root)
    environments = _package_shell("jaxmarl.environments", root / "environments")
    multi_agent_env = importlib.import_module("jaxmarl.environments.multi_agent_env")
    environments.MultiAgentEnv = multi_agent_env.MultiAgentEnv
    environments.State = multi_agent_env.State
    environments.spaces = importlib.import_module("jaxmarl.environments.spaces")
    overcooked = importlib.import_module("jaxmarl.environments.overcooked")
    environments.Overcooked = overcooked.Overcooked
    environments.overcooked_layouts = overcooked.overcooked_layouts


def _install_envs_shell(root: Path) -> None:
    existing = sys.modules.get("envs")
    if existing is not None:
        if getattr(existing, "__jax_aht_bridge_shell__", False):
            return
        location = Path(getattr(existing, "__file__", "") or "").resolve()
        if root.resolve() not in location.parents:
            raise JaxAHTUnavailable(
                f"A different top-level 'envs' package is already imported ({location}); it shadows jax-aht's "
                "'envs'. Import experiments.partner_adaptation.jax_aht_bridge before it, or run scripts as modules "
                "(python -m ...) from the repository root."
            )
    envs = _package_shell("envs", root / "envs")
    envs.make_env = make_env


def make_env(env_name: str, env_kwargs: dict | None = None):
    env_kwargs = dict(env_kwargs or {})
    if env_name == MEAL_OVERCOOKED:
        from experiments.partner_adaptation.jax_aht_bridge.env_adapter import MealOvercookedAHT
        return MealOvercookedAHT(**env_kwargs)
    if env_name == JAXAHT_OVERCOOKED:
        from experiments.partner_adaptation.jax_aht_bridge.env_adapter import make_jaxaht_overcooked
        return make_jaxaht_overcooked(**env_kwargs)
    raise NotImplementedError(
        f"jax_aht_bridge.make_env supports {MEAL_OVERCOOKED!r} and {JAXAHT_OVERCOOKED!r}, not {env_name!r}."
    )


_JAX_TREE_ALIASES = ("tree_map", "tree_leaves", "tree_flatten", "tree_unflatten", "tree_structure")


def _install_jax_aliases() -> None:
    import jax
    for name in _JAX_TREE_ALIASES:
        try:
            getattr(jax, name)
        except AttributeError:
            setattr(jax, name, getattr(jax.tree_util, name))


def _bootstrap() -> Path:
    root = _jax_aht_root()
    _install_jax_aliases()
    try:
        _install_jaxmarl_shell()
        _install_envs_shell(root)
    except JaxAHTUnavailable:
        raise
    except ImportError as exc:
        raise JaxAHTUnavailable(f"{exc}. {INSTALL_HINT}") from exc
    return root


JAX_AHT_ROOT = _bootstrap()

__all__ = [
    "JAX_AHT_ROOT", "JAXAHT_OVERCOOKED", "MEAL_OVERCOOKED", "INSTALL_HINT", "JaxAHTUnavailable", "make_env",
]
