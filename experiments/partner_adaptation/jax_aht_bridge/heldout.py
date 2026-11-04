from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from experiments.partner_adaptation.jax_aht_bridge.env_adapter import AhtDynamicsOvercooked, MealOvercookedAHT
from meal.env.overcooked.overcooked_env import DELIVERY_REWARD
from meal.wrappers.logging import LogWrapper

AHT_EVAL_STEPS = 400
NORMALISATION = ("jax-aht performance_bounds: NormReturn = (base_return - lo) / (hi - lo), base_return = "
                 f"{DELIVERY_REWARD:g} x soups of the pair, on an evaluation-only MEAL Overcooked with "
                 f"max_steps={AHT_EVAL_STEPS}, JaxMARL collision dynamics and jax-aht's random_reset (random agent cells)")


class AhtEvalOvercooked(AhtDynamicsOvercooked):

    def __init__(self, layout, layout_name: str = "", max_steps: int = AHT_EVAL_STEPS):
        super().__init__(layout=layout, layout_name=layout_name, random_reset=False, max_steps=max_steps)
        self._aht = MealOvercookedAHT(layout, random_reset=True, random_obj_state=False, max_steps=max_steps)

    def reset(self, key):
        k_meal, k_rand = jax.random.split(key)
        _, state = super().reset(k_meal)
        state = self._aht._randomise(k_rand, state)
        return self.get_obs(state), state


def make_aht_eval_env(layout, layout_name: str = "") -> LogWrapper:
    return LogWrapper(AhtEvalOvercooked(layout, layout_name))


def normalised_return(base_return: float, bounds: Optional[Dict[str, Any]]) -> Optional[float]:
    b = (bounds or {}).get("base_return")
    if not b:
        return None
    lo, hi = float(b[0]), float(b[1])
    return (float(base_return) - lo) / (hi - lo) if hi > lo else None


def make_eval_fn(env, ego_network, partner_population, num_episodes: int, use_cnn: bool = False):
    from experiments.partner_adaptation.train_br import make_partner_switches
    ego = env.agents[0]

    def episode(rng, ego_params, env_idx, partner_params):
        reset_switch, step_switch = make_partner_switches(env, partner_population, partner_params)
        rng, reset_key = jax.random.split(rng)
        obs, state = reset_switch(reset_key, env_idx)

        def step(carry, _):
            obs, state, rng, soups = carry
            rng, act_key, step_key = jax.random.split(rng, 3)
            ego_obs = obs[ego][None] if use_cnn else obs[ego].reshape(1, -1)
            pi, _, _ = ego_network.apply(ego_params, ego_obs, env_idx=env_idx)
            obs, state, _, _, info = step_switch(step_key, state, {ego: pi.sample(seed=act_key)}, env_idx)
            soups = soups + sum(jnp.asarray(info["soups"][a]).reshape(()) for a in env.agents)
            return (obs, state, rng, soups), None

        (_, _, _, soups), _ = jax.lax.scan(step, (obs, state, rng, jnp.float32(0)), None, AHT_EVAL_STEPS)
        return soups

    def run(rng, ego_params, env_idx, partner_params):
        keys = jax.random.split(rng, num_episodes)
        return jax.vmap(episode, in_axes=(0, None, None, None))(keys, ego_params, env_idx, partner_params)

    return jax.jit(run)


def _metric_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")


class AhtEvaluator:

    def __init__(self, layout, layout_name, ego_network, train_partners, heldout_partners, num_episodes: int,
                 seed: int, use_cnn: bool = False, heldout_prefix: str = "Eval/Heldout"):
        from experiments.partner_adaptation.train_br import make_partner_population
        self.env = make_aht_eval_env(layout, layout_name)
        self.heldout_prefix = heldout_prefix
        self.ego_network, self.num_episodes, self.seed, self.use_cnn = ego_network, int(num_episodes), seed, use_cnn
        self.targets = []
        for i, p in enumerate(train_partners):
            if p.record.kind != "planner" and getattr(p.record, "performance_bounds", None):
                self.targets.append(("train", i, p))
        for j, p in enumerate(heldout_partners):
            self.targets.append(("heldout", j, p))
        self._populations = [make_partner_population(p.policy, p.params, None) for _, _, p in self.targets]
        self._fns: Dict[Any, Any] = {}
        groups = {}
        for n, (population, params) in enumerate(self._populations):
            fn = self._fn(population)
            groups.setdefault(fn, []).append((n, params))
        self._groups = [(fn, jnp.asarray([n for n, _ in items], jnp.int32),
                         jax.tree.map(lambda *xs: jnp.stack(xs), *(params for _, params in items)))
                        for fn, items in groups.items()]

    def __bool__(self):
        return bool(self.targets)

    def _fn(self, population):
        key = (type(population), population.policy_cls, getattr(population, "test_mode", None))
        if key not in self._fns:
            self._fns[key] = make_eval_fn(self.env, self.ego_network, population, self.num_episodes, self.use_cnn)
        return self._fns[key]

    def identities(self) -> List[Dict[str, Any]]:
        return [dict(group=g, index=i, label=p.record.label, performance_bounds=p.record.performance_bounds)
                for g, i, p in self.targets]

    def device_evaluate(self, ego_params, current_head, event):
        rng = jax.random.fold_in(jax.random.PRNGKey(self.seed), 20_000 + event)
        heads = jnp.stack([jnp.asarray(idx, jnp.int32) if group == "train" else current_head
                           for group, idx, _ in self.targets])
        means = jnp.zeros(len(self.targets), jnp.float32)
        for fn, indices, params in self._groups:
            keys = jax.vmap(lambda n: jax.random.fold_in(rng, n))(indices)
            soups = jax.vmap(fn, in_axes=(0, None, 0, 0))(keys, ego_params, heads[indices], params)
            means = means.at[indices].set(soups.mean(axis=-1))
        return means

    def evaluate(self, ego_params, current_head: int, event: int) -> Dict[str, Any]:
        return self.summarise(self.device_evaluate(ego_params, jnp.int32(current_head), event), current_head, event)

    def summarise(self, mean_soups, current_head, event):
        mean_soups = np.asarray(mean_soups)
        metrics, rows, held_soups, held_norm = {}, [], [], []
        for n, (group, idx, partner) in enumerate(self.targets):
            head = idx if group == "train" else int(current_head)
            soups = float(mean_soups[n])
            base_return = DELIVERY_REWARD * soups
            norm = normalised_return(base_return, partner.record.performance_bounds)
            label = _metric_label(partner.record.label)
            if group == "train":
                if norm is not None:
                    metrics[f"Eval/NormReturn_Partner{idx}"] = norm
            else:
                metrics[f"{self.heldout_prefix}/Soup_{label}"] = soups
                held_soups.append(soups)
                if norm is not None:
                    metrics[f"{self.heldout_prefix}/NormReturn_{label}"] = norm
                    held_norm.append(norm)
            rows.append(dict(event=int(event), group=group, index=idx, partner=partner.record.label, head=head,
                             episodes=self.num_episodes, soups=soups, base_return=base_return,
                             norm_return=norm))
        if held_soups:
            metrics[f"{self.heldout_prefix}/Mean"] = float(np.mean(held_soups))
        if held_norm:
            metrics[f"{self.heldout_prefix}/NormReturn_Mean"] = float(np.mean(held_norm))
        return dict(metrics=metrics, rows=rows)
