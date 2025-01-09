from craftax.craftax_classic.envs.craftax_state import EnvState, Inventory
from dataclasses import fields
from craftax.craftax_classic.constants import *
from jax import Array


def compute_score(state: EnvState, done: Array):
    achievements = state.achievements
    info = {}
    for achievement in Achievement:
        name = f"AlreadyAchieved/{achievement.name.lower()}"
        info[name] = achievements[:, achievement.value]

    for inventory in fields(Inventory):
        info["Inventory/" + inventory.name] = getattr(state.inventory, inventory.name)

    info["health"] = state.player_health
    info["food"] = state.player_food
    info["drink"] = state.player_drink
    info["energy"] = state.player_energy
    info["sleeping"] = state.is_sleeping
    info["recover"] = state.player_recover
    info["hunger"] = state.player_hunger
    info["thirst"] = state.player_thirst
    info["fatigure"] = state.player_fatigue

    # Geometric mean with an offset of 1%
    info["score"] = jnp.exp(jnp.mean(jnp.log(1 + achievements), axis=1)) - 1.0
    return info
