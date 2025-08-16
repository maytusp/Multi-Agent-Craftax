import jax
import jax.numpy as jnp

from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
from craftax.craftax_classic.constants import Action

rng = jax.random.PRNGKey(42)  # generate a random number from a seed
env = CraftaxClassicSymbolicEnv()  # create environment instance
env_params = env.default_params
# you can set the number of players like so
env.static_env_params = env.static_env_params.replace(num_players=2)

# reset the environment. Obs has shape (n, num_obs). You can get num_obs through env.observation_space(env_params).shape[0]
# This gives one observation per player, for n players
rng, _rng = jax.random.split(rng)
obs, env_state = env.reset(_rng, env_params)

# Let's pick some actions for the players (there are 17 different actions
action = jnp.array(
    [Action.UP.value, Action.DOWN.value, Action.DO.value, Action.SLEEP.value]
)

# Step the environment
rng, _rng = jax.random.split(rng)
obs, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
