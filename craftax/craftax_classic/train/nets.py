import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers


class ZNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Sequential([nn.Dense(64), nn.relu, nn.Dense(12)])(x)

class LSTM(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, terminations, last_state):
        features = self.features
        batch_size = last_state[0].shape[0]

        reshaped_x = x.reshape(-1, batch_size, features)
        reshaped_terminations = terminations.reshape(-1, batch_size)

        class LSTMOut(nn.Module):
            @nn.compact
            def __call__(self, carry, inputs):
                inputs, terminate = inputs
                # Reset the hidden state on termination
                carry = jax.tree.map(
                    lambda carry_component: jax.lax.select(
                        jnp.repeat(
                            jnp.reshape(terminate != 0.0, (-1, 1)),
                            carry_component.shape[1],
                            axis=1,
                        ),
                        jnp.zeros_like(carry_component),
                        carry_component,
                    ),
                    carry,
                )
                (new_c, new_h), new_h = nn.OptimizedLSTMCell(
                    features,
                    kernel_init=initializers.orthogonal(jnp.sqrt(2)),
                    bias_init=initializers.constant(0.0),
                )(carry, inputs)
                return (new_c, new_h), ((new_c, new_h), new_h)

        model = nn.scan(
            LSTMOut, variable_broadcast="params", split_rngs={"params": False}
        )
        final_state, (new_states, y_t) = model()(
            last_state, (reshaped_x, reshaped_terminations)
        )

        if reshaped_x.shape[0] == 1:
            y_t = jnp.squeeze(y_t, axis=0)

        return y_t, final_state
