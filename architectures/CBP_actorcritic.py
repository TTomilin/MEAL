import flax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import distrax
from flax.linen.initializers import constant, orthogonal
from cl_methods.CBP import CBPDense



class ActorCritic(nn.Module):
    """Two-layer actor & critic with CBP-enabled hidden layers."""
    action_dim: int
    activation: str = "tanh"
    cbp_eta: float = 0.0   # CBP utility decay, off by default

    def setup(self):
        # --------- ACTOR ----------
        self.a_fc1 = CBPDense(128, eta=self.cbp_eta, name="actor_fc1", activation=self.activation)
        self.a_fc2 = CBPDense(128, eta=self.cbp_eta, name="actor_fc2", activation=self.activation)
        self.actor_out = nn.Dense(self.action_dim,
                                  kernel_init=orthogonal(0.01),
                                  bias_init=constant(0.0),
                                  name="actor_out")
        # --------- CRITIC ----------
        self.c_fc1 = CBPDense(128, eta=self.cbp_eta, name="critic_fc1", activation=self.activation)
        self.c_fc2 = CBPDense(128, eta=self.cbp_eta, name="critic_fc2", activation=self.activation)
        self.critic_out = nn.Dense(1,
                                   kernel_init=orthogonal(1.0),
                                   bias_init=constant(0.0),
                                   name="critic_out")

    def _maybe_kernel(self, module_name: str, shape):
        """Return real kernel if it exists, else zeros (during init)."""
        if self.has_variable("params", module_name):
            return self.scope.get_variable("params", module_name)["kernel"]
        return jnp.zeros(shape, dtype=jnp.float32)


    def __call__(self, x, *, train: bool):
        # shapes used only for the dummy kernel during init
        dummy_hid = (128, 128)
        dummy_out = (128, self.action_dim)

        # ----- get weights (kernels) of next_layer -------
        k_a2   = self._maybe_kernel("actor_fc2_d",  dummy_hid)
        k_aout = self._maybe_kernel("actor_out",    dummy_out)
        k_c2   = self._maybe_kernel("critic_fc2_d", dummy_hid)
        k_cout = self._maybe_kernel("critic_out",   (128, 1))

        # ---------- actor ----------
        h1 = self.a_fc1(x, next_kernel=k_a2, train=train)
        h2 = self.a_fc2(h1, next_kernel=k_aout, train=train)
        logits = self.actor_out(h2)
        pi = distrax.Categorical(logits=logits)

        # ---------- critic ----------
        hc1 = self.c_fc1(x, next_kernel=k_c2, train=train)
        hc2 = self.c_fc2(hc1, next_kernel=k_cout, train=train)
        value = jnp.squeeze(self.critic_out(hc2), axis=-1)
        return pi, value
