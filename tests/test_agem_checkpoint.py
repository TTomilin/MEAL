import jax
from flax import serialization

from experiments.continual.agem import init_agem_memory
from experiments.partner_adaptation.run_outputs import restore_state


def test_agem_memory_scalar_metadata_round_trips_with_matching_dtypes():
    memory = init_agem_memory(max_size=200, obs_shape=(3,))
    encoded = serialization.msgpack_serialize(serialization.to_state_dict(memory))
    decoded = serialization.msgpack_restore(encoded)

    restored = restore_state(decoded, memory, "continual-learning state")

    assert jax.numpy.asarray(restored.max_tasks).dtype == jax.numpy.int32
    assert jax.numpy.asarray(restored.max_size_per_task).dtype == jax.numpy.int32