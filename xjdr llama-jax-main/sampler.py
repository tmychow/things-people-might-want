import jax
import jax.numpy as jnp

class Sampler:
    def __init__(self, seed:int = 1337 ):
        self.key = jax.random.PRNGKey(1337)

    def multinomial_sample_one(self, probs_sort):
        """Performs multinomial sampling without CUDA synchronization in JAX."""

        # Generate exponential random variates with shape matching probs_sort
        q = jax.random.exponential(key=jax.random.PRNGKey(0), shape=probs_sort.shape)

        # Efficiently find the indices of maximum values along the last dimension
        # using argmax with axis_indexing=True for consistency with NumPy
        return jnp.argmax(probs_sort / q, axis=-1, keepdims=True).astype(jnp.int32)


    def sample(self, probs: jnp.ndarray, temperature=1.0, top_p=None, top_k=None):
        """Samples from the probabilities using temperature, top_p, and top_k sampling in JAX."""

        p = top_p

        probs_sort_jax = jnp.flip(jnp.sort(probs, axis=-1), axis=-1).astype(jnp.float32)
        probs_idx_jax = jnp.flip(jnp.argsort(probs, axis=-1), axis=-1).astype(jnp.float32)
        probs_sum_jax = jnp.cumsum(probs_sort_jax, axis=-1)
        # mask_jax = probs_sum_jax - probs_sort_jax > p
        # probs_sort_jax = probs_sort_jax.at[mask_jax].set(0.0)
        # probs_sort_jax = jax.lax.div(probs_sort_jax, probs_sort_jax.sum(axis=-1, keepdims=True))
        mask_jax = jnp.where(probs_sum_jax - probs_sort_jax > top_p, True, False)  # Use jnp.where
        probs_sort_jax = probs_sort_jax * (1 - mask_jax)  # Set values to 0.0 using multiplication
        probs_sort_jax = probs_sort_jax / jnp.sum(probs_sort_jax, axis=-1, keepdims=True)

        next_token_jax = self.multinomial_sample_one(probs_sort_jax)
        next_token_g_jax = jnp.take_along_axis(probs_idx_jax, next_token_jax.reshape(1,1), axis=-1)

        return next_token_g_jax.astype(jnp.int32)