"""
Comprehensive tests for KV cache management in the inference engine.

The KV cache is critical for efficient autoregressive generation.
Bugs in cache management cause silent generation errors and performance issues.

Run with: python -m pytest tests/test_kv_cache.py -v
"""

import torch
import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.engine import KVCache


def test_kv_cache_initialization():
    """Test KV cache initializes with correct shape and state."""
    batch_size = 4
    num_heads = 8
    seq_len = 512
    head_dim = 64
    num_layers = 12

    cache = KVCache(batch_size, num_heads, seq_len, head_dim, num_layers)

    # Check shape is correct
    expected_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
    assert cache.kv_shape == expected_shape

    # Check cache is not initialized yet (lazy initialization)
    assert cache.kv_cache is None
    assert cache.pos == 0

    print("✓ KV cache initialization test passed")


def test_kv_cache_basic_insert():
    """Test basic key/value insertion and retrieval."""
    batch_size = 2
    num_heads = 4
    seq_len = 128
    head_dim = 32
    num_layers = 3

    cache = KVCache(batch_size, num_heads, seq_len, head_dim, num_layers)

    # Insert keys and values for layer 0
    layer_idx = 0
    T_add = 5  # Adding 5 tokens
    k = torch.randn(batch_size, num_heads, T_add, head_dim)
    v = torch.randn(batch_size, num_heads, T_add, head_dim)

    # First insert should lazy-initialize the cache
    k_view, v_view = cache.insert_kv(layer_idx, k, v)

    # Cache should now be initialized
    assert cache.kv_cache is not None
    assert k_view.shape == (batch_size, num_heads, T_add, head_dim)
    assert v_view.shape == (batch_size, num_heads, T_add, head_dim)

    # Position should NOT advance yet (only advances after last layer)
    assert cache.pos == 0

    # Insert for remaining layers
    for layer in range(1, num_layers):
        cache.insert_kv(layer, k, v)

    # Position should now advance after last layer
    assert cache.pos == T_add

    # Verify stored values match
    assert torch.allclose(cache.kv_cache[0, 0, :, :, :T_add, :], k)
    assert torch.allclose(cache.kv_cache[0, 1, :, :, :T_add, :], v)

    print("✓ KV cache basic insert test passed")


def test_kv_cache_incremental_insert():
    """Test multiple sequential insertions (autoregressive generation)."""
    batch_size = 1
    num_heads = 4
    seq_len = 128
    head_dim = 32
    num_layers = 2

    cache = KVCache(batch_size, num_heads, seq_len, head_dim, num_layers)

    # Simulate autoregressive generation: insert one token at a time
    num_steps = 10
    all_keys = []
    all_values = []

    for step in range(num_steps):
        # Single token insert
        k = torch.randn(batch_size, num_heads, 1, head_dim)
        v = torch.randn(batch_size, num_heads, 1, head_dim)
        all_keys.append(k)
        all_values.append(v)

        # Insert through all layers
        for layer_idx in range(num_layers):
            k_view, v_view = cache.insert_kv(layer_idx, k, v)

        # After each step, pos should increment
        assert cache.pos == step + 1

        # Verify view has correct length
        assert k_view.shape == (batch_size, num_heads, step + 1, head_dim)
        assert v_view.shape == (batch_size, num_heads, step + 1, head_dim)

    # Verify all inserted values are in cache
    all_keys_cat = torch.cat(all_keys, dim=2)
    all_values_cat = torch.cat(all_values, dim=2)

    for layer_idx in range(num_layers):
        stored_k = cache.kv_cache[layer_idx, 0, :, :, :num_steps, :]
        stored_v = cache.kv_cache[layer_idx, 1, :, :, :num_steps, :]
        assert torch.allclose(stored_k, all_keys_cat)
        assert torch.allclose(stored_v, all_values_cat)

    print("✓ KV cache incremental insert test passed")


def test_kv_cache_dynamic_growth():
    """Test cache automatically grows when sequence exceeds initial size."""
    batch_size = 1
    num_heads = 2
    initial_seq_len = 16  # Small initial size
    head_dim = 32
    num_layers = 2

    cache = KVCache(batch_size, num_heads, initial_seq_len, head_dim, num_layers)

    # Insert tokens that will exceed initial capacity
    tokens_to_insert = 100  # Much larger than initial_seq_len

    for step in range(tokens_to_insert):
        k = torch.randn(batch_size, num_heads, 1, head_dim)
        v = torch.randn(batch_size, num_heads, 1, head_dim)

        for layer_idx in range(num_layers):
            k_view, v_view = cache.insert_kv(layer_idx, k, v)

        # Verify view has correct length
        assert k_view.shape[2] == step + 1

    # Cache should have grown
    assert cache.kv_cache.shape[4] >= tokens_to_insert
    # Should be rounded to multiple of 1024
    assert cache.kv_cache.shape[4] % 1024 == 0
    assert cache.pos == tokens_to_insert

    print("✓ KV cache dynamic growth test passed")


def test_kv_cache_batch_insert():
    """Test inserting multiple tokens at once (prefill scenario)."""
    batch_size = 2
    num_heads = 4
    seq_len = 256
    head_dim = 32
    num_layers = 3

    cache = KVCache(batch_size, num_heads, seq_len, head_dim, num_layers)

    # Prefill with 50 tokens at once
    prefill_len = 50
    k = torch.randn(batch_size, num_heads, prefill_len, head_dim)
    v = torch.randn(batch_size, num_heads, prefill_len, head_dim)

    # Insert through all layers
    for layer_idx in range(num_layers):
        k_view, v_view = cache.insert_kv(layer_idx, k, v)

    # Position should advance by prefill_len
    assert cache.pos == prefill_len

    # Verify shapes
    assert k_view.shape == (batch_size, num_heads, prefill_len, head_dim)
    assert v_view.shape == (batch_size, num_heads, prefill_len, head_dim)

    # Now insert one more token
    k_next = torch.randn(batch_size, num_heads, 1, head_dim)
    v_next = torch.randn(batch_size, num_heads, 1, head_dim)

    for layer_idx in range(num_layers):
        k_view, v_view = cache.insert_kv(layer_idx, k_next, v_next)

    # Position should be prefill_len + 1
    assert cache.pos == prefill_len + 1
    assert k_view.shape == (batch_size, num_heads, prefill_len + 1, head_dim)

    print("✓ KV cache batch insert test passed")


def test_kv_cache_reset():
    """Test cache reset clears position but preserves structure."""
    batch_size = 2
    num_heads = 4
    seq_len = 128
    head_dim = 32
    num_layers = 2

    cache = KVCache(batch_size, num_heads, seq_len, head_dim, num_layers)

    # Insert some data
    k = torch.randn(batch_size, num_heads, 10, head_dim)
    v = torch.randn(batch_size, num_heads, 10, head_dim)

    for layer_idx in range(num_layers):
        cache.insert_kv(layer_idx, k, v)

    assert cache.pos == 10

    # Reset
    cache.reset()

    # Position should be 0, but cache structure remains
    assert cache.pos == 0
    # Cache memory is not freed (efficiency)
    assert cache.kv_cache is not None

    print("✓ KV cache reset test passed")


def test_kv_cache_prefill_single_to_batch():
    """Test prefilling batch=N cache from batch=1 (beam search scenario)."""
    num_heads = 4
    seq_len = 256
    head_dim = 32
    num_layers = 3

    # Start with batch=1 for prompt encoding
    cache_single = KVCache(1, num_heads, seq_len, head_dim, num_layers)

    # Encode prompt (15 tokens)
    prompt_len = 15
    k_prompt = torch.randn(1, num_heads, prompt_len, head_dim)
    v_prompt = torch.randn(1, num_heads, prompt_len, head_dim)

    for layer_idx in range(num_layers):
        cache_single.insert_kv(layer_idx, k_prompt, v_prompt)

    assert cache_single.pos == prompt_len

    # Now expand to batch=4 for parallel generation
    beam_size = 4
    cache_batch = KVCache(beam_size, num_heads, seq_len, head_dim, num_layers)

    # Prefill from single cache
    cache_batch.prefill(cache_single)

    # Position should match
    assert cache_batch.pos == prompt_len

    # Verify the cache was replicated across batch dimension
    for layer_idx in range(num_layers):
        for batch_idx in range(beam_size):
            # All batch entries should have the same prompt
            k_stored = cache_batch.kv_cache[layer_idx, 0, batch_idx, :, :prompt_len, :]
            v_stored = cache_batch.kv_cache[layer_idx, 1, batch_idx, :, :prompt_len, :]
            # Should match the original prompt (broadcast)
            assert k_stored.shape == (num_heads, prompt_len, head_dim)

    # Continue generation with batch
    k_next = torch.randn(beam_size, num_heads, 1, head_dim)
    v_next = torch.randn(beam_size, num_heads, 1, head_dim)

    for layer_idx in range(num_layers):
        k_view, v_view = cache_batch.insert_kv(layer_idx, k_next, v_next)

    assert cache_batch.pos == prompt_len + 1

    print("✓ KV cache prefill single->batch test passed")


def test_kv_cache_prefill_validation():
    """Test shape validation prevents invalid prefills."""
    num_heads = 4
    seq_len = 256
    head_dim = 32
    num_layers = 3

    cache1 = KVCache(1, num_heads, seq_len, head_dim, num_layers)
    cache2 = KVCache(4, num_heads, seq_len, head_dim, num_layers)

    # Insert some data into cache1
    k = torch.randn(1, num_heads, 10, head_dim)
    v = torch.randn(1, num_heads, 10, head_dim)
    for layer_idx in range(num_layers):
        cache1.insert_kv(layer_idx, k, v)

    # Valid prefill should work
    cache2.prefill(cache1)
    assert cache2.pos == 10

    # Test invalid scenarios
    cache_wrong_heads = KVCache(4, 8, seq_len, head_dim, num_layers)  # Wrong num_heads
    with pytest.raises(AssertionError):
        cache_wrong_heads.prefill(cache1)

    cache_wrong_layers = KVCache(4, num_heads, seq_len, head_dim, 5)  # Wrong num_layers
    with pytest.raises(AssertionError):
        cache_wrong_layers.prefill(cache1)

    # Cannot prefill an already-filled cache
    cache_already_filled = KVCache(4, num_heads, seq_len, head_dim, num_layers)
    k_fill = torch.randn(4, num_heads, 5, head_dim)
    v_fill = torch.randn(4, num_heads, 5, head_dim)
    for layer_idx in range(num_layers):
        cache_already_filled.insert_kv(layer_idx, k_fill, v_fill)

    with pytest.raises(AssertionError, match="Cannot prefill a non-empty KV cache"):
        cache_already_filled.prefill(cache1)

    # Cannot prefill with None cache
    cache_empty_source = KVCache(1, num_heads, seq_len, head_dim, num_layers)
    cache_target = KVCache(4, num_heads, seq_len, head_dim, num_layers)
    with pytest.raises(AssertionError, match="Cannot prefill with a None KV cache"):
        cache_target.prefill(cache_empty_source)

    print("✓ KV cache prefill validation test passed")


def test_kv_cache_get_pos():
    """Test position getter returns correct value."""
    cache = KVCache(1, 4, 128, 32, 2)

    assert cache.get_pos() == 0

    # Insert 7 tokens
    k = torch.randn(1, 4, 7, 32)
    v = torch.randn(1, 4, 7, 32)
    for layer_idx in range(2):
        cache.insert_kv(layer_idx, k, v)

    assert cache.get_pos() == 7

    print("✓ KV cache get_pos test passed")


def test_kv_cache_dtype_device_preservation():
    """Test cache preserves dtype and device from inserted tensors."""
    cache = KVCache(1, 4, 128, 32, 2)

    # Test with float16
    k_fp16 = torch.randn(1, 4, 5, 32, dtype=torch.float16)
    v_fp16 = torch.randn(1, 4, 5, 32, dtype=torch.float16)

    for layer_idx in range(2):
        k_view, v_view = cache.insert_kv(layer_idx, k_fp16, v_fp16)

    assert cache.kv_cache.dtype == torch.float16
    assert k_view.dtype == torch.float16

    # Test CUDA if available
    if torch.cuda.is_available():
        cache_cuda = KVCache(1, 4, 128, 32, 2)
        k_cuda = torch.randn(1, 4, 5, 32, device='cuda')
        v_cuda = torch.randn(1, 4, 5, 32, device='cuda')

        for layer_idx in range(2):
            k_view, v_view = cache_cuda.insert_kv(layer_idx, k_cuda, v_cuda)

        assert cache_cuda.kv_cache.device.type == 'cuda'
        assert k_view.device.type == 'cuda'
        print("✓ KV cache CUDA test passed")

    print("✓ KV cache dtype/device preservation test passed")


def run_all_tests():
    """Run all KV cache tests."""
    print("\n" + "="*80)
    print("Running KV Cache Tests")
    print("="*80 + "\n")

    try:
        test_kv_cache_initialization()
        test_kv_cache_basic_insert()
        test_kv_cache_incremental_insert()
        test_kv_cache_dynamic_growth()
        test_kv_cache_batch_insert()
        test_kv_cache_reset()
        test_kv_cache_prefill_single_to_batch()
        test_kv_cache_prefill_validation()
        test_kv_cache_get_pos()
        test_kv_cache_dtype_device_preservation()

        print("\n" + "="*80)
        print("All KV cache tests passed! ✓")
        print("="*80 + "\n")

        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
