"""
Comprehensive tests for tokenizer conversation rendering.

Conversation rendering is complex with special tokens for users, assistants,
and tool usage. Bugs here cause training/inference mismatches.

Run with: python -m pytest tests/test_tokenizer.py -v
"""

import pytest
import sys
import tempfile
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS


@pytest.fixture(scope="module")
def tokenizer():
    """Create a simple tokenizer for testing."""
    # Train on simple text
    text_iterator = [
        "Hello world! How are you doing today?",
        "I am fine, thank you.",
        "What is your name?",
        "My name is Assistant.",
    ] * 100

    tok = RustBPETokenizer.train_from_iterator(text_iterator, vocab_size=300)
    return tok


def test_tokenizer_special_tokens(tokenizer):
    """Test all special tokens can be encoded."""
    for special_token in SPECIAL_TOKENS:
        token_id = tokenizer.encode_special(special_token)
        assert token_id is not None
        assert isinstance(token_id, int)

        # Should be able to decode back
        decoded = tokenizer.decode([token_id])
        assert special_token in decoded

    print("✓ Special tokens test passed")


def test_tokenizer_bos_token(tokenizer):
    """Test BOS token encoding."""
    bos_id = tokenizer.get_bos_token_id()
    assert bos_id is not None
    assert isinstance(bos_id, int)

    # BOS token should match encode_special
    assert bos_id == tokenizer.encode_special("<|bos|>")

    print("✓ BOS token test passed")


def test_tokenizer_basic_encode_decode(tokenizer):
    """Test basic encoding and decoding."""
    text = "Hello, world!"
    ids = tokenizer.encode(text)

    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)

    # Decode back
    decoded = tokenizer.decode(ids)
    assert text in decoded or decoded in text  # Might have minor differences

    print("✓ Basic encode/decode test passed")


def test_tokenizer_prepend_append(tokenizer):
    """Test prepending and appending special tokens."""
    text = "Hello"
    bos_id = tokenizer.get_bos_token_id()

    # Prepend BOS
    ids_with_bos = tokenizer.encode(text, prepend="<|bos|>")
    assert ids_with_bos[0] == bos_id

    # Append BOS
    ids_with_bos_end = tokenizer.encode(text, append="<|bos|>")
    assert ids_with_bos_end[-1] == bos_id

    # Both
    ids_both = tokenizer.encode(text, prepend="<|bos|>", append="<|bos|>")
    assert ids_both[0] == bos_id
    assert ids_both[-1] == bos_id

    print("✓ Prepend/append test passed")


def test_render_conversation_simple(tokenizer):
    """Test rendering simple user-assistant conversation."""
    conversation = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }

    ids, mask = tokenizer.render_conversation(conversation)

    # Check types
    assert isinstance(ids, list)
    assert isinstance(mask, list)
    assert len(ids) == len(mask)
    assert len(ids) > 0

    # Check mask values
    assert all(m in [0, 1] for m in mask)

    # BOS token should be first (mask=0)
    bos_id = tokenizer.get_bos_token_id()
    assert ids[0] == bos_id
    assert mask[0] == 0

    # Should have user_start and user_end tokens
    user_start_id = tokenizer.encode_special("<|user_start|>")
    user_end_id = tokenizer.encode_special("<|user_end|>")
    assert user_start_id in ids
    assert user_end_id in ids

    # Should have assistant_start and assistant_end tokens
    assistant_start_id = tokenizer.encode_special("<|assistant_start|>")
    assistant_end_id = tokenizer.encode_special("<|assistant_end|>")
    assert assistant_start_id in ids
    assert assistant_end_id in ids

    print("✓ Simple conversation rendering test passed")


def test_render_conversation_mask_correctness(tokenizer):
    """Test mask correctly marks assistant tokens."""
    conversation = {
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
    }

    ids, mask = tokenizer.render_conversation(conversation)

    # Find assistant content region
    assistant_start_id = tokenizer.encode_special("<|assistant_start|>")
    assistant_end_id = tokenizer.encode_special("<|assistant_end|>")

    start_idx = ids.index(assistant_start_id)
    end_idx = ids.index(assistant_end_id)

    # Assistant start and end tokens should have mask=0 or 1
    assert mask[start_idx] == 0  # Start token not supervised
    assert mask[end_idx] == 1  # End token IS supervised

    # Tokens between assistant_start and assistant_end should be supervised (mask=1)
    # except for the start token itself
    assistant_content_masks = mask[start_idx+1:end_idx+1]
    assert all(m == 1 for m in assistant_content_masks)

    # User tokens should not be supervised (mask=0)
    user_start_id = tokenizer.encode_special("<|user_start|>")
    user_end_id = tokenizer.encode_special("<|user_end|>")
    user_start_idx = ids.index(user_start_id)
    user_end_idx = ids.index(user_end_id)

    user_region_masks = mask[user_start_idx:user_end_idx+1]
    assert all(m == 0 for m in user_region_masks)

    print("✓ Mask correctness test passed")


def test_render_conversation_multi_turn(tokenizer):
    """Test multi-turn conversation rendering."""
    conversation = {
        "messages": [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer"},
        ]
    }

    ids, mask = tokenizer.render_conversation(conversation)

    # Should have multiple user/assistant pairs
    user_start_id = tokenizer.encode_special("<|user_start|>")
    assistant_start_id = tokenizer.encode_special("<|assistant_start|>")

    user_count = ids.count(user_start_id)
    assistant_count = ids.count(assistant_start_id)

    assert user_count == 2
    assert assistant_count == 2

    print("✓ Multi-turn conversation test passed")


def test_render_conversation_with_python_tool(tokenizer):
    """Test conversation with Python tool usage."""
    conversation = {
        "messages": [
            {"role": "user", "content": "Calculate 2+2"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me calculate that"},
                    {"type": "python", "text": "2+2"},
                    {"type": "python_output", "text": "4"},
                    {"type": "text", "text": "The answer is 4"},
                ]
            },
        ]
    }

    ids, mask = tokenizer.render_conversation(conversation)

    # Should have python_start and python_end tokens
    python_start_id = tokenizer.encode_special("<|python_start|>")
    python_end_id = tokenizer.encode_special("<|python_end|>")
    assert python_start_id in ids
    assert python_end_id in ids

    # Should have output_start and output_end tokens
    output_start_id = tokenizer.encode_special("<|output_start|>")
    output_end_id = tokenizer.encode_special("<|output_end|>")
    assert output_start_id in ids
    assert output_end_id in ids

    # Python code should be supervised (mask=1)
    python_start_idx = ids.index(python_start_id)
    python_end_idx = ids.index(python_end_id)
    python_region_masks = mask[python_start_idx:python_end_idx+1]
    # All tokens in python region should have mask=1
    assert all(m == 1 for m in python_region_masks)

    # Python output should NOT be supervised (mask=0)
    output_start_idx = ids.index(output_start_id)
    output_end_idx = ids.index(output_end_id)
    output_region_masks = mask[output_start_idx:output_end_idx+1]
    assert all(m == 0 for m in output_region_masks)

    print("✓ Python tool conversation test passed")


def test_render_conversation_system_message(tokenizer):
    """Test conversation with system message (merged into first user message)."""
    conversation = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
    }

    ids, mask = tokenizer.render_conversation(conversation)

    # System message should be merged with user message
    # We should only have 1 user message and 1 assistant message
    user_start_id = tokenizer.encode_special("<|user_start|>")
    user_count = ids.count(user_start_id)
    assert user_count == 1

    assistant_start_id = tokenizer.encode_special("<|assistant_start|>")
    assistant_count = ids.count(assistant_start_id)
    assert assistant_count == 1

    print("✓ System message test passed")


def test_render_conversation_truncation(tokenizer):
    """Test conversation is truncated to max_tokens."""
    # Create a very long conversation
    long_content = "word " * 1000
    conversation = {
        "messages": [
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": long_content},
        ]
    }

    max_tokens = 100
    ids, mask = tokenizer.render_conversation(conversation, max_tokens=max_tokens)

    # Should be truncated
    assert len(ids) <= max_tokens
    assert len(mask) <= max_tokens
    assert len(ids) == len(mask)

    print("✓ Truncation test passed")


def test_render_for_completion(tokenizer):
    """Test rendering for completion (RL setting)."""
    conversation = {
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
    }

    ids = tokenizer.render_for_completion(conversation)

    # Should return just ids, no mask
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)

    # Should end with assistant_start token (ready for completion)
    assistant_start_id = tokenizer.encode_special("<|assistant_start|>")
    assert assistant_start_id in ids

    # Should NOT contain the last assistant message content
    # (because it was popped for completion)
    # We can't easily verify this without encoding "Answer", but at least
    # check that the conversation is shorter than the full render
    full_ids, _ = tokenizer.render_conversation(conversation)
    assert len(ids) < len(full_ids)

    print("✓ Render for completion test passed")


def test_visualize_tokenization(tokenizer):
    """Test tokenization visualization works."""
    conversation = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    }

    ids, mask = tokenizer.render_conversation(conversation)

    # Test without token IDs
    vis = tokenizer.visualize_tokenization(ids, mask, with_token_id=False)
    assert isinstance(vis, str)
    assert len(vis) > 0

    # Test with token IDs
    vis_with_ids = tokenizer.visualize_tokenization(ids, mask, with_token_id=True)
    assert isinstance(vis_with_ids, str)
    assert len(vis_with_ids) > len(vis)  # Should be longer with IDs

    print("✓ Visualization test passed")


def test_tokenizer_save_load(tokenizer):
    """Test tokenizer can be saved and loaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        tokenizer.save(tmpdir)

        # Check files exist
        pickle_path = Path(tmpdir) / "tokenizer.pkl"
        assert pickle_path.exists()

        # Load
        loaded_tokenizer = RustBPETokenizer.from_directory(tmpdir)

        # Test that it works the same
        text = "Hello world"
        ids1 = tokenizer.encode(text)
        ids2 = loaded_tokenizer.encode(text)
        assert ids1 == ids2

        # Test conversation rendering
        conversation = {
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ]
        }

        ids1, mask1 = tokenizer.render_conversation(conversation)
        ids2, mask2 = loaded_tokenizer.render_conversation(conversation)

        assert ids1 == ids2
        assert mask1 == mask2

    print("✓ Save/load test passed")


def test_conversation_alternating_roles(tokenizer):
    """Test that conversations must have alternating user/assistant roles."""
    # This should work
    valid_conversation = {
        "messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
    }

    ids, mask = tokenizer.render_conversation(valid_conversation)
    assert len(ids) > 0

    # This should fail (two user messages in a row)
    invalid_conversation = {
        "messages": [
            {"role": "user", "content": "Q1"},
            {"role": "user", "content": "Q2"},  # Invalid!
            {"role": "assistant", "content": "A"},
        ]
    }

    with pytest.raises(AssertionError):
        tokenizer.render_conversation(invalid_conversation)

    print("✓ Alternating roles validation test passed")


def test_conversation_must_start_with_user(tokenizer):
    """Test that conversations must start with user (after optional system)."""
    # This should work
    valid_conversation = {
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
    }

    ids, mask = tokenizer.render_conversation(valid_conversation)
    assert len(ids) > 0

    # This should also work (system + user)
    valid_with_system = {
        "messages": [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
    }

    ids, mask = tokenizer.render_conversation(valid_with_system)
    assert len(ids) > 0

    # This should fail (starts with assistant)
    invalid_conversation = {
        "messages": [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Hi"},
        ]
    }

    with pytest.raises(AssertionError):
        tokenizer.render_conversation(invalid_conversation)

    print("✓ Start with user validation test passed")


def run_all_tests():
    """Run all tokenizer tests."""
    print("\n" + "="*80)
    print("Running Tokenizer Tests")
    print("="*80 + "\n")

    # Create tokenizer fixture
    text_iterator = [
        "Hello world! How are you doing today?",
        "I am fine, thank you.",
        "What is your name?",
        "My name is Assistant.",
    ] * 100

    tok = RustBPETokenizer.train_from_iterator(text_iterator, vocab_size=300)

    try:
        test_tokenizer_special_tokens(tok)
        test_tokenizer_bos_token(tok)
        test_tokenizer_basic_encode_decode(tok)
        test_tokenizer_prepend_append(tok)
        test_render_conversation_simple(tok)
        test_render_conversation_mask_correctness(tok)
        test_render_conversation_multi_turn(tok)
        test_render_conversation_with_python_tool(tok)
        test_render_conversation_system_message(tok)
        test_render_conversation_truncation(tok)
        test_render_for_completion(tok)
        test_visualize_tokenization(tok)
        test_tokenizer_save_load(tok)
        test_conversation_alternating_roles(tok)
        test_conversation_must_start_with_user(tok)

        print("\n" + "="*80)
        print("All tokenizer tests passed! ✓")
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
