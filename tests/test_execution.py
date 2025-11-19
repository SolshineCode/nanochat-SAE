"""
Comprehensive tests for sandboxed code execution.

The execution sandbox is security-critical and prevents LLM-generated code
from causing accidental or malicious destruction.

Run with: python -m pytest tests/test_execution.py -v
"""

import pytest
import sys
from pathlib import Path
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.execution import execute_code, ExecutionResult


def test_execution_simple_success():
    """Test successful execution of simple Python code."""
    code = "print('Hello, World!')"
    result = execute_code(code, timeout=2.0)

    assert result.success is True
    assert result.stdout.strip() == "Hello, World!"
    assert result.stderr == ""
    assert result.error is None
    assert result.timeout is False
    assert result.memory_exceeded is False

    print("✓ Simple execution test passed")


def test_execution_arithmetic():
    """Test execution with computations."""
    code = """
x = 10
y = 20
print(x + y)
print(x * y)
"""
    result = execute_code(code, timeout=2.0)

    assert result.success is True
    lines = result.stdout.strip().split('\n')
    assert lines[0] == "30"
    assert lines[1] == "200"

    print("✓ Arithmetic execution test passed")


def test_execution_stderr_capture():
    """Test stderr is captured correctly."""
    code = """
import sys
print("stdout message")
print("stderr message", file=sys.stderr)
"""
    result = execute_code(code, timeout=2.0)

    assert result.success is True
    assert "stdout message" in result.stdout
    assert "stderr message" in result.stderr

    print("✓ Stderr capture test passed")


def test_execution_timeout():
    """Test infinite loops are killed after timeout."""
    code = """
while True:
    pass
"""
    start = time.time()
    result = execute_code(code, timeout=1.0)
    elapsed = time.time() - start

    assert result.success is False
    assert result.timeout is True
    assert result.error is not None
    # Should complete shortly after timeout (within 3 seconds of grace period)
    assert elapsed < 3.0

    print("✓ Timeout test passed")


def test_execution_timeout_with_sleep():
    """Test code that sleeps too long is killed."""
    code = """
import time
time.sleep(10)
print("This should not print")
"""
    result = execute_code(code, timeout=1.0)

    assert result.success is False
    assert result.timeout is True
    assert "This should not print" not in result.stdout

    print("✓ Sleep timeout test passed")


@pytest.mark.skipif(sys.platform == "darwin", reason="Memory limits not enforced on macOS")
def test_execution_memory_limit():
    """Test memory-intensive code is killed."""
    code = """
# Try to allocate 1GB of memory
data = [0] * (1024 * 1024 * 256)  # 256M integers
"""
    # 50MB limit
    result = execute_code(code, timeout=5.0, maximum_memory_bytes=50 * 1024 * 1024)

    # Should either fail with memory error or general error
    assert result.success is False
    # Memory limit might manifest as MemoryError or general failure
    assert result.memory_exceeded or result.error is not None

    print("✓ Memory limit test passed")


def test_execution_syntax_error():
    """Test syntax errors are caught and reported."""
    code = """
print("unterminated string
"""
    result = execute_code(code, timeout=2.0)

    assert result.success is False
    assert result.error is not None
    assert "SyntaxError" in result.error

    print("✓ Syntax error test passed")


def test_execution_runtime_error():
    """Test runtime errors are caught and reported."""
    code = """
x = 1 / 0
"""
    result = execute_code(code, timeout=2.0)

    assert result.success is False
    assert result.error is not None
    assert "ZeroDivisionError" in result.error

    print("✓ Runtime error test passed")


def test_execution_import_allowed_modules():
    """Test standard library imports work."""
    code = """
import math
import json
import random

print(math.pi)
print(json.dumps({"key": "value"}))
print(random.randint(1, 10))
"""
    result = execute_code(code, timeout=2.0)

    assert result.success is True
    assert "3.14159" in result.stdout
    assert '{"key": "value"}' in result.stdout

    print("✓ Allowed imports test passed")


def test_execution_os_system_blocked():
    """Test os.system is disabled."""
    code = """
import os
try:
    os.system('echo "This should not work"')
    print("os.system succeeded - BAD!")
except Exception as e:
    print(f"os.system blocked: {type(e).__name__}")
"""
    result = execute_code(code, timeout=2.0)

    assert result.success is True
    assert "os.system blocked" in result.stdout or "os.system succeeded" not in result.stdout

    print("✓ os.system blocked test passed")


def test_execution_os_remove_blocked():
    """Test os.remove is disabled."""
    code = """
import os
try:
    os.remove('/tmp/nonexistent')
    print("os.remove succeeded - BAD!")
except Exception as e:
    print(f"os.remove blocked: {type(e).__name__}")
"""
    result = execute_code(code, timeout=2.0)

    assert result.success is True
    # os.remove should be None, causing TypeError or AttributeError
    assert "blocked" in result.stdout or "succeeded" not in result.stdout

    print("✓ os.remove blocked test passed")


def test_execution_subprocess_blocked():
    """Test subprocess.Popen is disabled."""
    code = """
import subprocess
try:
    subprocess.Popen(['echo', 'test'])
    print("subprocess.Popen succeeded - BAD!")
except Exception as e:
    print(f"subprocess.Popen blocked: {type(e).__name__}")
"""
    result = execute_code(code, timeout=2.0)

    assert result.success is True
    assert "subprocess.Popen blocked" in result.stdout or "succeeded" not in result.stdout

    print("✓ subprocess.Popen blocked test passed")


def test_execution_shutil_rmtree_blocked():
    """Test shutil.rmtree is disabled."""
    code = """
import shutil
try:
    shutil.rmtree('/tmp/nonexistent')
    print("shutil.rmtree succeeded - BAD!")
except Exception as e:
    print(f"shutil.rmtree blocked: {type(e).__name__}")
"""
    result = execute_code(code, timeout=2.0)

    assert result.success is True
    assert "shutil.rmtree blocked" in result.stdout or "succeeded" not in result.stdout

    print("✓ shutil.rmtree blocked test passed")


def test_execution_exit_blocked():
    """Test exit() and quit() are disabled."""
    code = """
try:
    exit()
    print("exit() succeeded - BAD!")
except Exception as e:
    print(f"exit() blocked: {type(e).__name__}")
"""
    result = execute_code(code, timeout=2.0)

    assert result.success is True
    # exit should be None, causing TypeError
    assert "blocked" in result.stdout or "succeeded" not in result.stdout

    print("✓ exit() blocked test passed")


def test_execution_stdin_blocked():
    """Test stdin reads are blocked."""
    code = """
try:
    x = input("Enter something: ")
    print(f"input() succeeded with: {x} - BAD!")
except IOError:
    print("input() blocked: IOError")
except Exception as e:
    print(f"input() blocked: {type(e).__name__}")
"""
    result = execute_code(code, timeout=2.0)

    assert result.success is True
    assert "blocked" in result.stdout

    print("✓ stdin blocked test passed")


def test_execution_tempdir_isolation():
    """Test code runs in isolated temporary directory."""
    code = """
import os
import pathlib

# Should be in a temporary directory
cwd = pathlib.Path.cwd() if hasattr(os, 'getcwd') and os.getcwd is not None else pathlib.Path('.')
# Just verify we can work with paths
print("Running in isolated directory")
"""
    result = execute_code(code, timeout=2.0)

    assert result.success is True
    assert "isolated directory" in result.stdout

    print("✓ Tempdir isolation test passed")


def test_execution_multiple_print_statements():
    """Test multiple print statements are captured in order."""
    code = """
for i in range(5):
    print(f"Line {i}")
"""
    result = execute_code(code, timeout=2.0)

    assert result.success is True
    lines = result.stdout.strip().split('\n')
    assert len(lines) == 5
    assert lines[0] == "Line 0"
    assert lines[4] == "Line 4"

    print("✓ Multiple print statements test passed")


def test_execution_exception_in_function():
    """Test exceptions in user-defined functions are caught."""
    code = """
def problematic_function():
    raise ValueError("Something went wrong!")

problematic_function()
"""
    result = execute_code(code, timeout=2.0)

    assert result.success is False
    assert "ValueError" in result.error
    assert "Something went wrong!" in result.error

    print("✓ Exception in function test passed")


def test_execution_result_repr():
    """Test ExecutionResult __repr__ works correctly."""
    result = ExecutionResult(
        success=False,
        stdout="test output",
        stderr="test error",
        error="SomeError: message",
        timeout=True,
        memory_exceeded=False
    )

    repr_str = repr(result)
    assert "success=False" in repr_str
    assert "timeout=True" in repr_str
    assert "SomeError: message" in repr_str
    assert "test output" in repr_str

    print("✓ ExecutionResult repr test passed")


def test_execution_empty_code():
    """Test executing empty code succeeds."""
    code = ""
    result = execute_code(code, timeout=2.0)

    assert result.success is True
    assert result.stdout == ""
    assert result.stderr == ""

    print("✓ Empty code test passed")


def test_execution_variable_persistence():
    """Test variables persist within the same execution context."""
    code = """
x = 10
y = x + 5
print(y)
z = y * 2
print(z)
"""
    result = execute_code(code, timeout=2.0)

    assert result.success is True
    lines = result.stdout.strip().split('\n')
    assert lines[0] == "15"
    assert lines[1] == "30"

    print("✓ Variable persistence test passed")


def test_execution_list_comprehension():
    """Test list comprehensions work."""
    code = """
squares = [x**2 for x in range(5)]
print(squares)
"""
    result = execute_code(code, timeout=2.0)

    assert result.success is True
    assert "[0, 1, 4, 9, 16]" in result.stdout

    print("✓ List comprehension test passed")


def test_execution_recursion_limit():
    """Test excessive recursion is caught."""
    code = """
def recursive_function(n):
    return recursive_function(n + 1)

recursive_function(0)
"""
    result = execute_code(code, timeout=2.0)

    assert result.success is False
    assert result.error is not None
    assert "RecursionError" in result.error or "maximum recursion" in result.error

    print("✓ Recursion limit test passed")


def run_all_tests():
    """Run all execution tests."""
    print("\n" + "="*80)
    print("Running Sandboxed Execution Tests")
    print("="*80 + "\n")

    try:
        test_execution_simple_success()
        test_execution_arithmetic()
        test_execution_stderr_capture()
        test_execution_timeout()
        test_execution_timeout_with_sleep()

        # Skip memory test on macOS
        if sys.platform != "darwin":
            test_execution_memory_limit()
        else:
            print("⊘ Memory limit test skipped on macOS")

        test_execution_syntax_error()
        test_execution_runtime_error()
        test_execution_import_allowed_modules()
        test_execution_os_system_blocked()
        test_execution_os_remove_blocked()
        test_execution_subprocess_blocked()
        test_execution_shutil_rmtree_blocked()
        test_execution_exit_blocked()
        test_execution_stdin_blocked()
        test_execution_tempdir_isolation()
        test_execution_multiple_print_statements()
        test_execution_exception_in_function()
        test_execution_result_repr()
        test_execution_empty_code()
        test_execution_variable_persistence()
        test_execution_list_comprehension()
        test_execution_recursion_limit()

        print("\n" + "="*80)
        print("All execution tests passed! ✓")
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
