import pytest
import subprocess
import sys

# Determine the correct python executable
PYTHON_EXE = sys.executable

def test_backfill_cli_missing_arguments():
    """
    Tests that running backfill.py with no arguments results in an error
    and a usage message.
    """
    # Using sys.executable to ensure the same Python interpreter is used
    # that pytest is running under, especially in virtual environments.
    process = subprocess.run(
        [PYTHON_EXE, "tools/backfill.py"],
        capture_output=True,
        text=True
    )

    assert process.returncode != 0, "Expected a non-zero return code for missing arguments."
    # Argparse typically returns 2 for argument errors
    assert process.returncode == 2, f"Expected return code 2, got {process.returncode}. Stderr: {process.stderr}"

    stderr_output = process.stderr.lower() # Convert to lower for case-insensitive matching
    assert "usage: backfill.py" in stderr_output, "Stderr should contain 'usage: backfill.py'."
    assert "error: the following arguments are required" in stderr_output, \
        "Stderr should contain 'the following arguments are required'."

def test_backfill_cli_help_message():
    """
    Tests that running backfill.py with --help shows the help message.
    """
    process = subprocess.run(
        [PYTHON_EXE, "tools/backfill.py", "--help"],
        capture_output=True,
        text=True
    )

    assert process.returncode == 0, f"Expected return code 0 for --help. Stderr: {process.stderr}"

    stdout_output = process.stdout.lower() # Convert to lower for case-insensitive matching
    assert "usage: backfill.py" in stdout_output, "Stdout should contain 'usage: backfill.py'."
    assert "download historical ohlcv data" in stdout_output, "Help message description not found."
    assert "exchange" in stdout_output, "Help message for 'exchange' not found."
    assert "symbol" in stdout_output, "Help message for 'symbol' not found."
    assert "start_date" in stdout_output, "Help message for 'start_date' not found."
    assert "tf" in stdout_output, "Help message for 'tf' (timeframe) not found."

# Note: A test like test_backfill_cli_valid_args_calls_main() would be more of an integration test
# and would require mocking data_pipeline functions to prevent actual downloads or file system operations.
# For this subtask focusing on "simple unit tests" for the CLI, the above tests for argument parsing
# and help messages are sufficient.
