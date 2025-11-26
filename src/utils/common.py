"""
Common utilities for the Isaac Sim2Real Pipeline.

This module provides shared functionality used across all pipeline steps including
logging configuration, visualization tools, data processing utilities, and helper functions.
"""

import subprocess
import logging

logger = logging.getLogger(__name__)


def load_machine_pool(pool_path="configs/machines_pool.txt", test_ssh=True, timeout=10):
    """
    Load machine pool from a text file and optionally test SSH connectivity.

    Args:
        pool_path: Path to the machine pool text file
        test_ssh: Whether to test SSH connectivity (default: True)
        timeout: SSH connection timeout in seconds (default: 10)

    Returns:
        List of accessible machine addresses (strings), empty list if file not found
    """
    machine_pool = []
    with open(pool_path, 'r') as f:
        machine_pool = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(machine_pool)} machines from pool")

    # Test SSH connectivity if requested
    if test_ssh and machine_pool:
        ssh_results = test_machine_pool_ssh(machine_pool, timeout=timeout)

        if ssh_results['failed']:
            logger.warning(f"Removing {len(ssh_results['failed'])} inaccessible machines from pool")

        machine_pool = ssh_results['accessible']

        if not machine_pool:
            logger.error("No accessible machines in pool after SSH testing")

    return machine_pool


def test_machine_pool_ssh(machine_pool, timeout=10):
    """
    Test SSH connectivity to all machines in the pool.

    Args:
        machine_pool: List of machine addresses (user@host format)
        timeout: SSH connection timeout in seconds (default: 10)

    Returns:
        dict with 'accessible' and 'failed' keys containing lists of machines
    """
    accessible_machines = []
    failed_machines = []

    logger.info(f"Testing SSH connectivity to {len(machine_pool)} machines...")

    for machine in machine_pool:
        try:
            # Test SSH with a simple echo command
            # -o BatchMode=yes: prevents password prompts
            # -o ConnectTimeout: sets connection timeout
            # -o StrictHostKeyChecking=accept-new: auto-accepts new host keys
            result = subprocess.run(
                ['ssh', '-o', 'BatchMode=yes',
                 '-o', f'ConnectTimeout={timeout}',
                 '-o', 'StrictHostKeyChecking=accept-new',
                 machine, 'echo "SSH connection successful"'],
                capture_output=True,
                timeout=timeout + 5,
                text=True
            )

            if result.returncode == 0:
                accessible_machines.append(machine)
                logger.info(f"✓ {machine} - SSH connection successful")
            else:
                failed_machines.append(machine)
                logger.warning(f"✗ {machine} - SSH failed with return code {result.returncode}")
                if result.stderr:
                    logger.warning(f"  Error: {result.stderr.strip()}")

        except subprocess.TimeoutExpired:
            failed_machines.append(machine)
            logger.warning(f"✗ {machine} - SSH connection timeout")
        except Exception as e:
            failed_machines.append(machine)
            logger.warning(f"✗ {machine} - Error: {e}")

    logger.info(f"SSH test complete: {len(accessible_machines)}/{len(machine_pool)} machines accessible")

    return {
        'accessible': accessible_machines,
        'failed': failed_machines
    }
