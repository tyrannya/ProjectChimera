"""HISTORICAL - not on the demo path.

The Freqtrade risk-adapter base package (``RiskAwareStrategy``). Disconnected
at S3 by PR-13 under section 3.3 of
``docs/proposed_demo_implementation_master_plan.md``. Kept, tested and reachable
in the history; not imported by ``chimera.demo``, ``chimera.carry``,
``chimera.recorder`` or any active CLI, and not started by the default compose
stack. ``tests/test_retired_runtime_disconnected.py`` asserts that
structurally. Deletion, if it happens at all, is PR-16 after S4 -- not this
change.
"""
