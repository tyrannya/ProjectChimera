"""A persistence failure is a process failure, not a trading decision (R1-i).

The adopted roadmap's R1-i ends with one sentence this module exists for:

    on any persistence failure (ENOSPC included) the runner exits non-zero
    after attempting to append the halt reason to a pre-allocated emergency
    record, so a restart never lacks a trace.

Before R1-i a failed write had three different fates depending on which file it
was. ``RiskEngine._persist`` logged and carried on, so a halt that could not be
written down stayed a halt in memory only. ``FuturesStore.save`` and
``CarryLedger.save`` raised ordinary exceptions, which the runner's own
``except Exception`` handlers turned into an ordinary ``HALT`` -- whose own
writes were then swallowed as well. Either way a process whose disk had stopped
accepting writes could carry on, or stop with an exit code that says "an
operator halt", and the next start found no trace of why.

:class:`PersistenceFailure` is the one answer. It derives from
:class:`BaseException`, not :class:`Exception`, and that is the design rather
than an accident of it: every ``except Exception`` on the demo path -- the rule
guard, the execution guard, the reconciliation guard, the halt writer's own
last-resort guard -- lets it through, exactly as they let ``KeyboardInterrupt``
and ``SystemExit`` through. A broken persistence substrate is a reason for the
process to stop, not a condition any stage of a tick may decide to absorb.

The low-level writers keep their own error types for every other caller. They
take an optional ``on_persist_failure`` hook and call it from their ``except
OSError`` branch; the demo factories (:func:`chimera.demo.risk_wiring.build_risk_engine`
and :func:`chimera.carry.factory.build_hedged_position`) pass
:func:`raise_persistence_failure`, so the demo runner gets the process-failure
contract without changing what a failed write means anywhere else.

What is carried is deliberately small and stable: the file's ROLE (its name,
never a host path), the operation, and the ``errno`` symbol. Not the exception's
text, which differs between hosts and platforms for the same failure and would
leak paths into the emergency record.
"""

from __future__ import annotations

import errno as _errno

__all__ = [
    "PersistenceFailure",
    "errno_name",
    "raise_persistence_failure",
]


def errno_name(exc: BaseException) -> str:
    """The symbolic errno of ``exc`` (``"ENOSPC"``), or ``"UNKNOWN"``.

    Symbolic rather than numeric, because the numbers differ between platforms
    for the same condition and the name is what an operator searches for.
    """
    number = getattr(exc, "errno", None)
    if isinstance(number, int):
        return _errno.errorcode.get(number, f"E{number}")
    return "UNKNOWN"


class PersistenceFailure(BaseException):
    """A durable write the runner depends on did not complete.

    ``role`` names the file (``risk.json``, ``carry_ledger.json``, the decision
    log's day file), ``operation`` what was being done to it, and ``errno`` the
    symbolic error. The original exception is chained as ``__cause__``.
    """

    def __init__(self, role: str, operation: str, errno: str) -> None:
        self.role = str(role)
        self.operation = str(operation)
        self.errno = str(errno)
        super().__init__(
            f"persistence failure: {self.errno} on {self.operation} of {self.role}"
        )

    def summary(self) -> str:
        """One line, free of host paths and platform error text."""
        return f"{self.errno} on {self.operation} of {self.role}"


def raise_persistence_failure(role: str, exc: BaseException, operation: str = "write") -> None:
    """The ``on_persist_failure`` hook the demo factories install. Never returns."""
    raise PersistenceFailure(role, operation, errno_name(exc)) from exc
