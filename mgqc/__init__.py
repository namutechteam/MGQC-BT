"""Bond-type-conditioned coupling in graph-structured quantum circuits.

The package is two things:

    mgqc.core          the circuit, the dataset, the trainer, the statistics
    mgqc.experiments   one module per experiment the manuscript reports

Every experiment module is a thin wrapper: it defines the arms it compares and
hands them to ``core.run_arm`` through ``core.parallel_jobs``. The physics and
the statistics live in ``core`` and are shared by all of them.

Run them through the single entry point::

    python -m mgqc list
    python -m mgqc run accuracy --configs A1,B5 --seeds 20
    python -m mgqc selftest
"""

__version__ = "1.0.0"
