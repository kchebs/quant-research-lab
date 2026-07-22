# ABIDES (vendored)

This directory contains a vendored copy of the Agent-Based Interactive Discrete
Event Simulation (ABIDES) environment, originally from
https://github.com/abides-sim/abides (see LICENSE.txt).

It is included so Quant Research Lab can run agent-based market experiments
without a separate clone. Prefer installing optional deps via:

```bash
pip install -e ".[abides]"
```

Then use `quantlab.abides_runner` to put this tree on `sys.path`.
