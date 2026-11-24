"""Local web dashboard to operate and observe Get the Gem RL runs (spec 13).

``python -m dashboard`` serves ``runs/`` on http://127.0.0.1:8765. The server
imports ``game``, ``pcg`` and ``rendering.palette`` only; everything that needs
PyTorch runs in subprocesses (``rl.ppo``, ``rl.rollout``).
"""
__version__ = '0.1.0'
