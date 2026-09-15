# Survival AI TUI

Run:

```bash
python survival_ai_tui.py
```

Windows:

```bash
pip install windows-curses
python survival_ai_tui.py
```

Controls:

- P: pause/resume
- SPACE: single step while paused
- +/-: change speed
- R: reset world
- S: save brain
- C: clear event log
- Q: quit

The learned Q-table is saved to `survival_brain.pkl`.

The AI receives raw 5x5 local vision plus coarse hunger/health.
It is not explicitly told where food is, what direction is safe,
or that moving toward food is good.
