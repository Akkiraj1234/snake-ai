# Snake AI 🐍

A simple Snake game with an AI that learns to play it.

This project is mainly for experimenting with AI concepts such as state representation, actions, rewards, and learning through interaction with an environment.

## Overview

The AI observes the current state of the Snake world, chooses an action, and receives a reward based on the result.

```text
        ┌─────────────┐
        │ Environment │
        └──────┬──────┘
               │
             State
               │
               ▼
          ┌─────────┐
          │  Agent  │
          └────┬────┘
               │
             Action
               │
               ▼
        ┌─────────────┐
        │ Environment │
        └──────┬──────┘
               │
             Reward
               │
               └──────► Agent learns
```

The project is built from scratch so the different parts of the process can be experimented with and changed easily.

## Features

* Snake game environment
* AI agent
* Training and viewing modes
* Configurable simulation speed
* Graphical UI
* Terminal UI
* Headless mode
* Training statistics

## Project Structure

```text
snake-ai/
├── assets/
├── data/
├── docs/
├── snakeAI/
├── config.toml
├── main.py
├── requirements.txt
├── README.md
└── LICENSE
```

The project documentation is kept in `docs/`.

## Requirements

* Python 3.13+
* Dependencies listed in `requirements.txt`

Create a virtual environment and install the dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running

Start the application with the default configuration:

```bash
python main.py
```

The CLI supports training, viewing, different UI modes, simulation speed, and other options.

See the CLI documentation in `docs/` for the available options.

## Status

Work in progress.

The project is being developed and experimented with, so the implementation may change over time.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
