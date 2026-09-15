"""
survival_ai_tui.py

A terminal-only reinforcement-learning survival experiment.

Controls
--------
q / Q        Quit
p / P        Pause
r / R        Reset current episode
s / S        Save brain
+ / =        Speed up
- / _        Slow down
space        Single-step while paused
c / C        Clear event log

On Windows:
    pip install windows-curses

On Linux/macOS, curses is normally included with Python.

The creature receives RAW LOCAL VISION, not hints such as:
    - direction to food
    - distance to food
    - "move toward food" rewards
    - survival rewards

It only sees nearby cell types + coarse hunger/health.
The world supplies consequences through the reward signal.
"""
from __future__ import annotations
try:
    import curses
except ImportError:
    curses = None
import os
import pickle
import random
import time
from array import array
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple



WORLD_W = 32
WORLD_H = 18
MAX_HEALTH = 100
MAX_HUNGER = 100
INITIAL_BODY_LENGTH = 3
INITIAL_FOOD = 25
OBSTACLE_COUNT = 34
STARVATION_DAMAGE = 1
STARVATION_REWARD = -5
WALL_DAMAGE = 10
WALL_REWARD = -10
OBSTACLE_DAMAGE = 10
OBSTACLE_REWARD = -10
SELF_DAMAGE = 35
SELF_REWARD = -35


FOOD_HUNGER_GAIN = 45
SURVIVAL_REWARD = 0.05

DEATH_REWARD = -100


MAX_AGE = 3000
LEARNING_RATE = 0.12
DISCOUNT = 0.92
START_EPSILON = 1.0
MIN_EPSILON = 0.02
EPSILON_DECAY = 0.997
MODEL_FILE = 'data.pkl'
MODEL_VERSION = 2
FOOD_HEALTH_GAIN = 20
HEALTH_GAIN_REWARD = 1.0
START_DELAY = 0.065
MIN_DELAY = 0.003
MAX_DELAY = 0.5
AUTOSAVE_EVERY = 25
HEADLESS_STEP_DELAY = 0.01
HEADLESS_REPORT_INTERVAL = 60.0
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
EAT = 4
WAIT = 5
ACTIONS = (UP, DOWN, LEFT, RIGHT, EAT, WAIT)
ACTION_NAMES = {UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT', EAT: 'EAT', WAIT: 'WAIT'}
EMPTY = 0
FOOD = 1
OBSTACLE = 2
BODY = 3
WALL = 4
C_BORDER = 1
C_TITLE = 2
C_SUBTITLE = 3
C_GOOD = 4
C_WARN = 5
C_BAD = 6
C_MUTED = 7
C_CREATURE = 8
C_FOOD = 9
C_OBSTACLE = 10
C_HEADER = 11
C_SELECTED = 12
C_DIM = 13
Position = Tuple[int, int]
State = Tuple[int, ...]

def clamp(value: int | float, low: int | float, high: int | float):
    return max(low, min(high, value))

def sign(value: int) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0

def pct(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return clamp(value / maximum, 0.0, 1.0)

def avg(values: Deque[int]) -> float:
    return sum(values) / len(values) if values else 0.0

def state_key(state: State) -> int:
    key = 0
    for value in state[:25]:
        key = key * 5 + value
    return (key * 4 + state[-2]) * 4 + state[-1]

@dataclass
class Creature:
    x: int = 0
    y: int = 0
    body: List[Position] = field(default_factory=list)
    health: int = MAX_HEALTH
    hunger: int = MAX_HUNGER
    age: int = 0
    score: float = 0.0
    alive: bool = True
    last_action: int = WAIT
    last_reward: float = 0.0
    foods_eaten: int = 0
    collisions: int = 0
    starvation_events: int = 0

    def reset(self) -> None:
        cx = WORLD_W // 2
        cy = WORLD_H // 2
        self.x = cx
        self.y = cy
        self.body = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.health = MAX_HEALTH
        self.hunger = MAX_HUNGER
        self.age = 0
        self.score = 0.0
        self.alive = True
        self.last_action = WAIT
        self.last_reward = 0.0
        self.foods_eaten = 0
        self.collisions = 0
        self.starvation_events = 0

class World:

    def __init__(self, record_events: bool=True) -> None:
        self.creature = Creature()
        self.food: Set[Position] = set()
        self.obstacles: Set[Position] = set()
        self.record_events = record_events
        self.events: Deque[str] = deque(maxlen=300)
        self.reset()

    def log(self, message: str) -> None:
        if not self.record_events:
            return
        stamp = time.strftime('%H:%M:%S')
        self.events.append(f'{stamp}  {message}')

    def reset(self) -> None:
        self.creature.reset()
        self.food.clear()
        self.obstacles.clear()
        self.events.clear()
        self._create_obstacles()
        self._create_food()
        self.log('new world generated')

    def _create_obstacles(self) -> None:
        attempts = 0
        while len(self.obstacles) < OBSTACLE_COUNT and attempts < 5000:
            attempts += 1
            position = (random.randrange(WORLD_W), random.randrange(WORLD_H))
            if position in self.creature.body:
                continue
            self.obstacles.add(position)

    def _create_food(self) -> None:
        attempts = 0
        while len(self.food) < INITIAL_FOOD and attempts < 5000:
            attempts += 1
            position = self.random_empty_position()
            if position is not None:
                self.food.add(position)

    def random_empty_position(self) -> Optional[Position]:
        for _ in range(500):
            position = (random.randrange(WORLD_W), random.randrange(WORLD_H))
            if position in self.obstacles:
                continue
            if position in self.creature.body:
                continue
            if position in self.food:
                continue
            return position
        return None

    def spawn_food(self) -> None:
        position = self.random_empty_position()
        if position is not None:
            self.food.add(position)

    def get_state(self) -> State:
        """
        NO INTERPRETATION / NO HINTS.

        The creature sees a 5x5 local grid centered on itself.

        Cell values:
            0 = empty
            1 = food
            2 = obstacle
            3 = own body
            4 = wall

        Then:
            hunger bucket (0..3)
            health bucket (0..3)

        There is deliberately no:
            "food is left"
            "food is closer"
            "wall is dangerous"
            etc.

        Those relationships must be learned from repeated
        state -> action -> consequence experiences.
        """
        x0 = self.creature.x
        y0 = self.creature.y
        state: List[int] = []
        radius = 2
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x = x0 + dx
                y = y0 + dy
                if x < 0 or x >= WORLD_W or y < 0 or (y >= WORLD_H):
                    value = WALL
                elif (x, y) in self.obstacles:
                    value = OBSTACLE
                elif (x, y) in self.creature.body:
                    value = BODY
                elif (x, y) in self.food:
                    value = FOOD
                else:
                    value = EMPTY
                state.append(value)
        if self.creature.hunger > 75:
            hunger_bucket = 3
        elif self.creature.hunger > 50:
            hunger_bucket = 2
        elif self.creature.hunger > 25:
            hunger_bucket = 1
        else:
            hunger_bucket = 0
        if self.creature.health > 75:
            health_bucket = 3
        elif self.creature.health > 50:
            health_bucket = 2
        elif self.creature.health > 25:
            health_bucket = 1
        else:
            health_bucket = 0
        state.append(hunger_bucket)
        state.append(health_bucket)
        return tuple(state)

    def step(self, action: int) -> float:
        creature = self.creature
        reward = 0.0
        before_health = creature.health
        creature.age += 1
        creature.last_action = action
        creature.hunger -= 1
        if action in (UP, DOWN, LEFT, RIGHT):
            nx = creature.x
            ny = creature.y
            if action == UP:
                ny -= 1
            elif action == DOWN:
                ny += 1
            elif action == LEFT:
                nx -= 1
            elif action == RIGHT:
                nx += 1
            new_pos = (nx, ny)
            if nx < 0 or nx >= WORLD_W or ny < 0 or (ny >= WORLD_H):
                creature.health -= WALL_DAMAGE
                creature.collisions += 1
                reward += WALL_REWARD
                self.log('collision: wall')
            elif new_pos in self.obstacles:
                creature.health -= OBSTACLE_DAMAGE
                creature.collisions += 1
                reward += OBSTACLE_REWARD
                self.log('collision: obstacle')
            elif new_pos in creature.body:
                creature.health -= SELF_DAMAGE
                creature.collisions += 1
                reward += SELF_REWARD
                self.log('collision: own body')
            else:
                creature.x = nx
                creature.y = ny
                creature.body.insert(0, new_pos)
                target_length = min(INITIAL_BODY_LENGTH + creature.foods_eaten // 2, 16)
                while len(creature.body) > target_length:
                    creature.body.pop()
        elif action == EAT:
            position = (creature.x, creature.y)
            if position in self.food:
                self.food.remove(position)
                before_hunger = creature.hunger
                creature.hunger = int(clamp(creature.hunger + FOOD_HUNGER_GAIN, 0, MAX_HUNGER))
                creature.health = int(clamp(creature.health + FOOD_HEALTH_GAIN, 0, MAX_HEALTH))
                creature.foods_eaten += 1
                self.log(f'food consumed  hunger {before_hunger}->{creature.hunger}')
                self.spawn_food()
            else:
                self.log('ate empty cell')
        elif action == WAIT:
            pass
        if creature.hunger <= 0:
            creature.hunger = 0
            creature.health -= STARVATION_DAMAGE
            creature.starvation_events += 1
            reward += STARVATION_REWARD
            self.log(f'starvation damage  hp={creature.health}')
        if creature.health <= 0:
            creature.health = 0
            creature.alive = False
            reward += DEATH_REWARD
            self.log('DEATH')
        health_gain = max(0, creature.health - before_health)
        reward += health_gain * HEALTH_GAIN_REWARD
        if creature.age >= MAX_AGE:
            creature.alive = False
            self.log('episode limit reached')
        if creature.alive:
            reward += SURVIVAL_REWARD
        creature.last_reward = reward
        creature.score += reward
        return reward

class Brain:

    def __init__(self) -> None:
        self.q_table: Dict[int, array] = {}
        self.epsilon = START_EPSILON
        self.total_updates = 0
        self.load()

    def values(self, state: State) -> array:
        key = state_key(state)
        if key not in self.q_table:
            self.q_table[key] = array('f', [0.0] * len(ACTIONS))
        return self.q_table[key]

    def choose_action(self, state: State) -> int:
        q = self.values(state)
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        best = max(q)
        return random.choice([i for i, value in enumerate(q) if value == best])

    def learn(self, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
        q = self.values(state)
        current = q[action]
        target = reward if done else reward + DISCOUNT * max(self.values(next_state))
        q[action] = current + LEARNING_RATE * (target - current)
        self.total_updates += 1

    def decay(self) -> None:
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

    def save(self) -> None:
        payload = {'version': MODEL_VERSION, 'q_table': self.q_table, 'epsilon': self.epsilon, 'total_updates': self.total_updates}
        temp = MODEL_FILE + '.tmp'
        with open(temp, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp, MODEL_FILE)

    def load(self) -> None:
        if not os.path.exists(MODEL_FILE):
            return
        try:
            with open(MODEL_FILE, 'rb') as f:
                payload = pickle.load(f)
            if payload.get('version') != MODEL_VERSION or not isinstance(payload.get('q_table'), dict):
                return
            self.q_table = payload['q_table']
            self.epsilon = payload.get('epsilon', START_EPSILON)
            self.total_updates = payload.get('total_updates', 0)
        except Exception:
            self.q_table, self.epsilon, self.total_updates = ({}, START_EPSILON, 0)

class TUI:
    """
    Full-screen curses UI.

    Layout:

    +----------------------------------------------------------+
    | HEADER                                                   |
    +-------------+--------------------------------+-----------+
    | STATE       |                                | BRAIN     |
    |             |             WORLD              |           |
    |             |                                |           |
    |             |                                |           |
    +-------------+--------------------------------+-----------+
    | SENSORS /   | EVENT LOG                      | Q VALUES  |
    +-------------+--------------------------------+-----------+
    | CONTROLS / STATUS                                         |
    +----------------------------------------------------------+
    """

    def __init__(self, screen):
        self.screen = screen
        self.paused = False
        self.step_once = False
        self.delay = START_DELAY
        self.toast = ''
        self.toast_until = 0.0
        self.flash_until = 0.0
        self.setup_terminal()

    def setup_terminal(self) -> None:
        curses.curs_set(0)
        self.screen.nodelay(True)
        self.screen.keypad(True)
        curses.noecho()
        curses.cbreak()
        if curses.has_colors():
            curses.start_color()
            curses.init_pair(C_BORDER, curses.COLOR_CYAN, curses.COLOR_BLACK)
            curses.init_pair(C_TITLE, curses.COLOR_WHITE, curses.COLOR_BLACK)
            curses.init_pair(C_SUBTITLE, curses.COLOR_BLUE, curses.COLOR_BLACK)
            curses.init_pair(C_GOOD, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(C_WARN, curses.COLOR_YELLOW, curses.COLOR_BLACK)
            curses.init_pair(C_BAD, curses.COLOR_RED, curses.COLOR_BLACK)
            curses.init_pair(C_MUTED, curses.COLOR_BLUE, curses.COLOR_BLACK)
            curses.init_pair(C_CREATURE, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
            curses.init_pair(C_FOOD, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(C_OBSTACLE, curses.COLOR_WHITE, curses.COLOR_BLACK)
            curses.init_pair(C_SELECTED, curses.COLOR_BLACK, curses.COLOR_CYAN)
            curses.init_pair(C_DIM, curses.COLOR_BLACK, curses.COLOR_WHITE)
            curses.init_pair(C_HEADER, curses.COLOR_CYAN, curses.COLOR_BLACK)

    def style(self, pair: int):
        if curses.has_colors():
            return curses.color_pair(pair)
        return 0

    def put(self, y: int, x: int, text: str, pair: int=C_TITLE, attr: int=0) -> None:
        try:
            self.screen.addnstr(y, x, text, max(0, self.screen.getmaxyx()[1] - x - 1), self.style(pair) | attr)
        except curses.error:
            pass

    def hline(self, y: int, x: int, width: int, pair: int=C_BORDER) -> None:
        if width <= 0:
            return
        try:
            self.screen.hline(y, x, curses.ACS_HLINE, width, self.style(pair))
        except curses.error:
            pass

    def vline(self, y: int, x: int, height: int, pair: int=C_BORDER) -> None:
        if height <= 0:
            return
        try:
            self.screen.vline(y, x, curses.ACS_VLINE, height, self.style(pair))
        except curses.error:
            pass

    def box(self, y: int, x: int, h: int, w: int, title: str='') -> None:
        if h < 2 or w < 2:
            return
        try:
            self.screen.addch(y, x, curses.ACS_ULCORNER, self.style(C_BORDER))
            self.screen.addch(y, x + w - 1, curses.ACS_URCORNER, self.style(C_BORDER))
            self.screen.addch(y + h - 1, x, curses.ACS_LLCORNER, self.style(C_BORDER))
            self.screen.addch(y + h - 1, x + w - 1, curses.ACS_LRCORNER, self.style(C_BORDER))
            self.screen.hline(y, x + 1, curses.ACS_HLINE, max(0, w - 2), self.style(C_BORDER))
            self.screen.hline(y + h - 1, x + 1, curses.ACS_HLINE, max(0, w - 2), self.style(C_BORDER))
            self.screen.vline(y + 1, x, max(0, h - 2), curses.ACS_VLINE, self.style(C_BORDER))
            self.screen.vline(y + 1, x + w - 1, max(0, h - 2), curses.ACS_VLINE, self.style(C_BORDER))
            if title:
                title_text = f' {title} '
                title_x = x + 2
                self.screen.addnstr(y, title_x, title_text, max(0, w - 4), self.style(C_TITLE) | curses.A_BOLD)
        except curses.error:
            pass

    def bar(self, value: float, maximum: float, width: int=18) -> str:
        ratio = pct(value, maximum)
        filled = int(round(ratio * width))
        return '█' * filled + '░' * (width - filled)

    def colored_bar(self, y: int, x: int, label: str, value: int, maximum: int, width: int) -> None:
        if value <= 25:
            pair = C_BAD
        elif value <= 60:
            pair = C_WARN
        else:
            pair = C_GOOD
        self.put(y, x, f'{label:<8}', C_MUTED)
        self.put(y, x + 8, self.bar(value, maximum, width), pair)
        self.put(y, x + 8 + width + 2, f'{value:3d}', pair)

    def toast_message(self, message: str, seconds: float=2.0) -> None:
        self.toast = message
        self.toast_until = time.monotonic() + seconds

    def draw_header(self, width: int, episode: int, brain: Brain, world: World, best_age: int) -> None:
        state = 'PAUSED' if self.paused else 'LEARNING'
        pair = C_WARN if self.paused else C_GOOD
        self.put(0, 2, 'SURVIVAL AI', C_TITLE, curses.A_BOLD)
        self.put(1, 2, 'reinforcement-learning laboratory', C_SUBTITLE)
        right = f'EP {episode:05d}   BEST {best_age:04d}   ε {brain.epsilon:0.3f}   {state}'
        self.put(1, max(2, width - len(right) - 2), right, pair, curses.A_BOLD)

    def draw_state_panel(self, y: int, x: int, h: int, w: int, world: World) -> None:
        creature = world.creature
        self.box(y, x, h, w, ' STATE ')
        row = y + 2
        self.put(row, x + 2, 'Vitals', C_TITLE, curses.A_BOLD)
        row += 1
        self.colored_bar(row, x + 2, 'HEALTH', creature.health, MAX_HEALTH, min(16, max(8, w - 17)))
        row += 1
        self.colored_bar(row, x + 2, 'HUNGER', creature.hunger, MAX_HUNGER, min(16, max(8, w - 17)))
        row += 2
        entries = [('AGE', str(creature.age), C_WHITE if False else C_TITLE), ('BODY', str(len(creature.body)), C_TITLE), ('FOOD', str(creature.foods_eaten), C_TITLE), ('COLLIS', str(creature.collisions), C_TITLE), ('STARVE', str(creature.starvation_events), C_TITLE)]
        for label, value, color in entries:
            self.put(row, x + 2, f'{label:<8}', C_MUTED)
            self.put(row, x + 12, value, C_TITLE if color == C_TITLE else color)
            row += 1
        row += 1
        self.put(row, x + 2, 'Current action', C_TITLE, curses.A_BOLD)
        row += 1
        self.put(row, x + 2, ACTION_NAMES[creature.last_action], C_SELECTED, curses.A_BOLD)
        row += 2
        reward_pair = C_GOOD if creature.last_reward >= 0 else C_BAD
        self.put(row, x + 2, 'Reward', C_MUTED)
        self.put(row, x + 12, f'{creature.last_reward:+.2f}', reward_pair, curses.A_BOLD)
        row += 2
        self.put(row, x + 2, 'Position', C_MUTED)
        self.put(row, x + 12, f'{creature.x:02d}, {creature.y:02d}', C_TITLE)
        row += 2
        self.put(row, x + 2, 'No hints are fed', C_WARN)
        self.put(row + 1, x + 2, 'into the AI.', C_WARN)

    def draw_world_panel(self, y: int, x: int, h: int, w: int, world: World) -> None:
        self.box(y, x, h, w, ' WORLD ')
        cell_w = 2
        needed_w = WORLD_W * cell_w + 2
        needed_h = WORLD_H + 2
        if w < needed_w or h < needed_h:
            self.put(y + 2, x + 2, 'WORLD PANEL TOO SMALL', C_BAD)
            return
        left = x + max(1, (w - needed_w) // 2) + 1
        top = y + max(1, (h - needed_h) // 2) + 1
        creature = world.creature
        self.put(top - 1, left - 1, '┌' + '─' * (WORLD_W * cell_w) + '┐', C_BORDER)
        for wy in range(WORLD_H):
            row_chars = ['│']
            row_pairs = [C_BORDER]
            for wx in range(WORLD_W):
                pos = (wx, wy)
                if pos == (creature.x, creature.y):
                    chars = '@@'
                    color = C_CREATURE
                elif pos in creature.body:
                    chars = 'oo'
                    color = C_MUTED
                elif pos in world.food:
                    chars = '**'
                    color = C_FOOD
                elif pos in world.obstacles:
                    chars = '##'
                    color = C_OBSTACLE
                else:
                    chars = '  '
                    color = C_DIM
                row_chars.append(chars)
                row_pairs.append(color)
            row_chars.append('│')
            for i, chars in enumerate(row_chars):
                if i == 0:
                    pair = C_BORDER
                    sx = left - 1
                elif i == len(row_chars) - 1:
                    pair = C_BORDER
                    sx = left - 1 + WORLD_W * cell_w
                else:
                    pair = row_pairs[i]
                    sx = left - 1 + i * cell_w
                self.put(top + wy, sx, chars, pair)
        self.put(top + WORLD_H, left - 1, '└' + '─' * (WORLD_W * cell_w) + '┘', C_BORDER)
        legend = '@ head   o body   * food   # obstacle'
        self.put(y + h - 2, x + 2, legend[:max(0, w - 4)], C_MUTED)

    def draw_brain_panel(self, y: int, x: int, h: int, w: int, world: World, brain: Brain) -> None:
        self.box(y, x, h, w, ' BRAIN ')
        state = world.get_state()
        values = brain.values(state)
        row = y + 2
        self.put(row, x + 2, 'Exploration', C_MUTED)
        self.put(row, x + 15, f'{brain.epsilon * 100:5.1f}%', C_WARN)
        row += 1
        self.put(row, x + 2, 'Known states', C_MUTED)
        self.put(row, x + 15, str(len(brain.q_table)), C_TITLE)
        row += 1
        self.put(row, x + 2, 'Updates', C_MUTED)
        self.put(row, x + 15, str(brain.total_updates), C_TITLE)
        row += 2
        self.put(row, x + 2, 'Q VALUES', C_TITLE, curses.A_BOLD)
        best_index = max(range(len(values)), key=lambda i: values[i])
        row += 1
        for action, value in enumerate(values):
            if row >= y + h - 8:
                break
            color = C_GOOD if action == best_index else C_TITLE
            self.put(row, x + 3, f'{ACTION_NAMES[action]:<6}', color)
            self.put(row, x + 12, f'{value:>9.3f}', color)
            normalized = clamp((value + 25.0) / 50.0, 0.0, 1.0)
            bar_width = max(6, w - 27)
            fill = int(normalized * bar_width)
            self.put(row, x + 23, '·' * fill, color)
            row += 1
        row += 1
        if row < y + h - 6:
            self.put(row, x + 2, 'RAW OBSERVATION', C_TITLE, curses.A_BOLD)
            row += 1
            sensor_count = 25
            grid = state[:sensor_count]
            symbols = {EMPTY: '.', FOOD: '*', OBSTACLE: '#', BODY: 'o', WALL: 'X'}
            for gy in range(5):
                line = ''
                for gx in range(5):
                    value = grid[gy * 5 + gx]
                    line += symbols.get(value, '?')
                self.put(row, x + 4, line, C_MUTED)
                row += 1
            if row < y + h - 2:
                self.put(row, x + 2, f'H={state[-2]}  HP={state[-1]}', C_SUBTITLE)

    def draw_log_panel(self, y: int, x: int, h: int, w: int, world: World) -> None:
        self.box(y, x, h, w, ' EVENT LOG ')
        available = max(0, h - 2)
        events = list(world.events)[-available:]
        row = y + 1
        for event in events:
            event_text = event[:max(0, w - 4)]
            lower = event.lower()
            if 'death' in lower or 'starvation' in lower or 'collision' in lower:
                pair = C_BAD
            elif 'food' in lower:
                pair = C_GOOD
            elif 'world' in lower:
                pair = C_SUBTITLE
            else:
                pair = C_MUTED
            self.put(row, x + 2, event_text, pair)
            row += 1
            if row >= y + h - 1:
                break

    def draw_controls(self, y: int, width: int) -> None:
        self.hline(y, 0, width, C_BORDER)
        controls = '[P] pause  [SPACE] step  [+/-] speed  [R] reset  [S] save  [C] clear log  [Q] quit'
        self.put(y + 1, 2, controls[:max(0, width - 4)], C_TITLE, curses.A_BOLD)
        if time.monotonic() < self.toast_until:
            toast = self.toast
            self.put(y + 1, max(2, width - len(toast) - 2), toast, C_GOOD)

    def draw(self, episode: int, world: World, brain: Brain, best_age: int, average_age: float) -> None:
        height, width = self.screen.getmaxyx()
        self.screen.erase()
        if width < 125 or height < 38:
            self.put(1, 2, 'SURVIVAL AI', C_TITLE, curses.A_BOLD)
            self.put(3, 2, f'Terminal: {width}x{height}', C_WARN)
            self.put(5, 2, 'Resize to at least 125x38 for the full interface.', C_MUTED)
            self.put(7, 2, 'q = quit', C_TITLE)
            self.screen.refresh()
            return
        self.draw_header(width, episode, brain, world, best_age)
        top = 3
        bottom_controls = 3
        main_height = height - top - bottom_controls
        left_w = 27
        right_w = 31
        center_w = width - left_w - right_w - 4
        main_top_h = 24
        if main_top_h > main_height:
            main_top_h = main_height
        log_y = top + main_top_h
        log_h = main_height - main_top_h
        self.draw_state_panel(top, 1, main_top_h, left_w, world)
        self.draw_world_panel(top, left_w + 2, main_top_h, center_w, world)
        self.draw_brain_panel(top, left_w + center_w + 3, main_top_h, right_w, world, brain)
        self.draw_log_panel(log_y, 1, log_h, width - 2, world)
        footer_y = height - bottom_controls
        status = f'episode={episode}  best={best_age}  avg50={average_age:0.1f}  speed={self.delay:0.3f}s'
        self.put(footer_y, 2, status[:max(0, width - 4)], C_SUBTITLE)
        self.draw_controls(footer_y + 1, width)
        self.screen.refresh()

    def handle_input(self, world: World, brain: Brain) -> Tuple[bool, bool]:
        """
        Returns:
            (running, single_step)
        """
        single_step = False
        while True:
            try:
                key = self.screen.getch()
            except curses.error:
                key = -1
            if key == -1:
                break
            if key in (ord('q'), ord('Q')):
                return (False, False)
            if key in (ord('p'), ord('P')):
                self.paused = not self.paused
                self.toast_message('paused' if self.paused else 'resumed')
            elif key == ord(' '):
                if self.paused:
                    single_step = True
            elif key in (ord('+'), ord('=')):
                self.delay = max(MIN_DELAY, self.delay * 0.75)
                self.toast_message(f'speed {self.delay:.3f}s')
            elif key in (ord('-'), ord('_')):
                self.delay = min(MAX_DELAY, self.delay * 1.25)
                self.toast_message(f'speed {self.delay:.3f}s')
            elif key in (ord('r'), ord('R')):
                world.reset()
                self.toast_message('world reset')
            elif key in (ord('s'), ord('S')):
                brain.save()
                self.toast_message('brain saved')
            elif key in (ord('c'), ord('C')):
                world.events.clear()
                self.toast_message('event log cleared')
        return (True, single_step)

def train(screen) -> None:
    ui = TUI(screen)
    world = World()
    brain = Brain()
    episode = 0
    best_age = 0
    recent_ages: Deque[int] = deque(maxlen=50)
    running = True
    while running:
        episode += 1
        world.reset()
        while running and world.creature.alive:
            running, single_step = ui.handle_input(world, brain)
            if not running:
                break
            if ui.paused and (not single_step):
                average_age = avg(recent_ages)
                ui.draw(episode, world, brain, best_age, average_age)
                time.sleep(0.03)
                continue
            state = world.get_state()
            action = brain.choose_action(state)
            reward = world.step(action)
            next_state = world.get_state()
            done = not world.creature.alive
            brain.learn(state, action, reward, next_state, done)
            average_age = avg(recent_ages)
            ui.draw(episode, world, brain, best_age, average_age)
            if not ui.paused:
                time.sleep(ui.delay)
        if not running:
            break
        age = world.creature.age
        recent_ages.append(age)
        if age > best_age:
            best_age = age
            ui.toast_message(f'NEW BEST: {best_age}')
        brain.decay()
        if episode % AUTOSAVE_EVERY == 0:
            try:
                brain.save()
                ui.toast_message('autosaved brain')
            except Exception:
                ui.toast_message('autosave failed')
        for _ in range(8):
            if not running:
                break
            running, _ = ui.handle_input(world, brain)
            average_age = avg(recent_ages)
            ui.draw(episode, world, brain, best_age, average_age)
            time.sleep(0.02)

def print_headless_status(label: str, episode: int, world: World, brain: Brain, best_age: int, average_age: float, total_foods: int) -> None:
    creature = world.creature
    print(f'[{label}] episode={episode} age={creature.age} health={creature.health} hunger={creature.hunger} foods={creature.foods_eaten} total_foods={total_foods} pos={creature.x},{creature.y} action={ACTION_NAMES[creature.last_action]} alive={creature.alive} score={creature.score:.2f} best={best_age} avg50={average_age:.1f} states={len(brain.q_table)} epsilon={brain.epsilon:.4f}', flush=True)

def train_headless() -> None:
    """
    Run at a steady, low-impact pace for overnight training.

    Nothing is rendered or logged. The brain is saved and its current state is
    printed once per minute.
    """
    world = World(record_events=False)
    brain = Brain()
    episode = 0
    best_age = 0
    recent_ages: Deque[int] = deque(maxlen=50)
    total_foods = 0
    next_report_at = time.monotonic() + HEADLESS_REPORT_INTERVAL
    try:
        while True:
            episode += 1
            world.reset()
            food_seen = 0
            while world.creature.alive:
                state = world.get_state()
                action = brain.choose_action(state)
                reward = world.step(action)
                next_state = world.get_state()
                done = not world.creature.alive
                brain.learn(state, action, reward, next_state, done)
                if world.creature.foods_eaten > food_seen:
                    food_seen = world.creature.foods_eaten
                    total_foods += 1
                now = time.monotonic()
                if now >= next_report_at:
                    brain.save()
                    print_headless_status('SAVED', episode, world, brain, best_age, avg(recent_ages), total_foods)
                    next_report_at = now + HEADLESS_REPORT_INTERVAL
                time.sleep(HEADLESS_STEP_DELAY)
            age = world.creature.age
            recent_ages.append(age)
            best_age = max(best_age, age)
            brain.decay()
    except KeyboardInterrupt:
        brain.save()
        print_headless_status('SAVED interrupted', episode, world, brain, best_age, avg(recent_ages), total_foods)

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Survival AI reinforcement-learning experiment')
    parser.add_argument('--no-ui', action='store_true', help='run headless at maximum speed with almost no output')
    args = parser.parse_args()
    if args.no_ui:
        train_headless()
        return
    if curses is None:
        parser.error('curses is unavailable; install windows-curses or run with --no-ui')
    curses.wrapper(train)
if __name__ == '__main__':
    main()
