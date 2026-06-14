import random
import tkinter as tk
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List
import copy


# ==============================================================================
# 1. СРЕДА ИГРЫ (8 направлений, разделение стены/тела)
# ==============================================================================
class SnakeEnv:
    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        self.reset()

    def reset(self) -> List[float]:
        # Спавним змейку так, чтобы сзади было место
        self.snake: List[Tuple[int, int]] = [
            (self.height // 2, self.width // 2),
            (self.height // 2, self.width // 2 + 1),
            (self.height // 2, self.width // 2 + 2)
        ]
        self.direction = 3  # 0=ВВЕРХ, 1=ВПРАВО, 2=ВНИЗ, 3=ВЛЕВО
        self.score = 0
        self.done = False
        self.food: Tuple[int, int] = self._place_food()
        self.steps_without_food = 0
        return self._get_state()

    def _place_food(self) -> Tuple[int, int]:
        while True:
            food = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
            if food not in self.snake:
                return food

    def _count_reachable_cells(self) -> int:
        """ BFS для проверки, не заперта ли змейка """
        if not self.snake: return 0
        head = self.snake[0]
        obstacles = set(self.snake[:-1])  # Хвост может уйти, его не считаем жестким блоком

        queue = deque([head])
        visited = {head}

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if (nr, nc) not in obstacles and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return len(visited) - 1

    def _get_state(self) -> List[float]:
        hr, hc = self.snake[0]

        # 8 направлений: N, NE, E, SE, S, SW, W, NW
        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]

        state = []

        # Для каждого направления ищем расстояние до стены и до тела
        for dr, dc in directions:
            r, c = hr + dr, hc + dc
            dist = 1.0
            wall_dist = 0.0
            body_dist = 0.0

            while 0 <= r < self.height and 0 <= c < self.width:
                if (r, c) in self.snake and body_dist == 0.0:
                    body_dist = 1.0 / dist  # Близость к телу
                r += dr
                c += dc
                dist += 1.0

            wall_dist = 1.0 / dist  # Близость к стене
            state.extend([wall_dist, body_dist])

        # Направление движения еды относительно головы (4 признака)
        state.append(1.0 if self.food[0] < hr else 0.0)  # Еда выше
        state.append(1.0 if self.food[0] > hr else 0.0)  # Еда ниже
        state.append(1.0 if self.food[1] < hc else 0.0)  # Еда левее
        state.append(1.0 if self.food[1] > hc else 0.0)  # Еда правее

        return state  # Итого: 8*2 + 4 = 20 признаков

    def step(self, action: int) -> Tuple[List[float], float, bool]:
        if self.done:
            return self._get_state(), 0.0, True

        # Запрет разворота в себя
        if abs(self.direction - action) != 2:
            self.direction = action

        hr, hc = self.snake[0]
        if self.direction == 0:
            hr -= 1
        elif self.direction == 1:
            hc += 1
        elif self.direction == 2:
            hr += 1
        elif self.direction == 3:
            hc -= 1

        new_head = (hr, hc)
        self.steps_without_food += 1

        # Считаем расстояние до еды ДО шага
        old_dist = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])

        # Условия смерти
        max_steps = 100 + len(self.snake) * 4  # Увеличим лимит для больших размеров
        if (hr < 0 or hr >= self.height or hc < 0 or hc >= self.width or
                new_head in self.snake or self.steps_without_food > max_steps):
            self.done = True
            return self._get_state(), -20.0, True  # Штраф умеренный, чтобы не забивать другие Q-значения

        self.snake.insert(0, new_head)

        # Считаем расстояние ПОСЛЕ шага
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
            self.steps_without_food = 0
            reward = 15.0  # Стимул расти
        else:
            self.snake.pop()
            # Бонус за приближение к еде, штраф за удаление
            reward = 0.2 if new_dist < old_dist else -0.3

        # Проверка на тупик через BFS
        if self._count_reachable_cells() == 0:
            reward -= 5.0  # Штраф за попадание в глухой капкан

        return self._get_state(), reward, False


# ==============================================================================
# 2. АРХИТЕКТУРА НЕЙРОСЕТИ DQN (Входной размер 20)
# ==============================================================================
class QNetwork(nn.Module):
    def __init__(self, input_size=20, output_size=4):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.fc(x)


# ==============================================================================
# 3. СТАБИЛЬНОЕ ОБУЧЕНИЕ (DQN + Target Network)
# ==============================================================================
def train_dqn(epochs: int = 1500) -> QNetwork:
    print("-> Запуск стабильного DQN...")
    env = SnakeEnv()

    # Основная и Целевая сети
    model = QNetwork()
    target_model = copy.deepcopy(model)
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Чуть уменьшим LR для стабильности
    loss_fn = nn.MSELoss()

    memory = deque(maxlen=50000)  # Увеличим буфер памяти

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.997  # Плавное затухание

    batch_size = 64
    gamma = 0.98  # Больше смотрим в будущее
    target_update_freq = 10  # Обновляем target сеть каждые 10 эпох

    for epoch in range(epochs):
        state = env.reset()

        while not env.done:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state)
                    action = torch.argmax(model(state_t)).item()

            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state

            if len(memory) > batch_size:
                minibatch = random.sample(memory, batch_size)

                states_b = torch.FloatTensor([m[0] for m in minibatch])
                actions_b = torch.LongTensor([m[1] for m in minibatch]).unsqueeze(1)
                rewards_b = torch.FloatTensor([m[2] for m in minibatch])
                next_states_b = torch.FloatTensor([m[3] for m in minibatch])
                dones_b = torch.FloatTensor([m[4] for m in minibatch])

                current_q = model(states_b).gather(1, actions_b).squeeze()

                # ИСПОЛЬЗУЕМ TARGET_MODEL ДЛЯ РАСЧЕТА БУДУЩИХ НАГРАД
                with torch.no_grad():
                    max_next_q = target_model(next_states_b).max(1)[0]
                    target_q = rewards_b + (1 - dones_b) * gamma * max_next_q

                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Синхронизация сетей
        if (epoch + 1) % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        if (epoch + 1) % 100 == 0:
            # Тестовый запуск без epsilon для чистой метрики
            print(f"Эпоха {epoch + 1}/{epochs} | Epsilon: {epsilon:.3f} | Счёт: {env.score}")

    print("-> Обучение успешно завершено!")
    torch.save(model.state_dict(), "snake_dqn_model.pth")
    return model


class SnakeVisualizer:
    def __init__(self, env: SnakeEnv, model_path: str, cell_size: int = 35):
        self.env = env
        self.cell_size = cell_size

        self.model = QNetwork()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.root = tk.Tk()
        self.root.title("Финальный результат DQN модели")

        self.info_label = tk.Label(self.root, text="Анализ поля...", font=("Arial", 14))
        self.info_label.pack()

        self.canvas = tk.Canvas(
            self.root,
            width=self.env.width * self.cell_size,
            height=self.env.height * self.cell_size,
            bg="#1e1e1e"
        )
        self.canvas.pack()

        self.state = self.env.reset()
        self.run_step()

    def draw(self):
        self.canvas.delete("all")
        self.info_label.config(text=f"Счёт: {self.env.score}")

        fr, fc = self.env.food
        self.canvas.create_oval(
            fc * self.cell_size + 4, fr * self.cell_size + 4,
            (fc + 1) * self.cell_size - 4, (fr + 1) * self.cell_size - 4,
            fill="#FF5555", outline=""
        )

        for i, (r, c) in enumerate(self.env.snake):
            color = "#db7b04" if i == 0 else "#00AA44"
            self.canvas.create_rectangle(
                c * self.cell_size, r * self.cell_size,
                (c + 1) * self.cell_size, (r + 1) * self.cell_size,
                fill=color, outline="#1e1e1e"
            )

    def run_step(self):
        if self.env.done:
            self.info_label.config(text=f"Game over! Score: {self.env.score}")
            return

        with torch.no_grad():
            state_t = torch.FloatTensor(self.state)
            action = torch.argmax(self.model(state_t)).item()

        self.state, _, _ = self.env.step(action)
        self.draw()
        self.root.after(50, self.run_step)  # Чуть замедлил шаг для читаемости глазами

    def start(self):
        self.root.mainloop()


if __name__ == "__main__":
    # 1. Раскомментируйте строчку ниже для обучения (хватит ~1500-2000 эпох):
    # train_dqn(epochs=2000)

    # 2. Визуализация:
    test_env = SnakeEnv(width=10, height=10)
    visualizer = SnakeVisualizer(test_env, model_path="snake_dqn_model.pth")
    visualizer.start()