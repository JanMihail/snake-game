import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


# =====================================================================
# 1. СЕТЬ (Архитектура "мозга" на PyTorch)
# Ей всё равно, какая среда. Она просто превращает вектор состояния в оценки действий.
# =====================================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)  # На выходе по оценке Q для каждого действия
        )

    def forward(self, x):
        return self.network(x)


# =====================================================================
# 2. АГЕНТ (Логика выбора действий и обучения на PyTorch)
# Этот класс будет одинаковым для Змейки, Робота или Трейдинга.
# =====================================================================
class DQNChainAgent:
    def __init__(self, state_dim: int, action_dim: int, lr=0.001, gamma=0.99, epsilon=0.1):
        self.action_dim = action_dim
        self.gamma = gamma  # Коэффициент дисконтирования (наш 0.9)
        self.epsilon = epsilon  # Вероятность случайного шага

        # Инициализируем модель и оптимизатор
        self.model = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def choose_action(self, state: np.ndarray) -> int:
        """Выбор действия: либо случайно, либо по уму нейросети"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        # Переводим состояние в тензор PyTorch
        state_t = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.model(state_t)
            return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state, done):
        """Один шаг градиентного спуска по формуле Белмана"""
        # Превращаем всё в тензоры PyTorch
        state_t = torch.tensor(state, dtype=torch.float32)
        next_state_t = torch.tensor(next_state, dtype=torch.float32)
        reward_t = torch.tensor(reward, dtype=torch.float32)

        # 1. Получаем текущее предсказание сети для сделанного действия
        current_q = self.model(state_t)[action]

        # 2. Считаем идеальный ответ (Target Q), заглядывая в будущее
        with torch.no_grad():
            if done:
                target_q = reward_t
            else:
                max_next_q = torch.max(self.model(next_state_t))
                target_q = reward_t + self.gamma * max_next_q

        # 3. Классический шаг обучения PyTorch
        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# =====================================================================
# 3. ГЛОБАЛЬНЫЙ ДВИЖОК ОБУЧЕНИЯ (Главный запуск)
# Сюда ты можешь подставить любую свою среду, и всё будет работать
# =====================================================================
def train_agent(env, episodes=500):
    # Динамически берем размеры пространств прямо из настроек среды
    state_dim = env.observation_space.shape[0]  # Сколько чисел описывают мир
    action_dim = env.action_space.n  # Сколько кнопок можно нажать

    # Создаем нашего универсального агента
    agent = DQNChainAgent(state_dim, action_dim)

    for episode in range(episodes):
        state, _ = env.reset()  # Стандартный сброс среды (совместим с Gymnasium)
        done = False
        total_reward = 0

        while not done:
            # Агент выбирает действие
            action = agent.choose_action(state)

            # Среда делает шаг
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Игра закончилась по любой причине

            # Агент обучается внутри своего класса через PyTorch
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        if (episode + 1) % 50 == 0:
            print(f"Эпизод {episode + 1} | Награда за игру: {total_reward:.2f}")

    return agent


class MyCustomGame:
    def __init__(self):
        # Обязательно описываем свойства, чтобы наш движок понял размеры матриц
        self.observation_space = ...  # Например, массив из 4 чисел (координаты)
        self.action_space = ...  # Например, дискретное пространство из 3 кнопок

    def reset(self):
        # Логика сброса твоей игры
        return initial_state, {}

    def step(self, action):
        # Твоя математика игры: подвигаться, проверить коллизии, посчитать награду
        return next_state, reward, terminated, truncated, {}


if __name__ == '__main__':
    env = MyCustomGame()
    trained_brain = train_agent(env, episodes=1000)