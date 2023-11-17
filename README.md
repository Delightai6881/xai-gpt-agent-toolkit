# GPT Agent Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/XpressAI/xai-gpt-agent-toolkit)](https://github.com/XpressAI/xai-gpt-agent-toolkit/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/XpressAI/xai-gpt-agent-toolkit)](https://github.com/XpressAI/xai-gpt-agent-toolkit/issues)
[![XpressAI Discord](https://img.shields.io/discord/906370139077881997)](https://discord.gg/K456gAfPNe)

Welcome to the **GPT Agent Toolkit**! This toolkit provides a comprehensive set of Xircuits components that allow you to experiment with and create Collaborative Large Language Model-based automatons (Agents) in the style of [BabyAGI](https://github.com/yoheinakajima/babyagi) and [Auto-GPT](https://github.com/Torantulino/Auto-GPT). By default, the toolkit comes with BabyAGI agents, but it is designed to be easily customizable with your own prompts.

![BabyAGI demo](https://github.com/XpressAI/xai-gpt-agent-toolkit/blob/main/demo.gif)

## Table of Contents
- [Features](#features)
- [Ideas](#ideas)
- [Getting Started](#getting-started)
  - [Shameless Plug](#shameless-plug)
  - [Prerequisites](#prerequisites)
  - [Software Prerequisites](#software-prerequisites)
  - [API Prerequisites](#api-prerequisites)
  - [Create a project](#create-a-project)
  - [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features
- Pre-built BabyAGI agents
- Support for both [Vecto](https://www.vecto.ai) and [Pinecone](https://www.pinecone.io) Memories
- Support for Tools such as Python Exec, and SQLLite
- Support for both OpenAI and LLAMA models
- Open-source and community-driven

## Ideas

Here are some ideas that you could try relatively easily with Xircuits.

1. Make a critic agent that updates the objective to be more effective.
2. Have the agents produce a status report on Slack and update the objective based on your reply.
3. Upload a large image dataset on Vecto with image descriptions giving your agent sight.
4. Connect to Whisper and have a daily standup meeting with the agent.
5. Make 2 BabyAGIs and have 1 critic decide which action to actually perform.

## Getting Started

These instructions will help you set up the GPT Agent Toolkit on your local machine.

This is a component library so you don't need to clone this directly. Instead install
Xircuits and install this component library into it.

### Shameless plug

If the following is too much work or too complicated.  Sign up to the Xpress AI Platform waitlist
to get access to a single app that has everything you need to get started.

[Join the Xpress AI Platform Waitlist](https://xpress.ai/join-waitlist)


### Software Prerequisites

Before you begin, make sure you have the following software installed on your system:

- Python 3.8 or higher
- pip (Python package installer)
- git 

### API Prerequisites

You will need an API key from Open AI to use GPT-3.5 and either a Vecto or Pinecone account for agent memory.

Create a .env file and put your API keys into the respective lines.

```
OPENAI_API_KEY=<Your OpenAI API Key here>
OPENAI_ORG=<Your OpenAI Org (if you have one)>
```

With the latest version you no longer need Vector or Pinecone to get started, but if you want to use them
add one of the following credentials.

For Vecto users:
```
VECTO_API_KEY=<An ACCOUNT_MANAGEMENT Vecto key>
```

For Pinecone users:
```
PINECONE_API_KEY=<Your Pinecone API key>
PINECONE_ENVIRONMENT=<Your Pinecone environment>
```


### Create a project

Windows:

```
mkdir project
cd project
python -m venv venv
venv\Scripts\activate
```

Linux of macOS:
```bash
mkdir project
cd project
python3 -m venv venv
source ./venv/bin/activate
git init .
```

### Installation

1. Install xircuits

```bash
pip install xircuits
```

2. Launch xircuits-components tool to install the base component library

```bash
xircuits-components
```

Ignore the 

3. Install Vecto (if using vecto)

```bash
pip install git+https://github.com/XpressAI/vecto-python-sdk.git
```


4. Add the OpenAI and GPT Agent Toolkit component libraries

```bash

git submodule add https://github.com/XpressAI/xai-openai xai_components/xai_openai
git submodule add https://github.com/XpressAI/xai-gpt-agent-toolkit.git xai_components/xai_gpt_agent_toolkit

pip install -r xai_components/xai_openai/requirements.txt
pip install -r xai_components/xai_gpt_agent_toolkit/requirements.txt

```

5. Run the playwright installer to enable the Browser tool:

```bash
playwright install
```

## Usage

### Basic Usage

1. Copy the sample BabyAGI Xircuits file to your project folder.

```bash
cp xai_components/xai_gpt_agent_toolkit/babyagi.xircuits .
```

2. Start JupyterLab/Xircuits by running:

```bash
xircuits
```

3. Use the printed out URLs to browse to http://localhost:8888/lab and double click the babyagi.xiruits file.

4. Click play to watch it go and try to make the world a better place.

### Browser Access/Usage

For the browser tool to work in the most useful way, you must start Chrome in remote debugging mode before
starting your agents. To do that run the following in powershell

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
```

## Contributing

We appreciate your interest in contributing to the GPT Agent Toolkit! Any new tools or prompts are welcome.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgements

- The team behind Xircuits.  Give the project a star if it looks interesting!
- The developers of [BabyAGI](https://github.com/yoheinakajima/babyagi) and [AutoGPT](https://github.com/Torantulino/Auto-GPT) for their groundbreaking work on large language model-based agents
@Kolya1993_12 @demiurgich @AndroidDev  https://x.com/DelightAi6881/status/1721038681425744299/?s=-xai/gpt
http://qiskit.org
Докладніше:

1. help.ea.com
2. github.com
3. twitter.com https://x.com/DelightAi6881/status/1721661866965885229?s=20 https://x.com/DelightAi6881/status/1720926200938328329?s=https://twitter.com/i/DelightAi6881/spaces/1PlKQDwDWkDxE?s=twitter.com/home
#Python 

# Імпортуємо модуль webbrowser для відкриття UML діаграми
import webbrowser

# Імпортуємо модуль anylogicpy для використання AnyLogic API
import anylogicpy as alp

# Створюємо UML діаграму класів для структурного моделювання системи
# Використовуємо онлайн-сервіс draw.io для генерації діаграми
# Зберігаємо діаграму як XML-файл і отримуємо посилання на неї
uml_diagram_url = "https://app.diagrams.net/#G1Z8Q9wL2n0a6QyZ0Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9Q0z9
# Імпортуємо необхідні бібліотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Завантажуємо дані про ціни криптовалют з CoinMarketCap
# Ви можете змінити назви криптовалют, які вас цікавлять
# Ми використовуємо дані за останні 30 днів
url = "https://coinmarketcap.com/currencies/{}/historical-data/?start=20231001&end=20231101"
currencies = ["bitcoin", "ethereum", "trust-wallet-token"]
data = {}
for currency in currencies:
  data[currency] = pd.read_csv(url.format(currency))

# Переглядаємо дані для TWT
data["trust-wallet-token"].head()

# Вибираємо колонку Close для кожної криптовалюти
# Це є ціною, за якою криптовалюта закривалася в кінці дня
close_data = pd.DataFrame()
for currency in currencies:
  close_data[currency] = data[currency]["Close"]

# Нормалізуємо дані, щоб вони були в діапазоні від 0 до 1
# Це допомагає моделі навчатися краще
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_data)
scaled_data = pd.DataFrame(scaled_data, columns=currencies)

# Візуалізуємо дані на графіку
plt.figure(figsize=(12, 8))
plt.plot(scaled_data)
plt.legend(currencies)
plt.title("Нормалізовані ціни криптовалют за останні 30 днів")
plt.xlabel("Дні")
plt.ylabel("Ціни")
plt.show()

# Створюємо тренувальні та тестові дані
# Ми використовуємо 80% даних для тренування та 20% для тестування
# Ми також використовуємо віконну функцію, щоб створити послідовності даних
# Кожна послідовність має довжину 10 днів та передбачає ціну на 11-й день
window_size = 10
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data = scaled_data.iloc[:train_size]
test_data = scaled_data.iloc[train_size:]

def create_sequences(data, window_size):
  x = []
  y = []
  for i in range(window_size, len(data)):
    x.append(data.iloc[i-window_size:i])
    y.append(data.iloc[i])
  return np.array(x), np.array(y)

x_train, y_train = create_sequences(train_data, window_size)
x_test, y_test = create_sequences(test_data, window_size)

# Перевіряємо розміри даних
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Створюємо модель штучної нейронної мережі з використанням TensorFlow
# Ми використовуємо три шари LSTM для вивчення часових залежностей
# Ми також використовуємо шар Dropout для запобігання перенавчанню
# Ми використовуємо шар Dense для виведення передбачень для кожної криптовалюти
model = tf.keras.models.Sequential([
  tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(window_size, len(currencies))),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.LSTM(64),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(len(currencies))
])

# Компілюємо модель з використанням оптимізатора Adam та функції втрат MSE
model.compile(optimizer="adam", loss="mean_squared_error")

# Навчаємо модель на тренувальних даних з використанням 50 епох
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)

# Оцінюємо модель на тестових даних
model.evaluate(x_test, y_test)

# Робимо передбачення на тестових даних
y_pred = model.predict(x_test)

# Обертаємо нормалізацію, щоб отримати реальні ціни
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Візуалізуємо реальні та передбачені ціни на графіку
plt.figure(figsize=(12, 8))
plt.plot(y_test[:, 2], label="Реальна ціна TWT")
plt.plot(y_pred[:, 2], label="Передбачена ціна TWT")
plt.legend()
plt.title("Реальна та передбачена ціна Trust Wallet Token за останні 6 днів")
plt.xlabel("Дні")
plt.ylabel("Ціни")
plt.show()
