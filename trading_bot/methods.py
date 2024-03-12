import os
import logging

import numpy as np

from tqdm import tqdm

from .utils import format_currency, format_position
from .ops import get_state

def action_int_2_str(action : int) -> str:
    """Converts action into to str, should use enums"""
    if action == 1 or action == 2:
        return 'BUY' if action ==1 else 'SELL'
    return 'HOLD'

def take_action(action, agent, data_point, total_profit, reward):
    delta = 0
    if action == "BUY":
        agent.inventory.append(data_point)

    elif action == "SELL" and len(agent.inventory) > 0:
        bought_price = agent.inventory.pop(0)
        delta = data_point - bought_price
        reward = delta 
        
    return reward, total_profit + delta
    
def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=10):
    """This function trains the model""" 
    total_profit = 0
    data_length = len(data) - 1

    agent.clear_inventory()
    avg_loss = []

    state = get_state(data, 0, window_size + 1)

    for t in tqdm(range(data_length), total=data_length, leave=True, desc=f'Episode {episode}/{ep_count}'):        
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)

        action = action_int_2_str(agent.act(state, is_eval=True))
        reward, total_profit = take_action(action, agent, data[t], total_profit, reward)        
        history.append((data[t], action))


        done = (t == data_length - 1)
        agent.remember(state, action, reward, next_state, done)

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    if episode % 10 == 0:
        agent.save(episode)

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, window_size):
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.clear_inventory()

    state = get_state(data, 0, window_size + 1)

    for t in range(data_length):        
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)
        
        action = action_int_2_str(agent.act(state, is_eval=True))
        reward, total_profit = take_action(action, agent, data[t], total_profit, reward)        
        history.append((data[t], action))

        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            return total_profit, history
