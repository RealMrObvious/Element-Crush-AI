from match3 import match3AI
import numpy as np
from collections import deque
from agent import Agent
import matplotlib.pyplot as plt
from IPython import display
import game_predictor_agent as gp

#Plotting code based on https://github.com/patrickloeber/snake-ai-pytorch/blob/main/helper.py
plt.ion()

def plot(scores, mean_scores,
         reward_sums=None, mean_reward_sums=None,
         random_scores=None, random_mean_scores=None):
    
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    #plot the scores for agent and the full random 'agent'
    plt.subplot(2, 1, 1)
    plt.title('Score per Game (Agent vs Random Baseline)')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    # Agent curves
    plt.plot(scores, label='Agent Score')
    plt.text(len(scores)-1, scores[-1], f"{scores[-1]:.1f}", fontsize=10, ha='left', va='center') 
    plt.plot(mean_scores, label='Agent Mean Score')
    plt.text(len(mean_scores)-1, mean_scores[-1], f"{mean_scores[-1]:.1f}", fontsize=10, ha='left', va='center') 

    plt.plot(random_scores, label='Random Score', linestyle='--')
    plt.text(len(random_scores)-1, random_scores[-1], f"{random_scores[-1]:.1f}", fontsize=10, ha='left', va='center')     

    plt.plot(random_mean_scores, label='Random Mean Score', linestyle=':')
    plt.text(len(random_mean_scores)-1, random_mean_scores[-1], f"{random_mean_scores[-1]:.1f}", fontsize=10, ha='left', va='center')   
        

    plt.ylim(ymin=0)

    plt.legend()

    #plot rewards for the agent
    if reward_sums is not None and mean_reward_sums is not None:
        plt.subplot(2, 1, 2)
        plt.title('Total Reward per Game (Agent Only)')
        plt.xlabel('Number of Games')
        plt.ylabel('Reward Sum')
        plt.plot(reward_sums, label='Reward Sum')
        plt.plot(mean_reward_sums, label='Mean Reward Sum')

        if reward_sums:
            plt.text(len(reward_sums)-1, reward_sums[-1], str(round(reward_sums[-1],1)))
        if mean_reward_sums:
            plt.text(len(mean_reward_sums)-1, mean_reward_sums[-1], str(round(mean_reward_sums[-1],1)))

        plt.legend()

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def main():
    headless = False

    # Used to track scores for graph for DQN agent
    plot_scores = []
    plot_mean_scores = []
    plot_reward_sums = []
    plot_mean_reward_sums = []
    # ^ but for random agent
    randomplot_scores = []
    randommean_scores = []
    rewards = []

    #Game params
    record = 0
    width = 7
    height = 9
    turns = 30
    goal = 450

    #agents/games
    agent = Agent(width,height)
    game = match3AI(width,height,turns,goal,headless)
    randomgame = match3AI(width,height,turns,goal,headless=True)
    prediction = ""

    while True:
        prev_state = game.getState()
        prev_state_encoded = agent.process_state(prev_state)
        state_tensor = prev_state_encoded.unsqueeze(0) 

        agent_move, action_id = agent.pick_action(state_tensor)
        prediction = gp.predict_game(goal, turns, prev_state["score"], prev_state["turns_left"])        #Predict game

        game.set_prediction(prediction)
        game.playStep(agent_move)
        
        new_state = game.getState()
        reward = agent.reward(prev_state,new_state)
        rewards.append(reward)

        new_state_encoded = agent.process_state(new_state)
        next_state_tensor = new_state_encoded.unsqueeze(0)
        
        # Train using action_id
        agent.train_short(state_tensor, action_id, reward, next_state_tensor, new_state["gameover"])

        # Store transition with action_id
        agent.memory.append((prev_state_encoded, action_id, reward, new_state_encoded, new_state["gameover"]))

        random_action, __ = agent.pick_action(state_tensor,full_random=True) #doesn't actually affect the agent, just re-uses its code
        randomgame.playStep(random_action)

        random_state = randomgame.getState()

        if random_state["gameover"]:
            randomplot_scores.append(random_state["score"])
            random_mean = np.mean(randomplot_scores)
            randommean_scores.append(random_mean)

            print(f"[Random] Score: {random_state['score']}  Mean: {round(random_mean, 2)}")

            randomgame.reset(turns,goal)

        if new_state["gameover"]:
            game.reset(turns,goal)
            agent.num_games += 1

            agent.train_long()

            if new_state["score"] > record:
                record = new_state["score"]
                agent.model.save()

            plot_scores.append(new_state["score"])
            mean_score = np.mean(plot_scores)
            plot_mean_scores.append(mean_score)
            
            reward_sum = sum(rewards)  # sum of all per-step rewards in this game
            plot_reward_sums.append(reward_sum)
            mean_reward_sum = np.mean(plot_reward_sums)
            plot_mean_reward_sums.append(mean_reward_sum)
            rewards.clear()  # reset for next game

            print('Game', agent.num_games, 'Highscore:', record)
            print(f"[Agent] Score: {plot_scores[-1]} Mean Score: {plot_mean_scores[-1]} Reward: {plot_reward_sums[-1]} Mean Reward: {plot_mean_reward_sums[-1]}")
            if not headless:
                plot(plot_scores, plot_mean_scores,
                    plot_reward_sums, plot_mean_reward_sums,
                    randomplot_scores, randommean_scores)

    


        
if __name__ == "__main__": main()