# Candy Crush AI With Built-in Game Predictor

## Statement of Contributions

During the completion of this project, each group member made significant contribution, with
the amount of contribution being roughly equal.

The division of the work is as follows:

Avani A.: Proposal, Report, Game predictor agent

Ava B.: Report

Azad K.: Game playing agent, task environment

## Introduction

**Background, Motivation & Objectives**
The Candy Crush Saga is a well-known mobile game that involves swapping candies to create
combinations. When combinations are made, the matched candies disappear and new ones
appear on the board, creating more scoring opportunities. The goal of Candy Crush, or any
similar match 3 game, is to maximize the score with a limited amount of moves by making these
matches, making it ideal for reinforcement learning, as the agent can be trained to optimize
rewards over time.

Our utility agent was trained to evaluate board states and learn through short-term and long-term
memory to train it to maximize score. Since each move alters the game board, the agent must use
its long-term training to determine the best outcome.

To make the premise more unique, the project also involves an in-game results predictor which
displays whether the agent is winning or losing. This secondary part uses fuzzy-based logic to
estimate this with the number of moves left and the number of points obtained. This predictor
agent along with the playing agent adds another layer of strategy to our project by executing
optimal moves as well as predicting the outcome. This dual agent makes the implementation and
project more novel.


**Related Prior Work**
Before building the agent, prior work required was to develop a simulator of the Candy Crush
game and research more on the methods used in the project. The simulator was developed using
existing code on Github which was adapted to work with our agent implementations
(theharrychen, 2019).

There have been multiple agents developed to play Candy Crush Saga using reinforcement
learning. For example, a student from Stanford University used a convolutional neural network
with a similar architecture, as well as the Monte Carlo reward (Deng, 2020). King, the company
behind Candy Crush Saga, also uses a game-playing agent to consistently improve the game,
update older levels, and create new ones (The Associated Press, 2025).

## Methods

**Artificial Intelligence Methods Used**
The game-playing agent uses reinforcement learning and neural networks to optimize the score
and improve gameplay. It uses temporal difference Q-learning and active reinforcement learning
using the Bellman equation. It also uses a convolutional neural network for input processing.

The neural network uses the ReLU activation function, with the input consisting of the current
state of the board. The input is passed into the neural network using one-hot encoding, which
converts the state of the board into a binary representation (Geeksforgeeks, 2025). The input
layer consists of a series of matrices, each one representing a different symbol on the board. The
matrix contains a value of 1 where that symbol is present, and a 0 where a different symbol is
present. The hidden layers of the neural network process the input to pinpoint patterns such as
matches of three or more, or other combinations that affect the game and improve the score. The
output layer returns a possible action that the agent can take.

The game playing agent plays multiple games to compute the Q values of state and action pairs.
The reward function heavily reinforces the clearing of tiles by multiplying the reward of the
gained tiles by 20. If the action does not change the board or clear any tiles, the agent receives a
punishment of -0.5. Upon game end, 250 is added to the reward if the point goal is reached,
otherwise 25 is subtracted. The values for the reward function were chosen to improve
performance and strongly reinforce good behaviour while punishing bad behaviour. The reward
function is as follows:


+250 if game over and score is met
Q(s,a) = { 20 * tile gain + { - 25 if game over and score is not met
-0.5 otherwise 0 otherwise

Fuzzy rule-based systems: The built-in game predictor agent uses fuzzy-based rule systems to
determine whether the agent may win or lose the game. This fuzzy rule-based system uses a
triangular membership function and the Godel t and s norms. The parameters are taken from the
values in the game. Moves represents the initial number of moves left at the beginning of the
game and goal is the goal score the player needs to hit to win.
The fuzzy rule-based system is as follows:

IF score is low AND moves is low, THEN likely outcome is LOSE
IF score is high OR moves is high, THEN likely outcome is WIN

Parameters:
Score low: 𝑎= 0 , 𝑏= 𝑔𝑜𝑎𝑙/4, 𝑐= 𝑔𝑜𝑎𝑙/2
Score high: 𝑎= 𝑔𝑜𝑎𝑙/2 𝑎𝑙, 𝑏=𝑔𝑜𝑎𝑙* 0.75 , 𝑐=𝑔𝑜𝑎𝑙
Moves low: 𝑎= 0 , 𝑏= 𝑚𝑜𝑣𝑒𝑠/4, 𝑐= 𝑚𝑜𝑣𝑒𝑠/2
Moves high: 𝑎= 𝑚𝑜𝑣𝑒𝑠/2, 𝑏=𝑚𝑜𝑣𝑒𝑠* 0. 75 , 𝑐=𝑚𝑜𝑣𝑒𝑠

**Dataset or Task Environment**
Since the agent is a dual agent, one of them requires a dataset and one of them requires a task
environment. For the game playing agent, the task environment of the game simulation is
required. Since the game playing agent decides actions to change the task environment, the
simulation of the candy crush game is needed. However, the game predictor agent requires a data
set to perform calculations. The data set required consists of the current score, as well as the
remaining number of moves. These values are used by the agent to compute a likely outcome
and output it.

A dataset of matrices is also required for the game playing agent, since they are part of the input
state. The board is converted into a tensor with several channels by one-hot encoding each
“candy type” into its own channel, with a 1 indicating the presence of a specific symbol at a
position and 0 indicating the absence (Dive into Deep Learning, n.d.). The agent also receives
several scalar game features; current score, number of turns left, goal score, combination count,
and a game-over flag. These are spread out across the entire board as extra channels so that they
can be processed alongside the tile information.The agent then chooses an action based on the
representation received.


The game playing agent and the game predicting agent have differences in their task
environments due to their different uses and actions. The task environment is a fully observable
environment, since the entire game is visible and the agent is able to see where each symbol is on
the board. The agents also have access to all the values such as the score. The task environment
is multiple agent, since we implemented a dual agent variation of the problem with both the
game playing and game predicting agents. The task environment is deterministic since the
agent’s actions influence the current state (such as which symbols are left on the board). The
game predictor agent also does not have any randomness, and the game playing agent’s actions
influence the predicting agent’s current state. The task environment is known since both agents
are aware of the output of all actions they can take.

There are 3 key differences between each agent’s task environment. While the game predictor’s
task environment is episodic (since previous calculations do not affect the current one), the game
playing agent’s task environment is sequential since the agent’s actions influence future ones (an
action changes the layout of the board and affects the next action). While the game playing agent
is deliberating, the environment does not change, making it a static environment. For this agent,
the actions are confined to discrete variables (a set number of directions a pearl can be swiped)
making it a discrete task environment. However, for the game predicting agent, inputs are not
confined to discrete variables, making it a continuous task environment. Since the game is a part
of the game predictor’s task environment, it is dynamic. While the agent is calculating, the
environment may change and the agent will then have to choose its next action based on the
sudden change of environment.

**Validation Strategy/Experiments**
Our primary validation strategy was using a random agent as a comparison to ensure our agent
was improving its score. This other agent made random moves every round and was used as a
baseline so the agent’s performance could be validated against a randomized action picker. Using
Matplotlib, our program graphed the results, giving us a visual representation of our agent’s
success. We also graphed the average reward and average score, ensuring it increased with more
games played. As well as this, we used set seed testing, meaning that our starting variables were
initialized instead of choosing random variables (Geeksforgeeks, 2025). This caused the game
board to start with the same tiles, so that the agent was able to learn on the same board. The
game-predictor agent was tested on a series of diverse cases using hold out validation, where
separate groups of test cases were used for validating and testing.


## Results

**Qualitative Results**
During testing, our agent demonstrated a significant improvement from its first game to its last.
In the first game played by the agent, it only makes random moves. This is because it has not
learned anything yet, and will have poor results due to the frequent incorrect matches, resulting
in a -0.5 deduction. As the simulation continues, the agent slowly begins to learn how to play,
making correct swaps and winning more frequently. The fuzzy-based rule agent accurately
predicted the outcome of the game, updating the guess each time a move was made. Overall,
both agents showed positive results, the game-playing one improving its strategy every game,
and the predictor agent accurately guessing the correct outcome.

**Quantitative Results**
<img width="2559" height="1390" alt="image" src="https://github.com/user-attachments/assets/849968f0-6fa0-439d-93fe-5895216c41cc" />

Pictured above are the results of running our simulation for 300 games. The top graph shows a
comparison of our agent and a random baseline agent, who both begin with similar results. Just
before game 50, the agent’s score began to improve steadily. The comparison to the baseline
agent further proves our agent’s ability to learn. The bottom graph shows the total reward per
game for just the agent, which displays our agent’s improvement. While the agent’s gameplay
began as purely exploratory, the reinforcement learning approach successfully increased
long-term reward outcomes, confirming that the learned policy outperforms random move
selection.


## Discussion

**Limitations and Future Directions**
Although our agent demonstrated successful results and properly predicted the win, there were
areas where the current implementation could be improved.

First, our board was a static one, meaning the initial state of the board was the same every time a
new game was started. This limits our agent to playing the same game every time, meaning it
won’t learn multiple situations. In the future, we would like to add dynamic boards, giving the
game different levels with varying complexity, size, and shape.

Another future direction for the game would be to add more diverse game mechanics. Currently,
our game has the same number of points for matches, not taking game mechanics–such as
explosion candies and number of matches in a row–into account. Ideally, our agent should be
trained to look for these special cases, and should know how to handle them. Adding this would
help train the agent further and improve the game complexity.

Overall, future versions of our system would aim to increase game complexity, add different
levels with varying difficulty, and add a more complicated reward system. These improvements
would not only improve the agent, but would also deepen our understanding of artificial
intelligence as a whole.

**Implications**
Our Candy Crush system shows how valuable dual-agent systems can be. While one agent
executed decisions, the other interpreted outcomes in real time. The combination of these two
agents can be used elsewhere, such as in monitoring systems or decision support tools that help
predict the user’s next word as well as learning their writing style. These agents can be used
together in any situation where both prediction and outcome are important. Overall, our work
highlights that future AI systems would benefit from this combination of agents.


## References

“6.4. Multiple Input and Multiple Output Channels — Dive into Deep Learning 0.17.

```
Documentation.” Classic.d2l.ai , 2025,
classic.d2l.ai/chapter_convolutional-neural-networks/channels.html. Accessed 5 Dec.
2025.
```
Deng, Bowen. _Learn to Play Candy Crush_.

```
cs230.stanford.edu/projects_winter_2020/reports/32031465.pdf. Accessed 5 Dec. 2025.
```
GeeksforGeeks. “One Hot Encoding in Machine Learning.” _GeeksforGeeks_ , 12 June 2019,

```
http://www.geeksforgeeks.org/machine-learning/ml-one-hot-encoding/. Accessed 5 Dec. 2025.
```
GeeksforGeeks. “Python - random.seed( ) method.” _GeeksforGeeks_ , 11 July 2025,

```
https://www.geeksforgeeks.org/python/random-seed-in-python/. Accessed 5 Dec. 2025.
```
Karnsund, Alice. “DQN Tackling the Game of Candy Crush Friends Saga : A Reinforcement

```
Learning Approach.” DEGREE PROJECT in the FIELD of TECHNOLOGY
ENGINEERING PHYSICS and the MAIN FIELD of STUDY COMPUTER SCIENCE and
ENGINEERING, SECOND CYCLE, 30 CRED , Jan. 2019. Accessed 5 Dec. 2025.
```
Loeber, Patrick. “Teach AI to Play Snake! Reinforcement Learning with PyTorch and Pygame.”

```
GitHub , 7 Jan. 2023, github.com/patrickloeber/snake-ai-pytorch. Accessed 5 Dec. 2025.
```
Press, The Associated. “How AI Helps Push Candy Crush Players through Its Most Difficult

```
Puzzles.” CTVNews , 11 May 2025,
http://www.ctvnews.ca/sci-tech/article/how-ai-helps-push-candy-crush-players-through-its-mos
t-difficult-puzzles/. Accessed 5 Dec. 2025.
```

theharrychen. “GitHub - Theharrychen/Element-Crush: Match 3 Tiles Game Built in Python with

```
Pygame.” GitHub , 17 Sept. 2019, github.com/theharrychen/Element-Crush. Accessed 5
```

