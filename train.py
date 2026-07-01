from agent import PolicyNetwork
import matplotlib.pyplot as plt
import gymnasium
import torch

env = gymnasium.make("CartPole-v1")

policy = PolicyNetwork()

optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

all_rewards = []

for episode in range(500):
    observation, info = env.reset()
    log_probs = []
    rewards = []
    for t in range(500):
        state = torch.FloatTensor(observation)
        logits = policy(state)
        probs = torch.softmax(logits, dim= -1)
        action = torch.multinomial(probs, 1).item()

        observation, reward, terminated, truncated, info = env.step(action)
        
        log_prob = torch.log(probs[action])
        log_probs.append(log_prob)
        rewards.append(reward)

        if terminated or truncated:
            break
    
    returns = []
    G = 0
    gamma = 0.99
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss -= log_prob * G

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Episode {episode}, Total Reward: {sum(rewards)}")
    
    all_rewards.append(sum(rewards))



plt.plot(all_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("CartPole Learning Curve")
plt.savefig("learning_curve.png")
plt.show()