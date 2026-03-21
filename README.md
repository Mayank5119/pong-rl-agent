# Pong OpenEnv - Complete Learning Project for Hackathon Prep

Train an RL agent using **PyTorch**, manage models with **Hugging Face Hub**, and deploy on **Hugging Face Spaces**.

## 🎯 What You'll Learn

This project teaches the **entire ML/AI ecosystem**:

```
┌─────────────────────────────────────────┐
│  COMPLETE LEARNING ECOSYSTEM            │
├─────────────────────────────────────────┤
│                                         │
│ PyTorch         → Neural Networks      │
│ HF Hub          → Model Versioning     │
│ TRL             → Advanced RL          │
│ Gradio          → Deployment & Demo    │
│ OpenEnv         → Standardization      │
│ FastAPI+Docker  → Production Server    │
│                                         │
└─────────────────────────────────────────┘
```

---

## 📚 Technologies & Concepts

### 1. **PyTorch** (Neural Networks)
What: Deep learning framework
Where: `train_dqn.py`
Learn:
- Neural network architecture (`nn.Module`)
- Forward/backward pass (training loop)
- Optimizers (Adam)
- Loss functions
- Backpropagation

### 2. **Hugging Face Hub** (Model Sharing)
What: GitHub for ML models
Where: `hf_hub_integration.py`
Learn:
- Save/load models
- Push to Hub
- Version control
- Model cards

### 3. **TRL** (Advanced RL)
What: Modern RL algorithms
Learn:
- DQN (current implementation)
- PPO (Proximal Policy Optimization)
- GRPO (Group Relative Policy Optimization)

### 4. **Gradio** (Web Interface)
What: Interactive ML demos
Where: `gradio_app.py`
Learn:
- Deploy models as web apps
- Create dashboards
- Share interactive demos

### 5. **OpenEnv** (Standardization)
What: Standard RL interface
Where: Custom Pong environment
Learn:
- `reset()` method
- `step(action)` method
- `.sync()` wrapper pattern

### 6. **FastAPI + Docker** (Production)
What: Deploy ML servers
Where: `server/`
Learn:
- WebSocket communication
- Container deployment
- REST APIs

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Docker Server
```bash
docker build -t pong-server .
docker run -p 8000:8000 pong-server
```

### 3. Train Neural Network (PyTorch)
```bash
python train_dqn.py
```

Expected output:
```
Episode  10/100 | Reward: +0.2 | Avg: -0.60 | Loss: 0.1891 | Epsilon: 0.951
Episode  20/100 | Reward: +1.5 | Avg: +0.50 | Loss: 0.1234 | Epsilon: 0.906
...
Episode 100/100 | Reward: +8.5 | Avg: +5.50 | Loss: 0.0234 | Epsilon: 0.105
```

### 4. Save & Share Model (HF Hub)
```bash
python hf_hub_integration.py
```

### 5. Deploy Interactive Demo (Gradio)
```bash
python gradio_app.py
```

Then open: http://localhost:7860

---

## 📁 Project Structure

```
pong_env/
├── 🔥 train_dqn.py                ← PyTorch DQN training
├── 🤗 hf_hub_integration.py        ← Hugging Face Hub model management
├── 🌐 gradio_app.py                ← Web interface for demo
├── models.py                        ← Data models (Pydantic)
├── client.py                        ← Game client
├── requirements.txt                 ← All dependencies
├── README.md
├── COMPLETE_LEARNING_ROADMAP.md    ← Detailed learning path
└── server/
    ├── app.py                       ← FastAPI server
    ├── pong_environment.py          ← Game logic
    └── Dockerfile                   ← Docker config
```

---

## 🧠 How Neural Network Training Works

```python
Episode 1:
  Network with random weights
  └─ Play game → Reward: -1
  └─ Update weights (learn what NOT to do)

Episode 2:
  Network with updated weights
  └─ Play game → Reward: +2
  └─ Update weights (reinforce good moves)

Episode 100:
  Network with learned weights
  └─ Play game → Reward: +8
  └─ Network knows how to play!
```

**Key: Weights change based on reward signal!**

---

## 📊 PyTorch Concepts Used

```python
# 1. Define Network (nn.Module)
class QNetwork(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(9, 128)      # Input → Hidden
        self.fc2 = nn.Linear(128, 128)    # Hidden → Hidden
        self.fc3 = nn.Linear(128, 3)      # Hidden → Output (3 actions)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))       # Activation
        x = F.relu(self.fc2(x))           # Activation
        return self.fc3(x)                 # Q-values for each action

# 2. Create Optimizer
optimizer = optim.Adam(network.parameters(), lr=0.001)

# 3. Training Loop
loss = loss_fn(predicted_q, target_q)     # Calculate error
loss.backward()                            # Backpropagation (compute gradients)
optimizer.step()                           # Update weights

# Result: Weights get better!
```

---

## 🤗 Hugging Face Hub Workflow

```python
# 1. Train agent
agent = DQNAgent()
train_dqn_agent(episodes=100)

# 2. Save locally
manager = HFHubManager("username/pong-dqn")
save_dir = manager.save_model(agent, "v1")

# 3. Push to Hub
manager.push_to_hub(save_dir)

# 4. Load from anywhere
loaded_agent = manager.load_model_from_hub()
```

Result: Your model is on HF Hub! 🚀

---

## 🌐 Gradio Deployment

```python
# Create interactive interface
interface = gr.Interface(
    fn=play_game,  # Your game function
    inputs=gr.Slider(10, 500, 100),
    outputs=[gr.Image(), gr.Markdown()],
)

# Launch
interface.launch(share=True)
```

**Share link:** Anyone can play with your agent!

---

## 📈 Learning Progression

### Week 1: PyTorch Fundamentals
- [ ] Understand `nn.Module`
- [ ] Understand `forward()` pass
- [ ] Understand backpropagation
- [ ] Run `train_dqn.py`
- [ ] Modify network architecture

### Week 2: Model Management
- [ ] Create HF Hub account
- [ ] Learn HF API
- [ ] Save/load models
- [ ] Push to Hub
- [ ] Version control

### Week 3: Deployment
- [ ] Create Gradio interface
- [ ] Deploy locally
- [ ] Deploy on HF Spaces
- [ ] Share with community

### Week 4: Advanced RL
- [ ] Study PPO algorithm
- [ ] Study GRPO algorithm
- [ ] Compare with DQN
- [ ] Plan future improvements

---

## 💡 Key Concepts

### Weights (What Gets Trained)
```
Input: [ball_x, ball_y, vx, vy, ...]
         ↓
    Multiply by WEIGHTS (neurons)
         ↓
    Pass through activation
         ↓
    Repeat multiple layers
         ↓
Output: [Q_UP, Q_DOWN, Q_STAY]

← These weights START random, GET UPDATED through training!
```

### Loss Function (How We Measure Error)
```
loss = (predicted_Q - actual_reward)²
       ↑                 ↑
    What network        What we
    predicted          actually got

Lower loss = better prediction
```

### Backpropagation (How Weights Update)
```
1. Compute loss
2. Backward propagation: ∇loss (gradient)
3. Optimizer applies: weight -= learning_rate * ∇loss
4. Repeat: weights get better!
```

---

## 🎮 How DQN Learning Works

```
for episode in range(100):
    state = env.reset()
    
    while not done:
        # Network decides best action
        action = agent.choose_action(state)
        
        # Play one step
        next_state, reward, done = env.step(action)
        
        # Remember experience
        agent.remember(state, action, reward, next_state, done)
        
        # Train network on random batch
        agent.train()  # ← Updates weights!
        
        state = next_state
    
    # Results: Network got better this episode!
```

---

## 🚀 Hackathon Pitch

```
"I built a complete ML/AI learning project:

1. PyTorch: Neural network that learns to play Pong
2. Hugging Face Hub: Version and share trained models
3. TRL: Understand advanced RL algorithms
4. Gradio: Deploy interactive demos
5. OpenEnv: Follow ML standardization

Result: Full-stack ML engineering experience in one project!"
```

---

## 📖 Detailed Learning Resources

See `COMPLETE_LEARNING_ROADMAP.md` for:
- Week-by-week learning plan
- Detailed concept explanations
- Implementation steps
- Research papers
- Community resources

---

## Troubleshooting

**"Connection refused" error**
```bash
docker run -p 8000:8000 pong-server
```

**PyTorch issues**
```bash
pip install torch torchvision torchaudio
```

**HF Hub token**
```bash
export HF_TOKEN=hf_xxxxx
# Get token: https://huggingface.co/settings/tokens
```

**Gradio not launching**
```bash
pip install gradio --upgrade
python gradio_app.py
```

---

## Next Steps

1. ✅ Run training: `python train_dqn.py`
2. ✅ Save model: `python hf_hub_integration.py`
3. ✅ Deploy demo: `python gradio_app.py`
4. ✅ Push to HF Spaces
5. ✅ Share with community
6. ✅ Upgrade to PPO/GRPO

---

## Resources

- **PyTorch**: https://pytorch.org/
- **Hugging Face**: https://huggingface.co/
- **TRL Library**: https://github.com/huggingface/trl
- **Gradio**: https://www.gradio.app/
- **DQN Paper**: https://arxiv.org/abs/1312.5602
- **OpenEnv**: https://github.com/openenv-foundation

---

## License

MIT

**Built for learning. Made for hackathons. Ready for production.** 🚀

## Quick Answer: How Does RL Training Work?

```
Episode 1:
  Network with random weights
  └─ Play game → Reward: -1 (bad)
  └─ Update weights (learn what NOT to do)

Episode 2:
  Network with updated weights
  └─ Play game → Reward: +2 (better!)
  └─ Update weights (reinforce good moves)

Episode 3:
  Network with even better weights
  └─ Play game → Reward: +5 (much better!)
  └─ Update weights (keep improving)

...continues learning and improving...
```

**The network LEARNS by updating weights based on rewards!**

---

## What is DQN?

**Deep Q-Network:** A neural network that learns the value of each action in a given state.

```
Input: Game State (ball position, paddle position, velocities, etc.)
  ↓
Neural Network (weights w1, w2, w3, ...)
  ↓
Output: Q-values for each action (UP, DOWN, STAY)
  ↓
Pick action with highest Q-value
  ↓
Play game → Get reward
  ↓
Update weights using the reward
  ↓
Network gets better!
```

---

## How Weights Change (Real Learning!)

```
PHASE 1: Random Network
├─ Network has random weights
├─ Play game → Reward: -1
├─ Weights are BAD
└─ Need to improve

PHASE 2: After Learning
├─ Network learned from rewards
├─ Weights updated 50 times
├─ Play game → Reward: +5
├─ Weights are GOOD!
└─ Network knows what to do

KEY: Weights are NOT fixed!
     They UPDATE after each episode based on reward!
```

---

## Key Differences: ML vs RL Training

| Aspect | Traditional ML | DQN (RL) |
|--------|---|---|
| **Weights** | Fixed after training | Updated every episode |
| **Training Data** | Labeled examples | Rewards from playing |
| **Learning Signal** | Error vs label | Reward signal |
| **Process** | Train once, use forever | Play → Learn → Play better → Learn → ... |
| **Improvement** | None after training | Improves over episodes |

---

## Quick Start

### 1. Build & Run Docker Server
```bash
docker build -t pong-server .
docker run -p 8000:8000 pong-server
```

### 2. Run Neural Network Training (New Terminal)
```bash
python train_dqn.py
```

Watch the network learn over 100 episodes!

---

## What Happens During Training

```python
# Episode 1
random_weights → play game → reward = -1 → update weights

# Episode 2
better_weights → play game → reward = +2 → update weights

# Episode 3
even_better_weights → play game → reward = +5 → update weights

# ...continues...
```

**The rewards guide the weight updates!** Network learns which moves lead to higher rewards.

---

## How the Neural Network Works

### Input (9 values from game state):
```
- Ball X position (normalized)
- Ball Y position (normalized)
- Ball X velocity
- Ball Y velocity
- Your paddle Y position
- AI paddle Y position
- Your score
- AI score
- Relative distance (you to ball)
```

### Network Architecture:
```
Input Layer (9 values)
    ↓
Hidden Layer 1 (128 neurons) + ReLU activation
    ↓
Hidden Layer 2 (128 neurons) + ReLU activation
    ↓
Output Layer (3 neurons = 3 actions: UP, DOWN, STAY)
    ↓
Q-values for each action
```

### Decision Making:
```
Input state → Network forward pass → Get Q-values
Q-values: [0.5, 1.2, 0.3] for [UP, DOWN, STAY]
Pick action with highest Q-value → DOWN (1.2 is highest)
```

---

## Training Algorithm: DQN

```
For each episode:
  1. Play game with current network
  2. Store (state, action, reward, next_state) in memory
  3. Sample random batch from memory
  4. Calculate target Q-values using rewards
  5. Update network weights to match targets
  6. Update target network every 10 episodes
  7. Reduce exploration rate (epsilon decay)
  
Result: Network learns optimal policy!
```

---

## Key Hyperparameters

Edit `train_dqn.py` to change:
- `learning_rate` (0.001) - How much weights change per update
- `gamma` (0.99) - How much future rewards matter
- `epsilon` (1.0) - Starting exploration rate
- `epsilon_decay` (0.995) - How fast to reduce exploration
- `batch_size` (32) - Samples for training
- `episodes` (100) - Training episodes

---

## Project Structure

```
pong_env/
├── models.py              ← Data models
├── client.py              ← WebSocket client
├── train_dqn.py           ← DQN neural network training ⭐
├── train.py               ← Old strategy testing (keep as reference)
├── requirements.txt
└── server/
    ├── pong_environment.py  ← Game logic
    ├── app.py               ← FastAPI server
    └── Dockerfile
```

---

## Expected Training Progress

```
Episode   1/100 | Reward: -1.5 | Avg: -1.50 | Loss: 0.2341 | Epsilon: 0.995
Episode  10/100 | Reward: +0.2 | Avg: -0.60 | Loss: 0.1891 | Epsilon: 0.951
Episode  20/100 | Reward: +1.5 | Avg: +0.50 | Loss: 0.1234 | Epsilon: 0.906
Episode  30/100 | Reward: +2.3 | Avg: +1.20 | Loss: 0.0891 | Epsilon: 0.864
Episode  40/100 | Reward: +3.1 | Avg: +1.80 | Loss: 0.0654 | Epsilon: 0.824
...
Episode 100/100 | Reward: +8.5 | Avg: +5.50 | Loss: 0.0234 | Epsilon: 0.105

TRAINING SUMMARY
Total episodes:           100
Avg reward (all):        +2.50
Avg reward (last 10):    +6.20  ← Improving!
Best episode:            +8.50
Worst episode:           -2.10
Total cumulative reward: +250.00
Final epsilon:           0.105
```

---

## How Weights are Updated

```
BEFORE (random weights):
w1 = 0.523, w2 = -0.812, w3 = 0.234
Play game → Reward: -1
Network predicts BAD actions

LOSS FUNCTION (how wrong we are):
loss = (predicted_Q - target_Q)^2

GRADIENT DESCENT (update direction):
w_new = w_old - learning_rate * ∇loss

AFTER update:
w1 = 0.525, w2 = -0.810, w3 = 0.236
Slightly changed based on reward!

AFTER 100 episodes of updates:
w1 = 0.891, w2 = 0.442, w3 = -0.156
Completely changed! Network learned!
Play game → Reward: +8
Network predicts GOOD actions
```

---

## Exploration vs Exploitation (Epsilon-Greedy)

```
Early training (high epsilon):
├─ 100% random actions initially
├─ Explores all possibilities
└─ Reward: low, but learns broadly

Mid training (medium epsilon):
├─ 50% network actions, 50% random
├─ Explores + exploits
└─ Reward: improving

Late training (low epsilon):
├─ 99% network actions, 1% random
├─ Mostly exploits learned policy
└─ Reward: high and stable
```

---

## Experience Replay (Why It Works)

```
NAIVE APPROACH (no replay):
Play action → immediate weight update
Problem: Actions are highly correlated, network can't learn well

BETTER APPROACH (replay buffer):
Play action → store in memory
Sample 32 random experiences → update network
Problem solved: Random batch breaks correlation!
```

---

## Reward System

| Event | Reward | Why |
|-------|--------|-----|
| Ball hits your paddle | +1 | Good - defended! |
| You score | +2 | Great - won rally! |
| AI scores | -1 | Bad - you failed! |
| Nothing | 0 | Neutral |

Network learns to maximize rewards!

---

## Usage Example

```python
from train_dqn import DQNAgent, extract_state
from client import PongEnvSync, PongEnvClient
from models import PongAction

# Create trained agent
agent = DQNAgent(state_size=9, action_size=3)

# Play with trained agent
with PongEnvSync(PongEnvClient("ws://localhost:8000/ws/client")).sync() as env:
    obs = env.reset()
    state = extract_state(obs)
    
    for _ in range(500):
        action_idx = agent.choose_action(state)  # Network decides
        action_name = ["UP", "DOWN", "STAY"][action_idx]
        obs, reward, done = env.step(PongAction(action=action_name))
        state = extract_state(obs)
        
        if done:
            break
```

---

## Troubleshooting

**"Connection refused"**
```bash
docker run -p 8000:8000 pong-server
```

**Training is slow**
- Reduce episodes temporarily to test
- Network trains in background while playing

**Rewards not improving**
- Increase learning_rate slightly
- Train for more episodes
- Check if network architecture is appropriate

---

## Next Steps

1. ✅ Run: `python train_dqn.py`
2. 📊 Watch rewards improve over episodes
3. 🎯 Observe network learning in real-time
4. 💾 Save trained agent weights
5. 🚀 Deploy agent to play optimally

---

## How This is Different from Traditional ML

```
TRADITIONAL ML:
├─ Train once on fixed dataset
├─ Get fixed weights
├─ Use same weights forever
└─ No improvement over time

DQN (REINFORCEMENT LEARNING):
├─ Train continuously while playing
├─ Weights update every episode
├─ Gets better over time
└─ Learns from own experience!
```

**That's the power of RL!** 🎮

---

## Key Concepts

✅ **Neural Network** - Learns to map states → actions  
✅ **Q-values** - Estimated value of each action  
✅ **Reward Signal** - Feedback for learning  
✅ **Weight Updates** - Network learns from rewards  
✅ **Experience Replay** - Batch training from memory  
✅ **Epsilon-Greedy** - Balance exploration vs exploitation  
✅ **Target Network** - Stabilizes learning  

---

## References

- Deep Q-Network (DQN): https://arxiv.org/abs/1312.5602
- PyTorch: https://pytorch.org/
- Reinforcement Learning: https://en.wikipedia.org/wiki/Reinforcement_learning

---

## License

MIT

### Traditional ML (Linear Regression, etc.)
```
Input Data (observations)
    ↓
Multiply by FIXED weights
    ↓
Output prediction

Example: house_price = 100 * sqft + 50000 * bedrooms + 25000
         (weights: 100, 50000, 25000 - FIXED after training)
         
You train ONCE to find the best weights,
then use the SAME weights for predictions forever.
```

### Reinforcement Learning (What You Have Here) ⭐
```
Episode 1: Try random strategy → get -1.5 reward
Episode 2: Try smart strategy → get +5.0 reward
Episode 3: Try another strategy → get +2.0 reward

NO WEIGHTS to multiply!
NO FIXED MODEL!

You're just TESTING different hardcoded strategies
and MEASURING which one is better.
```

---

## What "Training" Actually Means Here (Phase 1)

**Currently:** You're NOT using weights or neural networks at all!

You're simply:
1. **Test** strategy A (random moves) → measure performance
2. **Test** strategy B (follow ball) → measure performance  
3. **Compare** the results
4. **Pick** the winner

```python
# This is your "training":
random_reward = -1.5
smart_reward = +5.0

if smart_reward > random_reward:
    best_strategy = "follow ball"
```

**No machine learning involved yet!** Just testing and comparing.

---

## Phase 2: Real Training (Neural Network) - FUTURE

This is where weights come in:

```
Input: Board observation (40x20 grid)
    ↓
Neural Network:
├─ Layer 1 (weights w1)
├─ Layer 2 (weights w2)
└─ Output: Action (UP, DOWN, or STAY)
    ↓
Play game → get reward
    ↓
Update weights using reward signal
    ↓
Play again with new weights
    ↓
Repeat until network learns optimal strategy

Example:
Network v1 (random weights) → reward = -1
Update weights
Network v2 (better weights) → reward = +2
Update weights
Network v3 (even better) → reward = +5
...keeps improving...
```

**Here, weights are NOT fixed!** They change after each episode based on rewards.

---

## Comparison Table

| Aspect | Traditional ML | RL (Current) | RL (Phase 2) |
|--------|---|---|---|
| **Weights?** | YES - Fixed after training | NO weights yet | YES - Change every episode |
| **Training Data?** | Labeled examples (input→output) | Rewards from playing | Rewards from playing |
| **Fixed Model?** | YES - same weights forever | NO - just strategies | NO - weights update |
| **Process** | Fit to data once | Test strategies repeatedly | Learn from experience |
| **After Training** | Use same weights | Use best strategy | Keep learning, weights keep updating |

---

## Your Scenario Example

### If This Was Traditional ML (Not What You Have):

```
Training Phase:
├─ Collect dataset of 1000 game states
├─ Label each: "ball_x=20, ball_y=10" → "move UP"
├─ Fit weights to minimize error
└─ Weights found: w1=0.5, w2=0.3, w3=-0.2

Prediction Phase (FIXED weights):
├─ New state: "ball_x=22, ball_y=8"
├─ Calculate: 0.5*22 + 0.3*8 + (-0.2) = result
├─ If result > 0 → move UP
└─ Use SAME weights forever (never change)
```

### What You Actually Have (RL):

```
Phase 1 (Current - Policy Testing):
├─ Strategy A: random moves → -1.5 reward
├─ Strategy B: follow ball → +5.0 reward
├─ Pick: Strategy B (no weights, just a rule)
└─ Use forever (until you change code)

Phase 2 (Future - Neural Network):
├─ Network with random weights
├─ Play 1 game → get reward
├─ Update weights based on reward
├─ Play next game with new weights
├─ Keep updating and improving
└─ Network learns optimal policy (weights keep changing!)
```

---

## Key Difference: Fixed vs Adaptive

### Fixed Weights (Traditional ML)
```
Train → Find weights → Use same weights forever

Example: Spam detector trained once, weights never change
```

### Adaptive Weights (RL - Phase 2)
```
Play game → Get reward → Update weights → Play again → Update weights → ...

Example: Your agent learns to be better at Pong over time
```

---

## Current Status of Your Code

**Right now (Phase 1):**
- ❌ NO weights
- ❌ NO neural network
- ✅ Just testing hardcoded strategies
- ✅ Comparing which is better

**In Phase 2 (when you implement it):**
- ✅ YES weights (neural network)
- ✅ YES learning (weights update from rewards)
- ✅ YES adaptive (policy improves over time)

---

## What "Training Data" Means in Each Context

### Traditional ML (Linear Regression)
```
Training data = labeled examples
Example:
house_price = [200000, 150000, 300000, ...]
sqft = [2000, 1500, 3000, ...]

Train once → Find: price = 100*sqft + 25000
Weights FIXED: [100, 25000]
```

### RL (Your System - Phase 1)
```
Training data = game episodes and rewards
Example:
Random strategy: 10 episodes, avg reward = -1.5
Smart strategy: 10 episodes, avg reward = +5.0

Train (test both) → Pick winner
NO WEIGHTS: just pick "smart strategy"
```

### RL (Your System - Phase 2 - Future)
```
Training data = states, actions, rewards from gameplay
Example:
State: ball_at_10, my_paddle_at_8 → Action: UP → Reward: +1
State: ball_at_15, my_paddle_at_8 → Action: DOWN → Reward: +2

Train → Update network weights
Weights CHANGE: keep learning from rewards
```

---

## Quick Answer to Your Question

> "So here is that the same? We convert each scenario into an input and we multiply it by our fixed weights after training? Are the weights always fixed after training?"

**Right now (Phase 1):** NO weights, NO multiplication, just testing strategies  
**Phase 2 (future):** YES weights, YES multiplication, but NOT fixed - they UPDATE!

This is the KEY difference between ML and RL:
- **ML:** Weights fixed after training
- **RL:** Weights keep updating based on rewards (that's how it learns!)

---

## Quick Start

### 1. Build & Run Docker Server
```bash
docker build -t pong-server .
docker run -p 8000:8000 pong-server
```

### 2. Run Training (New Terminal)
```bash
python train.py
```

---

## How It Works

Your paddle (LEFT):
- You decide actions: UP, DOWN, STAY
- You get rewards: +1 (hit ball), +2 (scored), -1 (AI scored), 0 (nothing)
- You can test different strategies

AI paddle (RIGHT):
- Always follows the ball (hardcoded)
- Never improves
- Fixed opponent

**Training = Test strategy A → measure reward → Test strategy B → measure reward → Compare → Pick winner**

### Example:
```
Random Strategy:  avg reward = -1.5 (losing) ✗
Smart Strategy:   avg reward = +5.0 (winning) ✓
→ Smart wins!
```

---

## Project Structure

```
pong_env/
├── models.py              ← Data models
├── client.py              ← WebSocket client
├── train.py               ← Training script
├── requirements.txt
└── server/
    ├── pong_environment.py  ← Game logic
    ├── app.py               ← FastAPI server
    └── Dockerfile
```

---

## Usage Example

```python
from client import PongEnvSync, PongEnvClient
from models import PongAction

with PongEnvSync(PongEnvClient("ws://localhost:8000/ws/client")).sync() as env:
    for episode in range(10):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(500):
            action = PongAction(action="UP")  # Your decision
            obs, reward, done = env.step(action)
            episode_reward += reward
            if done:
                break
        
        print(f"Episode reward: {episode_reward}")
```

---

## Reward System

| Event | Reward |
|-------|--------|
| Ball hits your paddle | +1 |
| You score | +2 |
| AI scores | -1 |
| Nothing | 0 |

---

## Configuration

Edit `server/pong_environment.py`:
- `BALL_SPEED` - Speed (default: 0.3)
- `PADDLE_HEIGHT` - Size (default: 4)
- `AI_SPEED` - Difficulty (default: 0.4)

---

## Key Facts

✅ You control LEFT paddle (train it)  
✅ AI controls RIGHT paddle (fixed opponent)  
✅ Training = test strategies, measure, pick best  
✅ AI never improves  
✅ Episode ends at 11 points or 1000 steps  

---

## Troubleshooting

**Connection refused:** Make sure Docker is running
```bash
docker run -p 8000:8000 pong-server
```

**Test health:** 
```powershell
Invoke-WebRequest -Uri "http://localhost:8000/health" -Method GET
```

---

## Next Steps

1. Run the training: `python train.py`
2. Observe which policy wins
3. Create your own strategy
4. Build a neural network (Phase 2)
