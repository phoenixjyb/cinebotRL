
# Network Architecture Design for 9DOF Robot with EE Tracking 3D Trajectories

## Overview

The task involves controlling a **9DOF robot** consisting of a **6DOF arm** and a **3DOF differential drive chassis**. The robot's **end-effector (EE)** needs to track a **3D trajectory** in a time sequence. To handle this, we propose an **actor-critic architecture** using **LSTM** layers to capture **temporal dependencies** in the trajectory tracking task.

This architecture is based on the **PPO (Proximal Policy Optimization)** algorithm from **Stable Baselines3 (SB3)**.

---

## 1. **Input Dimensions**:

The input to the network will consist of the following:
- **Base state (chassis)**:
  - 3D position
  - 4D orientation (quaternion)
  - 3D linear velocity
  - 3D angular velocity
  - **Total**: 13 features
- **Joint state (arm)**:
  - 6 joints × 2 (position + velocity)
  - **Total**: 12 features
- **End-effector (EE) state**:
  - 3D position
  - 4D orientation (quaternion)
  - 3D linear velocity
  - 3D angular velocity
  - **Total**: 13 features
- **Tracking error**:
  - 3D position error
  - 4D orientation error
  - **Total**: 7 features

**Total observation size**: 45 features (could increase with action history or lookahead).

---

## 2. **Network Architecture**:

The **actor-critic** architecture consists of two separate networks:
- **Actor (Policy) Network**: Outputs the actions to be taken by the agent (arm joint positions and chassis velocities).
- **Critic (Value) Network**: Estimates the value function to provide feedback to the actor for policy updates.

### **Shared Layers**:
- **MLP Layers**: These layers will process the observations before passing the output to the LSTM for temporal sequence modeling.

### **LSTM Layer**:
- The **LSTM layer** will capture the sequential data (trajectory tracking) and learn **temporal dependencies**.

### **Actor (Policy) Network**:
- **Input**: Observation space (45 features).
- **Hidden Layers**: 2 layers of 64 units each (or larger if needed).
- **LSTM**: Used to capture temporal dependencies.
- **Output**: 8 actions: 6 arm joint targets (positions) and 2 base velocities (vx, wz).

### **Critic (Value) Network**:
- **Input**: Same observation space (45 features).
- **Hidden Layers**: 2 layers of 64 units each.
- **LSTM**: Used to capture temporal dependencies.
- **Output**: Single scalar value representing the state value.

---

## 3. **Network Design**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_size=256, lstm_layers=2):
        super(ActorCriticNetwork, self).__init__()

        # Shared feature extraction (MLP)
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # LSTM layer for temporal dependencies
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True)

        # Actor (policy) network
        self.actor_fc = nn.Linear(hidden_size, 64)  # Additional FC layer before output
        self.actor_out = nn.Linear(64, action_dim)

        # Critic (value) network
        self.critic_fc = nn.Linear(hidden_size, 64)  # Additional FC layer before output
        self.critic_out = nn.Linear(64, 1)  # Single scalar value

    def forward(self, x):
        # Shared MLP layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # LSTM layer to process sequences
        x, (hn, cn) = self.lstm(x)  # Get the output of LSTM (hn: hidden state)

        # Actor output (policy)
        actor_features = F.relu(self.actor_fc(x[:, -1, :]))  # Use the last hidden state
        action_probs = torch.tanh(self.actor_out(actor_features))  # Using tanh for action scaling ([-1, 1])

        # Critic output (value)
        critic_features = F.relu(self.critic_fc(x[:, -1, :]))  # Use the last hidden state
        state_value = self.critic_out(critic_features)  # Scalar value

        return action_probs, state_value
```

---

## 4. **Training Configuration**:

You can use this architecture with **Stable Baselines3**'s **PPO** implementation by specifying `policy_kwargs` to define the custom policy network:

```python
from stable_baselines3 import PPO

# Define the custom policy with your network
policy_kwargs = {
    "net_arch": [256, 256, 128, 64],  # Custom MLP architecture for shared layers
    "recurrent": True,  # Enable LSTM for recurrent processing of trajectory
}

# Create PPO model with custom policy
model = PPO(
    "MlpPolicy",  # Use custom policy
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=3e-4,
    n_steps=4096,  # Experiment with larger steps for better GAE estimates
    batch_size=1024,  # Experiment with larger batch sizes
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    clip_range=0.2,
    clip_range_vf=1.0,  # Clip value function updates for stability
    target_kl=0.01,  # Set KL divergence threshold for early stopping
)
```

---

## 5. **Key Network Parameters**:

| Parameter                     | Value                              |
|-------------------------------|------------------------------------|
| **Input Dimension**            | 45 (base state + joint state + EE state + tracking error) |
| **Hidden Layer Size**          | 256 units (adjustable based on task complexity) |
| **LSTM Layers**                | 2 (captures temporal dependencies) |
| **Action Dimension (Output)**  | 8 (6 arm joints + 2 base velocities) |
| **Actor Network Parameters**   | ~7,624 parameters |
| **Critic Network Parameters**  | ~7,169 parameters |
| **Total Parameters**           | ~14,793 parameters |

---

## 6. **Monitoring and Performance Tracking**:

Track the following metrics during training to ensure the model is learning effectively:
- **Explained Variance**: Should stabilize above 0 as the critic learns accurate value functions.
- **Action Magnitude**: Keep an eye on the action magnitude to ensure efficient control efforts.
- **Tracking Error**: Ensure the robot is accurately tracking the 3D trajectory.

---

## Conclusion:

This network design leverages both **MLP layers** for feature extraction and **LSTM layers** to capture **temporal dependencies**. The use of **PPO** with custom policy and value networks ensures that the robot can track complex 3D trajectories while maintaining stability and efficiency.

You can experiment with increasing the **hidden layer sizes**, **LSTM layers**, or even explore other architectures like **GRU** if needed.

---

[Download the Network Design Markdown](sandbox:/mnt/data/Network_Design_for_9DOF_Robot.md)
