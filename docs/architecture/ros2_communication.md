# ROS 2 Communication Architecture

## How It Actually Works

You have a **ROS 2 network bridge** running on Windows that enables Isaac Sim/Lab to communicate with WSL (and eventually your Jetson robot).

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Windows Host                                 │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Isaac Sim 5.0.0 (Python 3.11)                                │   │
│  │ ┌────────────────────────────────────────────────────────┐   │   │
│  │ │ ROS 2 Bridge Extension (omni.isaac.ros2_bridge)        │   │   │
│  │ │ - Publishes: /joint_states, /tf, /camera/image        │   │   │
│  │ │ - Subscribes: /cmd_vel, /joint_commands               │   │   │
│  │ └──────────────────┬─────────────────────────────────────┘   │   │
│  └────────────────────┼─────────────────────────────────────────┘   │
│                       │                                              │
│  ┌────────────────────┼─────────────────────────────────────────┐   │
│  │                    │                                          │   │
│  │     ┌──────────────▼──────────────┐                          │   │
│  │     │  Fast DDS (DDS Network)     │                          │   │
│  │     │  Domain ID: 55              │                          │   │
│  │     │  Ports: UDP 7400-7410, etc. │                          │   │
│  │     └──────────────┬──────────────┘                          │   │
│  │                    │                                          │   │
│  └────────────────────┼─────────────────────────────────────────┘   │
│                       │                                              │
│  ┌────────────────────┼─────────────────────────────────────────┐   │
│  │ ROS 2 Humble (Python 3.8)                                    │   │
│  │                    │                                          │   │
│  │     ┌──────────────▼──────────────┐                          │   │
│  │     │ ros2 CLI / Custom Nodes     │                          │   │
│  │     │ - ros2 topic list           │                          │   │
│  │     │ - ros2 topic echo /chatter  │                          │   │
│  │     │ - Custom monitoring nodes   │                          │   │
│  │     └─────────────────────────────┘                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                       │                                              │
└───────────────────────┼──────────────────────────────────────────────┘
                        │
                        │ Fast DDS Bridge
                        │ (UDP through Windows firewall)
                        │
┌───────────────────────┼──────────────────────────────────────────────┐
│                       │      WSL2 (Ubuntu 22.04)                     │
│                       │                                              │
│  ┌────────────────────▼─────────────────────────────────────────┐   │
│  │  Fast DDS Client (~/fastdds_windows.xml)                     │   │
│  │  - Windows IP: 10.255.255.254                                │   │
│  │  - Domain ID: 55                                             │   │
│  └────────────────────┬─────────────────────────────────────────┘   │
│                       │                                              │
│  ┌────────────────────┼─────────────────────────────────────────┐   │
│  │ ROS 2 Humble (Python 3.10)                                   │   │
│  │                    │                                          │   │
│  │     ┌──────────────▼──────────────┐                          │   │
│  │     │ ROS 2 Nodes                 │                          │   │
│  │     │ - talker/listener demos     │                          │   │
│  │     │ - Data collection nodes     │                          │   │
│  │     │ - Monitoring scripts        │                          │   │
│  │     └─────────────────────────────┘                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## What's Actually Happening

### The ROS 2 "Bridge" is Not a Separate Process

The "bridge" is actually **Fast DDS** (the DDS implementation) running on all machines:

1. **Isaac Sim** has a built-in ROS 2 bridge extension that:
   - Uses its own Python 3.11 interpreter
   - Has ROS 2 libraries compiled for Python 3.11
   - Publishes/subscribes to topics via Fast DDS

2. **Windows ROS 2 (Python 3.8)** can:
   - Monitor topics from Isaac Sim
   - Send commands to Isaac Sim
   - Communicate with WSL nodes

3. **WSL ROS 2 (Python 3.10)** can:
   - See all topics from Isaac Sim
   - See all topics from Windows ROS 2
   - Publish its own topics back

### They All Connect Through Fast DDS Network

```
Isaac Sim (Py 3.11) ─────┐
                         │
Windows ROS2 (Py 3.8) ───┼──► Fast DDS Network (Domain 55)
                         │    - UDP multicast/unicast
WSL ROS 2 (Py 3.10) ─────┘    - Message serialization
                              - Service discovery
```

---

## Message Flow Example

### Scenario: Isaac Sim publishes joint states, WSL monitors them

```
┌──────────────────────────────────────────────────────────────┐
│ Step 1: Isaac Sim starts with ROS 2 bridge enabled          │
└──────────────────────────────────────────────────────────────┘
   │
   │ Isaac Sim (Python 3.11) publishes:
   │   Topic: /robot/joint_states
   │   Type: sensor_msgs/JointState
   │   Rate: 50 Hz
   ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 2: Fast DDS serializes message and broadcasts          │
└──────────────────────────────────────────────────────────────┘
   │
   │ UDP packets sent to:
   │   - Multicast address (for discovery)
   │   - Known peers (10.255.255.254:7410 for WSL)
   │   - Port 7400-7410
   ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 3: WSL Fast DDS receives and deserializes              │
└──────────────────────────────────────────────────────────────┘
   │
   │ WSL ROS 2 (Python 3.10) receives:
   │   Same topic: /robot/joint_states
   │   Same type: sensor_msgs/JointState
   │   Same data, different Python version!
   ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 4: WSL node processes the message                      │
└──────────────────────────────────────────────────────────────┘
   
   # In WSL:
   ros2 topic echo /robot/joint_states
   # Shows live data from Isaac Sim!
```

---

## Key Configuration Points

### 1. Domain ID (Must Match Everywhere)
```bash
# Windows Isaac Sim
--/exts/ros2_bridge/useDomainID=55

# Windows ROS 2
$env:ROS_DOMAIN_ID = "55"

# WSL ROS 2
export ROS_DOMAIN_ID=55
```

### 2. RMW Implementation (Must Match)
```bash
# Windows
$env:RMW_IMPLEMENTATION = "rmw_fastrtps_cpp"

# WSL
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

### 3. Fast DDS Profile (WSL needs to know Windows IP)
```bash
# WSL: ~/fastdds_windows.xml
<initialPeersList>
  <locator>
    <address>10.255.255.254</address>  # Windows IP
    <port>7410</port>
  </locator>
</initialPeersList>
```

---

## What You Can Do

### Monitor Isaac Sim from WSL
```bash
# WSL Terminal
source scripts/wsl/setup_ros2_only.sh
ros2 topic list           # See all Isaac Sim topics
ros2 topic echo /joint_states  # Monitor joint states
ros2 topic hz /camera/image    # Check camera frame rate
```

### Control Isaac Sim from WSL
```bash
# WSL Terminal
ros2 topic pub /cmd_vel geometry_msgs/Twist \
  "{linear: {x: 1.0}, angular: {z: 0.5}}"
# Robot in Isaac Sim moves!
```

### Monitor from Windows ROS 2
```powershell
# Windows Terminal
.\scripts\networking\setup_ros2_humble_windows.ps1
ros2 topic list
ros2 topic echo /robot/joint_states
```

### Record Data for Analysis
```bash
# WSL Terminal
ros2 bag record -a -o ./data/training_run_001
# Records all topics to a bag file
# Later, analyze in .venv_rl311 with rosbags library
```

---

## No "Bridge Process" Needed!

The beauty of ROS 2 + DDS is that there's **no central broker** like ROS 1 had:

### ROS 1 (Old Way) ❌
```
Publisher → roscore (master) → Subscriber
          (single point of failure)
```

### ROS 2 (Current) ✅
```
Publisher ←→ DDS Network ←→ Subscriber
         (peer-to-peer discovery)
```

Each node discovers others automatically via DDS, so:
- ✅ No roscore to start
- ✅ No bridge process to maintain
- ✅ Works across machines, languages, Python versions

---

## Summary

**You don't have a separate "ROS 2 bridge" process.** Instead:

1. **Isaac Sim has ROS 2 built-in** (Python 3.11, compiled for 3.11)
2. **Windows ROS 2 installation** (Python 3.8, compiled for 3.8)
3. **WSL ROS 2 installation** (Python 3.10, compiled for 3.10)

**They all communicate via Fast DDS network layer**, which is:
- Language-agnostic
- Python-version-agnostic
- Network-based (UDP)
- Peer-to-peer (no central broker)

**The "bridge" is just the DDS network protocol that all ROS 2 nodes use automatically!** 🌉

---

**Last Updated:** 2025-10-13
