# CinebotRL Documentation Index

**Welcome to the CinebotRL project documentation!** This index will guide you to the right document for your needs.

---

## 🚀 Quick Start (Start Here!)

**New to the project?** Read these in order:

1. **[Project README](../README.md)** - Project overview and quick setup
2. **[Quick Reference Card](QUICK_REFERENCE.md)** - One-page cheat sheet (⭐ print this!)
3. **[WSL Setup Verification](setup/wsl_setup_guide.md)** - Verify your WSL environment is ready

---

## 📚 Documentation Structure

```
docs/
├── README.md (this file)              # Documentation index
├── QUICK_REFERENCE.md                 # One-page reference card
│
├── setup/                             # Installation & Setup
│   ├── wsl_setup_guide.md            # WSL environment setup & verification
│   ├── windows_setup_guide.md        # Windows Isaac Sim/Lab setup
│   ├── ros2_communication_setup.md   # ROS 2 networking configuration
│   └── isaaclab_windows.md           # Legacy: Isaac Lab setup notes
│
├── architecture/                      # System Architecture
│   ├── overview.md                   # High-level architecture
│   ├── ros2_communication.md         # How ROS 2 works across systems
│   └── python_environments.md        # Why multiple Python versions
│
├── workflows/                         # How-To Guides
│   ├── daily_workflow.md             # Common daily tasks
│   ├── training_workflow.md          # How to run RL training
│   └── monitoring_workflow.md        # How to monitor training
│
├── reference/                         # Technical Reference
│   ├── environment_variables.md      # All environment variables
│   ├── scripts_reference.md          # What each script does
│   └── troubleshooting.md            # Common issues & solutions
│
└── tracking/                          # Project History
    ├── phase0_environment.md         # Environment setup log
    ├── ee_frame_alignment.md         # Robot frame alignment notes
    └── lessons_learned.md            # Lessons from development

```

---

## 📖 By Topic

### Getting Started
- 🏁 **[Quick Reference Card](QUICK_REFERENCE.md)** - Most common commands
- 🔧 **[WSL Setup Guide](setup/wsl_setup_guide.md)** - Configure WSL environment
- 🪟 **[Windows Setup Guide](setup/windows_setup_guide.md)** - Configure Windows side

### Understanding the System
- 🏗️ **[Architecture Overview](architecture/overview.md)** - How everything fits together
- 🌉 **[ROS 2 Communication](architecture/ros2_communication.md)** - How WSL ↔ Windows works
- 🐍 **[Python Environments Explained](architecture/python_environments.md)** - Why we have multiple Python versions

### Daily Usage
- ⚡ **[Daily Workflow](workflows/daily_workflow.md)** - Your everyday commands
- 🎓 **[Training Workflow](workflows/training_workflow.md)** - How to train RL agents
- 📊 **[Monitoring Workflow](workflows/monitoring_workflow.md)** - How to monitor training

### Reference
- 📋 **[Environment Variables](reference/environment_variables.md)** - Complete variable reference
- 📜 **[Scripts Reference](reference/scripts_reference.md)** - What each script does
- 🐛 **[Troubleshooting Guide](reference/troubleshooting.md)** - Fix common issues

---

## 🎯 By Use Case

### "I want to verify my setup is working"
→ Run: `bash scripts/wsl/check_wsl_setup.sh`  
→ Read: [WSL Setup Guide](setup/wsl_setup_guide.md)

### "I want to test ROS 2 communication"
→ Read: [ROS 2 Communication Setup](setup/ros2_communication_setup.md)  
→ Test: [Quick Reference - Communication Test](QUICK_REFERENCE.md#communication-test)

### "I want to understand why we have different Python versions"
→ Read: [Python Environments Explained](architecture/python_environments.md)

### "I want to start training an RL agent"
→ Read: [Training Workflow](workflows/training_workflow.md)  
→ Reference: [Windows Setup Guide](setup/windows_setup_guide.md)

### "I'm getting errors and need help"
→ Read: [Troubleshooting Guide](reference/troubleshooting.md)  
→ Check: [Environment Variables](reference/environment_variables.md)

### "I want to know what a script does"
→ Read: [Scripts Reference](reference/scripts_reference.md)

---

## 📂 Document Status

| Document | Status | Last Updated |
|----------|--------|--------------|
| Quick Reference | ✅ Complete | 2025-10-13 |
| WSL Setup Guide | ✅ Complete | 2025-10-13 |
| Windows Setup Guide | ✅ Complete | 2025-10-13 |
| Architecture Overview | ✅ Complete | 2025-10-13 |
| ROS 2 Communication | ✅ Complete | 2025-10-13 |
| Python Environments | ✅ Complete | 2025-10-13 |
| Daily Workflow | ✅ Complete | 2025-10-13 |
| Troubleshooting | ✅ Complete | 2025-10-13 |
| Training Workflow | ⏳ Pending | - |
| Monitoring Workflow | ⏳ Pending | - |
| Environment Variables | ⏳ Pending | - |
| Scripts Reference | ⏳ Pending | - |

---

## 🗂️ Legacy Documents (Being Consolidated)

These documents contain useful information but are being reorganized:

- `lessons_learnt_ros2OnWindows.md` → Being merged into setup guides
- `wsl_workflow_guide.md` → Split into setup + workflows
- `wsl_windows_integration.md` → Reorganized into architecture docs
- `ros2_python_versions_explained.md` → Moved to architecture
- `ros2_bridge_explained.md` → Moved to architecture

---

## 🔄 Document Migration Plan

We're consolidating documentation to reduce scatter. Here's the plan:

### Phase 1: Core Documents (✅ Complete)
- [x] Quick Reference Card
- [x] WSL Setup Guide (consolidated)
- [x] Windows Setup Guide (consolidated)
- [x] Architecture Overview
- [x] ROS 2 Communication Explanation
- [x] Python Environments Explanation

### Phase 2: Workflow Documents (⏳ In Progress)
- [ ] Daily Workflow Guide
- [ ] Training Workflow Guide
- [ ] Monitoring Workflow Guide

### Phase 3: Reference Documents (⏳ Pending)
- [ ] Environment Variables Reference
- [ ] Scripts Reference
- [ ] Troubleshooting Guide (consolidate all issues)

### Phase 4: Cleanup (⏳ Pending)
- [ ] Archive legacy documents
- [ ] Update all cross-references
- [ ] Verify no broken links

---

## 💡 Documentation Principles

1. **Single Source of Truth** - Each topic has ONE authoritative document
2. **Clear Navigation** - Easy to find what you need
3. **Progressive Disclosure** - Quick start → Details → Deep dive
4. **Practical First** - Show how to do it, then explain why
5. **Keep Updated** - Date stamps and status indicators

---

## 🤝 Contributing to Documentation

Found an issue? Want to improve something?

1. Check if a document already exists in the structure above
2. If reorganizing, update this index
3. Add date stamps when making significant changes
4. Use clear headers and examples
5. Link to related documents

---

## 📞 Quick Help

**Can't find what you need?**

1. Check **[Quick Reference Card](QUICK_REFERENCE.md)** first
2. Use the **[By Use Case](#by-use-case)** section above
3. Check **[Troubleshooting](reference/troubleshooting.md)** for errors

**Still stuck?** Check the legacy documents or project history in `tracking/`.

---

**Last Updated:** 2025-10-13  
**Maintainer:** CinebotRL Team
