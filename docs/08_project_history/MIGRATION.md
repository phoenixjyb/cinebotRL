# Documentation Migration Notice

**Date:** 2025-10-13

## What Changed

The documentation has been reorganized into a structured hierarchy for better navigation and maintenance.

## New Structure

```
docs/
├── README.md                    # Documentation index (START HERE)
├── QUICK_REFERENCE.md          # One-page cheat sheet
│
├── setup/                      # Installation & Configuration
│   ├── wsl_setup_guide.md
│   ├── windows_setup_guide.md
│   └── ros2_communication_setup.md
│
├── architecture/               # System Design
│   ├── overview.md
│   ├── ros2_communication.md
│   └── python_environments.md
│
├── workflows/                  # How-To Guides
│   └── daily_workflow.md
│
├── reference/                  # Technical Reference
│   └── troubleshooting.md
│
└── tracking/                   # Project History
    └── phase0_environment.md
```

## Legacy Documents (Moved)

| Old Location | New Location | Status |
|--------------|--------------|--------|
| `wsl_workflow_guide.md` | `workflows/daily_workflow.md` | ✅ Moved |
| `windows_side_requirements.md` | `setup/windows_setup_guide.md` | ✅ Moved |
| `wsl_windows_integration.md` | `architecture/overview.md` | ✅ Copied |
| `ros2_bridge_explained.md` | `architecture/ros2_communication.md` | ✅ Moved |
| `ros2_python_versions_explained.md` | `architecture/python_environments.md` | ✅ Moved |

## Legacy Documents (Kept for Reference)

These remain in `docs/` root but content is being migrated:

- `wsl_windows_integration.md` - Being split into architecture docs
- `lessons_learnt_ros2OnWindows.md` - Contains historical context

## How to Navigate

1. **Always start with:** [`docs/README.md`](README.md)
2. **For quick help:** [`docs/QUICK_REFERENCE.md`](QUICK_REFERENCE.md)
3. **For setup:** [`docs/setup/`](setup/)
4. **To understand:** [`docs/architecture/`](architecture/)
5. **For daily tasks:** [`docs/workflows/`](workflows/)

## Broken Links?

If you find any broken links:
1. Check the new structure above
2. Look in the [`docs/README.md`](README.md) index
3. Documents may have been consolidated - check the legacy table above

## Why This Change?

**Problems with old structure:**
- ❌ Too many scattered Markdown files
- ❌ Unclear which document to read first
- ❌ Duplicate information across files
- ❌ Hard to find specific topics

**Benefits of new structure:**
- ✅ Clear hierarchy and navigation
- ✅ Single source of truth per topic
- ✅ Easy to find relevant information
- ✅ Better maintenance

---

**Questions?** Check [`docs/README.md`](README.md) first!
