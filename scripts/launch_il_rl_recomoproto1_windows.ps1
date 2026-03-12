<#
.SYNOPSIS
    Three-phase IL+RL training launcher for RecomoProto1 on Windows with Isaac Lab.

.DESCRIPTION
    Phase 0 — Collect expert demonstrations via il_dataset.py  (needs Isaac Lab)
    Phase 1 — BC pre-training via pretrain_bc.py               (pure Python / PyTorch)
    Phase 2 — PPO fine-tuning via train.py with --pretrained_policy

.PARAMETER Task
    Task ID (default: RecomoProto1TrackEE-v0)

.PARAMETER NumEnvs
    Number of parallel environments for Phase 2 PPO training (default: 1024)

.PARAMETER Headless
    Run Isaac Lab phases in headless mode (no GUI)

.PARAMETER TotalTimesteps
    Total PPO training timesteps (default: 10000000)

.PARAMETER DemoEpisodes
    Number of expert episodes to collect in Phase 0 (default: 500)

.PARAMETER DemoDir
    Directory for demo .npz files (default: data/il_demos)

.PARAMETER BCEpochs
    Number of BC pre-training epochs (default: 50)

.PARAMETER SkipDemoCollection
    Skip Phase 0 — use if demos already exist in DemoDir

.PARAMETER SkipBC
    Skip Phase 1 — use if a BC policy already exists

.PARAMETER PretrainedPolicy
    Override: provide a direct path to an existing BC policy .zip to use in Phase 2

.EXAMPLE
    .\scripts\launch_il_rl_recomoproto1_windows.ps1 -Headless -NumEnvs 1024

.EXAMPLE
    .\scripts\launch_il_rl_recomoproto1_windows.ps1 -Headless -SkipDemoCollection -BCEpochs 100
#>

param(
    [string]$Task              = "RecomoProto1TrackEE-v0",
    [int]$NumEnvs              = 1024,
    [switch]$Headless,
    [int]$TotalTimesteps       = 10000000,
    [int]$DemoEpisodes         = 500,
    [string]$DemoDir           = "data/il_demos",
    [int]$BCEpochs             = 50,
    [switch]$SkipDemoCollection,
    [switch]$SkipBC,
    [string]$PretrainedPolicy  = ""
)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
$ESC    = [char]27
$Green  = "$ESC[32m"
$Yellow = "$ESC[33m"
$Red    = "$ESC[31m"
$Blue   = "$ESC[34m"
$Cyan   = "$ESC[36m"
$Reset  = "$ESC[0m"

function Write-Phase([string]$PhaseLabel, [string]$Description) {
    Write-Host ""
    Write-Host "${Cyan}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${Reset}"
    Write-Host "${Cyan}  $PhaseLabel — $Description${Reset}"
    Write-Host "${Cyan}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${Reset}"
    Write-Host ""
}

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "${Blue}╔══════════════════════════════════════════════════════════════════════╗${Reset}"
Write-Host "${Blue}║       CinebotRL — IL + RL Two-Phase Launcher (RecomoProto1)         ║${Reset}"
Write-Host "${Blue}╚══════════════════════════════════════════════════════════════════════╝${Reset}"
Write-Host ""

# ---------------------------------------------------------------------------
# Resolve Isaac Lab path
# ---------------------------------------------------------------------------
if ($env:ISAAC_LAB_ROOT -and (Test-Path $env:ISAAC_LAB_ROOT)) {
    $IsaacLabPath = $env:ISAAC_LAB_ROOT
} else {
    $IsaacLabPath = "I:\isaaclab"
}

if (-not (Test-Path $IsaacLabPath)) {
    Write-Host "${Red}✗ Isaac Lab not found at: $IsaacLabPath${Reset}"
    Write-Host "  Set ISAAC_LAB_ROOT env var to your Isaac Lab installation path."
    exit 1
}
Write-Host "${Green}✓${Reset} Isaac Lab  : $IsaacLabPath"

# ---------------------------------------------------------------------------
# Resolve project path
# ---------------------------------------------------------------------------
$ProjectPath = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $ProjectPath "pyproject.toml"))) {
    foreach ($candidate in @("C:\Users\yanbo\wSpace\cinebotRL", "I:\wSpace\cinebotRL")) {
        if (Test-Path (Join-Path $candidate "pyproject.toml")) {
            $ProjectPath = $candidate
            break
        }
    }
}
if (-not (Test-Path (Join-Path $ProjectPath "pyproject.toml"))) {
    Write-Host "${Red}✗ Project root not found (no pyproject.toml found near script)${Reset}"
    exit 1
}
Write-Host "${Green}✓${Reset} Project     : $ProjectPath"

# ---------------------------------------------------------------------------
# Derived paths
# ---------------------------------------------------------------------------
$DemoDirAbs     = Join-Path $ProjectPath $DemoDir
$DemoFile       = Join-Path $DemoDirAbs "demos.npz"
$BCPolicyPath   = Join-Path $DemoDirAbs "bc_policy"   # pretrain_bc.py adds .zip
$BCPolicyZip    = "$BCPolicyPath.zip"

# GPU check
Write-Host ""
Write-Host "${Blue}GPU:${Reset}"
try {
    $gpuInfo = nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader
    foreach ($line in ($gpuInfo -split "`n")) {
        if ($line.Trim() -ne "") { Write-Host "  $line" }
    }
} catch {
    Write-Host "${Yellow}  ⚠  Could not detect GPU (nvidia-smi not available)${Reset}"
}

# ---------------------------------------------------------------------------
# Configuration summary
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "${Blue}Configuration:${Reset}"
Write-Host "  Task              : $Task"
Write-Host "  PPO Num Envs      : $NumEnvs"
Write-Host "  Headless          : $Headless"
Write-Host "  Total Timesteps   : $TotalTimesteps"
Write-Host "  Demo Episodes     : $DemoEpisodes"
Write-Host "  Demo Dir          : $DemoDirAbs"
Write-Host "  BC Epochs         : $BCEpochs"
Write-Host "  Skip Demo Phase   : $SkipDemoCollection"
Write-Host "  Skip BC Phase     : $SkipBC"
if ($PretrainedPolicy) {
    Write-Host "  Pretrained Policy : $PretrainedPolicy (override)"
}

# Disable Gymnasium plugin entrypoints to prevent ale_py crash on Windows
$env:GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS = "1"

# Change to Isaac Lab directory for isaaclab.bat invocations
Set-Location $IsaacLabPath

# ---------------------------------------------------------------------------
# PHASE 0 — Collect expert demonstrations
# ---------------------------------------------------------------------------
if (-not $SkipDemoCollection) {
    Write-Phase "PHASE 0" "Collecting expert demonstrations (${DemoEpisodes} episodes)"

    $ILDatasetScript = Join-Path $ProjectPath "src\rl_platform\tasks\recomoproto1\il_dataset.py"
    $phase0Args = @(
        "-p", $ILDatasetScript,
        "--task",         $Task,
        "--num_envs",     "1",
        "--num_episodes", $DemoEpisodes,
        "--output_dir",   $DemoDir
    )
    if ($Headless) { $phase0Args += "--headless" }

    Write-Host "${Yellow}Executing: .\isaaclab.bat $phase0Args${Reset}"
    & .\isaaclab.bat @phase0Args

    if ($LASTEXITCODE -ne 0) {
        Write-Host "${Red}✗ Phase 0 failed (exit code $LASTEXITCODE)${Reset}"
        exit $LASTEXITCODE
    }
    Write-Host "${Green}✓ Phase 0 complete — demos saved to $DemoFile${Reset}"
} else {
    Write-Host ""
    Write-Host "${Yellow}⊘  Phase 0 skipped (--SkipDemoCollection)${Reset}"
    if (-not (Test-Path $DemoFile)) {
        Write-Host "${Red}✗ Demo file not found at: $DemoFile${Reset}"
        Write-Host "  Run without -SkipDemoCollection or check --DemoDir path."
        exit 1
    }
    Write-Host "  Using existing demos: $DemoFile"
}

# ---------------------------------------------------------------------------
# PHASE 1 — BC pre-training (pure Python, no Isaac Lab)
# ---------------------------------------------------------------------------
if (-not $SkipBC -and -not $PretrainedPolicy) {
    Write-Phase "PHASE 1" "Behavioural Cloning pre-training (${BCEpochs} epochs)"

    # Detect obs_dim from the saved demo file using Python
    $obsDimCmd = "import numpy as np; d=np.load('$DemoFile'); print(d['observations'].shape[1])"
    $ObsDim = python -c $obsDimCmd 2>$null
    if (-not $ObsDim) {
        Write-Host "${Yellow}⚠  Could not auto-detect obs_dim; defaulting to 70.${Reset}"
        $ObsDim = "70"
    }
    Write-Host "  Detected obs_dim: $ObsDim"

    $BCScript = Join-Path $ProjectPath "scripts\reinforcement_learning\bc\pretrain_bc.py"
    $phase1Args = @(
        $BCScript,
        "--demo_file",    $DemoFile,
        "--obs_dim",      $ObsDim,
        "--output_path",  ($BCPolicyPath -replace "\\", "/"),
        "--epochs",       $BCEpochs
    )

    Write-Host "${Yellow}Executing: python $phase1Args${Reset}"
    python @phase1Args

    if ($LASTEXITCODE -ne 0) {
        Write-Host "${Red}✗ Phase 1 (BC) failed (exit code $LASTEXITCODE)${Reset}"
        exit $LASTEXITCODE
    }

    if (-not (Test-Path $BCPolicyZip)) {
        Write-Host "${Red}✗ BC policy not found at: $BCPolicyZip${Reset}"
        exit 1
    }
    Write-Host "${Green}✓ Phase 1 complete — BC policy saved to $BCPolicyZip${Reset}"

    $PretrainedPolicy = $BCPolicyPath   # pass without .zip — SB3 adds it internally
} elseif ($SkipBC -and -not $PretrainedPolicy) {
    Write-Host ""
    Write-Host "${Yellow}⊘  Phase 1 skipped (--SkipBC)${Reset}"
    if (Test-Path $BCPolicyZip) {
        Write-Host "  Using existing BC policy: $BCPolicyZip"
        $PretrainedPolicy = $BCPolicyPath
    } else {
        Write-Host "${Yellow}  No BC policy found at $BCPolicyZip — PPO will start from scratch.${Reset}"
    }
} else {
    Write-Host ""
    Write-Host "${Yellow}⊘  Phase 1 skipped (--SkipBC or --PretrainedPolicy provided)${Reset}"
}

# ---------------------------------------------------------------------------
# PHASE 2 — PPO fine-tuning
# ---------------------------------------------------------------------------
Write-Phase "PHASE 2" "PPO fine-tuning (${TotalTimesteps} timesteps)"

$TrainScript = Join-Path $ProjectPath "scripts\reinforcement_learning\sb3\train.py"
$phase2Args = @(
    "-p", $TrainScript,
    "--task",             $Task,
    "--num_envs",         $NumEnvs,
    "--total_timesteps",  $TotalTimesteps
)
if ($Headless)          { $phase2Args += "--headless" }
if ($PretrainedPolicy)  { $phase2Args += @("--pretrained_policy", $PretrainedPolicy) }

Write-Host "${Yellow}Executing: .\isaaclab.bat $phase2Args${Reset}"
& .\isaaclab.bat @phase2Args

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "${Green}╔══════════════════════════════════════╗${Reset}"
    Write-Host "${Green}║  ✓ IL + RL pipeline complete!        ║${Reset}"
    Write-Host "${Green}╚══════════════════════════════════════╝${Reset}"
} else {
    Write-Host ""
    Write-Host "${Red}╔══════════════════════════════════════╗${Reset}"
    Write-Host "${Red}║  ✗ Phase 2 failed (code $LASTEXITCODE)  ║${Reset}"
    Write-Host "${Red}╚══════════════════════════════════════╝${Reset}"
    exit $LASTEXITCODE
}
