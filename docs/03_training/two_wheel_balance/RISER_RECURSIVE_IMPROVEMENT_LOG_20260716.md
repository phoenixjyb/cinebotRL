# Two-wheel riser recursive improvement log

This log is append-only. Every round must preserve the balance-first priority
and must state whether its candidate was accepted or rejected.

## Round 0: isolated riser baseline

- Change: removed the three arm joints; added the 1.2 m, 1.0 m/s riser and the
  deterministic physical-gimbal adapter.
- Result: Gate 0, static height, and riser dynamics passed on the original
  identity gimbal mount.
- Decision: superseded. The identity mount made corrected camera attitudes
  infeasible and those Isaac results must be rerun after the mount repair.
- Lesson: dynamics evidence is asset-version-specific.

## Round 1: fixed gimbal bracket and corrected stage

- Change: replaced the identity mount with the accepted-corpus median removed
  arm orientation; reproduced the authoritative corrected stage locally.
- Result: 62 accepted corrected cases and 17 rejected cases were preserved;
  old physical-gimbal labels remained quarantined. Case 1 path-aligned
  attitude error fell to 0.0068 degrees.
- Decision: provisionally accepted for pure kinematics; Isaac/USD validation
  remains pending because `192.168.100.98:2222` is unreachable.
- Lesson: use semantic DFR targets and physical `cam_link` FK, never old motor
  joint labels as the learned target.

## Round 2: consistent reference gate

- Hypothesis: contradictory tolerances and an incorrect lateral-velocity
  projection were producing false negatives.
- Change: tied solver convergence to the declared attitude gate, added p95
  position/attitude metrics, added numerical tolerance at rate bounds, and
  projected unicycle chord velocity at interval midpoint yaw.
- Verification: 17 local pure tests passed. On representative corrected cases
  with Gate-3 bounds, cases 1 and 73 passed and case 31 failed.
- Decision: accepted. The change corrected measurement only and did not hide
  case 31.
- Remaining case-31 evidence: position p95 0.856 m and maximum 0.894 m; the
  chassis rotates from about -0.39 rad to -2.38 rad while the legacy simulated
  roll joint reaches its +1.57 rad limit.
- Lesson: the old serial-joint envelope is not proven equivalent to RS4
  attitude control. Do not widen those joints blindly.

## Next round

- Hypothesis: case 31 is feasible under the real RS4 camera-attitude contract
  but appears infeasible through the legacy CAD motor-joint envelope.
- Required evidence: explicit DFR-to-FLU axis/sign mapping, separate command
  attitude and physical-joint state, and an attitude-envelope smoke test.
- Stop rule: no PPO and no full 62-case Isaac run until the attitude adapter
  passes the representative kinematic gate and the regenerated USD passes
  Gate 0 on `.98`.

## Round 3: semantic RS4 attitude envelope

- Hypothesis: case 31 is feasible under the deployed RS4 attitude command
  contract and only fails through the legacy CAD motor-joint envelope.
- Change: implemented the explicit ZYX resolver and deployment mapping
  `Ronin yaw <- rot_x`, `roll <- rot_y`, `pitch <- rot_z`; fitted a fixed
  chassis-to-DFR zero-command basis from accepted corrected references; added
  sequence-level Euler-branch optimization.
- First result at `0.4 rad/s` base yaw: all 62 cases fit the position envelope
  and hard `360 deg/s` rate, but 13 exceeded the conservative `24 deg/s` p95
  filming rate.
- Accepted bounded candidate: reduce reference chassis-yaw cap to
  `0.25 rad/s`. The complete accepted stage then passed `62/62` command
  envelope, hard-rate, and filming-rate checks.
- Evidence: `evidence_20260716_riser_rs4_attitude/summary.json` and
  `evidence_20260716_riser_rs4_attitude/cases.csv`.
- Decision: accept the pure attitude contract and `0.25 rad/s` planning cap.
  This does not authorize training because the Isaac articulation and USD have
  not yet adopted or rendered the adapter.
- Lesson: lower chassis yaw is both more balance-compatible and avoids forcing
  the gimbal to cancel a fast base rotation. Retiming alone did not fix this
  because the base yaw controller remained rate-saturated.

## Next round after Round 3

- Implement the deterministic RS4 attitude adapter in the riser Isaac path,
  while keeping physical `cam_link` as the observation/reward frame.
- Regenerate the URDF/USD and rerun Gate 0, static heights, riser dynamics, and
  rendered cases 1, 31, and 73 on `.98`.
- Keep PPO blocked until those dynamic gates pass without balance regression.

## Round 4: full-pose RS4 reference portfolio

- Hypothesis: a fixed path-heading planner and a joint adaptive yaw planner
  cover complementary trajectory families, provided candidate selection first
  enforces semantic attitude, proxy-rate, riser, and nonholonomic constraints.
- First joint-adaptive result: `55/62`. It repaired all 11 failures from the
  fixed planner but regressed seven previously feasible cases through local
  one-step yaw choices. Three regressions were position errors and four were
  small `24 deg/s` proxy-rate overshoots.
- Change: retained both deterministic candidates and selected them
  lexicographically, rejecting any candidate with an attitude or proxy-rate
  violation before comparing p95 and maximum position error.
- Final pure-kinematic result: `62/62`; 15 cases selected `fixed_path` and 47
  selected `joint_adaptive`. Worst p95 position error was `0.146477 m`, worst
  maximum position error was `0.155496 m`, worst proxy-joint rate was
  `0.407419 rad/s`, and semantic attitude error remained numerical noise.
- Evidence:
  `evidence_20260716_riser_rs4_pose_portfolio/summary.json` and
  `evidence_20260716_riser_rs4_pose_portfolio/cases.csv`.
- Decision: accepted as the corrected-62 reference generator. This authorizes
  regenerated-asset Isaac validation but does not authorize PPO or residual
  training.
- Lesson: a single greedy planner is not robust across all trajectory
  families. Complementary deterministic teachers are useful only when safety
  feasibility is ranked before tracking error.

## Next round after Round 4

- Transfer the updated URDF, semantic RS4 proxy, and planner to the isolated
  `.98` worktree and regenerate the USD.
- Rerun Gate 0, camera heights `0.6/1.2/1.8 m`, riser speeds
  `0.1/0.25/0.5/1.0 m/s`, and rendered full-pose cases 1, 31, and 73.
- Stop immediately on a balance, finite-state, articulation, camera-frame, or
  proxy-rate regression. Keep PPO blocked until this dynamic gate is green.

## Round 5: continuous RS4 yaw-proxy target

- Diagnosis: the original proxy-rate metric treated Ronin yaw as cyclic while
  the Isaac articulation target was a bounded revolute coordinate. Cases 39,
  49, and 50 crossed the wrapped `+/-pi` command boundary in the fixed-heading
  diagnostic, so a nominal two-degree command change could have appeared to a
  position servo as a nearly 360-degree reversal.
- Change: made only the Ronin-yaw proxy articulation coordinate continuous,
  kept the DJI-facing semantic command wrapped to `[-pi, pi]`, and unwrapped
  generated proxy targets to the nearest equivalent coordinate.
- Verification: 28 pure tests passed. The complete corrected-62 portfolio
  again passed `62/62`; worst raw proxy-target rate was `0.407419 rad/s` and
  worst raw target step was `0.020489 rad`. Raw and cyclic rate metrics now
  agree.
- Decision: accepted. This closes a simulation-command ambiguity without
  widening the physical DJI command envelope or changing camera FK.
- Lesson: cyclic command semantics and articulation position coordinates must
  be audited separately; a wrapped diagnostic alone is insufficient evidence
  for a servo target.

## Next round after Round 5

- Sync the validated file manifest to `.98`, regenerate USD, and require the
  imported Ronin-yaw proxy to have no authored angular limits.
- Run the fail-fast Gate 0--2 script, then add and execute rendered semantic
  full-pose replay for cases 1, 31, and 73.
- Keep PPO and residual-DNN training blocked until the dynamic and rendered
  gates are green.

## Round 6: deterministic Isaac playback package

- Change: added a versioned, self-contained playback format containing the
  corrected target position and semantic DFR attitude, base pose, riser
  reference, continuous RS4 proxy reference, and all feed-forward rates.
- Exported representative cases 1, 31, and 73 with SHA-256 hashes. Their full
  durations are 25.124 s, 14.641 s, and 7.919 s respectively; no temporary
  teacher-stage path is required at playback time.
- Added an Isaac playback gate that initializes the complete articulation,
  drives the existing balance LQR, commands riser and proxy position plus
  velocity feed-forward, measures physical `cam_link`, and checks position,
  attitude, pitch, servo error, saturation, termination, and full-duration
  completion.
- The phase governor takes the minimum of tracking-error progress and
  pitch-safety progress. It cannot trade balance margin for trajectory speed.
- Added offscreen recording for each representative case with target and path
  markers. The fail-fast remote runner requires the non-rendered three-case
  metric gate before recording any MP4.
- Verification: 30 pure tests passed; the 44-file transfer manifest is
  complete and both shell runners pass syntax validation.
- Remaining evidence boundary: none of this is Isaac runtime evidence until
  the regenerated USD and playback run on `.98`.

## Resume condition after Round 6

- `192.168.100.98:2222` must become reachable.
- Run `scripts/two_wheel_balance/sync_riser_to_98.sh` from the staging tree,
  then execute `scripts/two_wheel_balance/run_riser_remote_gates.sh` in the
  isolated remote worktree.
- Do not regenerate the remaining 17 teachers or begin residual training until
  Gate 0--3 and all three render audits pass.

## Round 7: proxy-rate planning margin

- Diagnosis: case 73 passed the pure position gate at p95 `0.146477 m`, only
  3.5 mm below the dynamic threshold. Its joint-adaptive candidate tracked at
  sub-millimetre error but exceeded the public `24 deg/s` proxy limit by about
  0.14%, so the portfolio correctly selected the poorer fixed candidate.
- Change: retained the unchanged public `24 deg/s` gate and planned the
  adaptive candidate against a `0.995` internal rate margin. No actuator limit
  or acceptance threshold was relaxed.
- Representative result: case 73 switched to `joint_adaptive`; position p95
  improved from `0.146477 m` to `0.000723 m`, maximum error to `0.001970 m`,
  and raw proxy rate remained bounded at `0.417392 rad/s`.
- Full corrected-62 result: `62/62` still pass; strategy mix is 13 fixed and
  49 adaptive. Corpus worst p95 position error improved from `0.146477 m` to
  `0.120690 m`, and worst maximum improved from `0.155496 m` to `0.132305 m`.
  Case 24 also improved from p95 `0.056963 m` to `0.011673 m`.
- Export boundary: the exporter now independently reconstructs physical
  `cam_link` FK and refuses any plan that regresses position, attitude,
  nonholonomic motion, riser bounds/rate, or raw proxy-target rate.
- Decision: accepted. This increases dynamic tracking margin without changing
  the balance hierarchy or hardware command envelope.
- Remaining boundary: all results in this round are pure kinematics; `.98`
  Isaac validation is still required before Gate 0--3 can be called green.

## Round 8: regenerated asset and riser dynamics on `.98`

- Change: synced the manifest-scoped riser worktree, regenerated URDF/USD, and
  made the Gate runner parse every JSON `passed` field because Windows Isaac
  can return shell success for a failed Python gate.
- Gate 0 passed with 28 kg total mass, 14 rigid bodies, 13 joints, no arm DOFs,
  physical and semantic camera frames, and an effectively unbounded continuous
  Ronin-yaw proxy in USD.
- Static height balance passed at 0.6, 0.9, and 1.8 m. Maximum pitch was
  1.846 degrees and maximum camera-height error was 0.02395 m.
- All 0.1, 0.25, 0.5, and 1.0 m/s riser round trips passed. Maximum pitch was
  1.918 degrees, maximum camera-height error was 0.02501 m, and neither wheel
  nor riser saturated.
- Decision: accept Gate 0--2 for the regenerated asset.
- Lesson: shell exit status is not authoritative across WSL/Windows Isaac;
  gate JSON must be parsed explicitly.

## Round 9: semantic RS4 realization and online base-tilt compensation

- First Gate-3 failure: position and balance passed, but treating the virtual
  attitude coordinates as 10 Nm physical motor shafts caused 59--90% proxy
  saturation and 8--14 degree camera-attitude lag.
- Diagnosis: these coordinates are semantic DJI attitude commands. Physical
  motor-joint velocity is owned by the Ronin controller and is not a teacher
  label or policy action. A rate-audited ideal state adapter reduced proxy
  realization error from 9.36 degrees to 0.11 degrees.
- Second failure: physical camera attitude still had 9.03 degree p95 error
  because the exported proxy angles assumed an upright chassis.
- Change: solve proxy coordinates online from current full root quaternion and
  desired semantic DFR attitude, preserve yaw branch continuity, and cap
  internal stabilization at 360 deg/s. Continue to measure reward and
  observation from physical `cam_link` through Option B.
- Case-73 result: physical attitude p95 0.148 degrees, proxy realization p95
  0.131 degrees, zero IK failures, and 38.37 deg/s peak internal compensation.
- Decision: accept the separate contracts
  `semantic_attitude_position_only` for hardware and
  `rate_audited_ideal_state_adapter` for simulation.
- Lesson: world-attitude control must compensate live base roll/pitch; offline
  upright-base proxy angles cannot be replayed as final gimbal coordinates.

## Round 10: representative Gate 3 and D3D12 rendering

- Non-rendered cases 1, 31, and 73 all passed full duration. Position p95 was
  0.0905, 0.0833, and 0.0929 m; attitude p95 was 0.1637, 0.1548, and 0.1501
  degrees; maximum pitch remained below 5 degrees.
- Internal compensation stayed below 41 deg/s for all cases against the
  separate 360 deg/s hard envelope, with zero IK failures and zero proxy
  saturation.
- The first render launch crashed in `rtx.scenedb` after selecting Vulkan.
  Explicitly selecting the known D3D12 headless rendering experience and
  allowing a five-second GPU-context cooldown fixed the launch.
- All three render-time JSON gates passed. H.264 1280x720 real-time 50 fps
  derivatives were verified and copied to
  `/Users/yanbo/Downloads/cinebotRL_two_wheel_riser_20260716`.
- Evidence: `evidence_20260716_riser_gate0_gate3_online_comp/`.
- Decision: accept the deterministic corrected-62 representative milestone.
  PPO remains blocked because 17 trajectories still require corrected teacher
  regeneration and all-79 deterministic evaluation.

## Next round after Round 10

- Regenerate the remaining 17 cases only from corrected semantic DFR targets;
  never restore quarantined old NPZ or physical-gimbal labels.
- Run pure all-79 feasibility first, then deterministic Isaac smoke cases from
  every repaired failure family. Repair structural reference issues before any
  residual-DNN dataset is built.
- Keep PPO blocked until all 79 corrected trajectories pass full-duration
  deterministic gates without a higher-priority balance or riser regression.

## Round 11: strict all-79 admission and acquisition recovery

- Source audit: adopted the corrected all-79 v3 semantic teacher package. Its
  per-case schema exposes base-arm state and separate world-DFR attitude, but
  no source action or physical DJI motor-joint label is admitted as a riser
  policy target. The unexported MATLAB initialization sample is also excluded.
- First result: directly placing the full trajectories behind the riser home
  state passed `0/79`; all cases violated the synthetic acquisition-rate
  contract. This candidate was rejected.
- Structural change: regenerate home-to-first-target acquisition from physical
  riser FK and physical `cam_link` converted to semantic DFR. Apply any
  low-height translation before acquisition generation.
- Second result: `72/79` passed. Cases 23, 24, 41, and 72 only failed inside
  synthetic acquisition; cases 18 and 79 required whole-path slowing; case 50
  exposed a double-applied vertical shift.
- Bounded repair: whole-path scales are case 18 `1.5x` and case 79 `1.5x`;
  acquisition-only scales are case 23 `6.5x`, case 24 `4.5x`, case 41 `1.3x`,
  and case 72 `1.5x`; case 50 applies its shift before home acquisition. No
  public gate or actuator limit changed.
- Final pure result: `79/79` passed. Strategy mix is 30 fixed-path and 49
  joint-adaptive. Worst position p95 is `0.149015 m`, maximum is `0.152459 m`,
  and proxy rate is `0.413218 rad/s` against the unchanged limit.
- Decision: accept the corrected all-79 deterministic stage. Training and PPO
  remain blocked pending dynamic repaired-family validation and a residual
  dataset contract.
- Lesson: removed-arm acquisition is robot-specific executable scaffolding,
  not a teacher label. It must be regenerated and provenance-audited.

## Round 12: repaired-family Isaac dynamics

- Change: exported self-contained playback plans for cases 18, 23, 24, 41,
  50, 72, and 79, then ran all seven through the regenerated 28 kg Isaac asset
  with balance LQR, riser control, online semantic-attitude IK, and the phase
  governor.
- Result: `7/7` passed over 39,430 steps with full-duration completion, no
  termination, no attitude-IK failure, and zero action/riser saturation. Worst
  proxy saturation ratio was the accepted `0.000762` in case 72.
- Dynamic worst metrics: position p95 `0.131495 m`, position maximum
  `0.144621 m`, attitude p95 `0.182725 deg`, attitude maximum `0.225993 deg`,
  pitch maximum `5.968319 deg`, riser-servo p95 `0.011064 m`, and internal
  proxy rate `62.243368 deg/s`.
- Evidence: `evidence_20260716_riser_all79_recovery/`.
- Decision: accept the all-79 recovery milestone. This closes the known
  repaired failure families but is not evidence for a residual DNN policy.
- Lesson: dynamic smokes should be selected from each structural repair family,
  not only from easy or representative trajectories.

## Next round after Round 12

- Define a versioned residual observation/action contract from deterministic
  executed state. Do not copy source GIK actions or physical gimbal labels.
- Split by complete case, never by adjacent samples, to prevent trajectory
  leakage across train, validation, and holdout sets.
- Require dataset schema, dimensional, finite-value, timing, leakage, and
  deterministic-baseline regression gates before any learned-policy rollout.
- Keep PPO blocked. The next accepted artifact is a validated residual dataset,
  not another blind training run.

## Round 13: executed-state residual dataset smoke

- Contract decision: the DNN does not directly replace the balance controller.
  It predicts bounded high-level residuals `delta-vx`, `delta-wz`, and a riser
  target increment with scales `0.20 m/s`, `0.40 rad/s`, and `0.10 m`.
  Cascaded LQR still owns wheel effort and the semantic DJI adapter remains
  deterministic.
- Observation contract: 26 deployable values from physical LQR state, body-frame
  base and camera errors, physical-camera attitude error, riser state/error,
  trajectory feed-forward, phase/governor state, and prior residual action.
- Collection rule: record dense pre-action Isaac state only; do not copy GIK
  actions or physical gimbal labels. A per-case NPZ is saved only after the
  unchanged dynamic replay gate passes.
- Result: repaired cases 18, 23, 24, 41, 50, 72, and 79 passed `7/7` and produced
  39,430 rows. No action channel clipped; absolute maxima were `0.754922`,
  `0.594813`, and `0.121189`.
- Dataset audit: finite values, 26-by-3 dimensions, no case leakage, and teacher
  command reconstruction error `1.1921e-7`. The deterministic split uses five
  train cases, case 23 for validation, and case 18 for holdout.
- Decision: accept the collection and schema pipeline, but reject this seven-case
  corpus as a training dataset. Training and PPO remain unauthorized.
- Evidence: `evidence_20260716_riser_residual_dataset_smoke/`.
- Lesson: the safe learned boundary is above the proven LQR, and dataset admission
  must be coupled to successful physical-camera dynamic replay.

## Next round after Round 13

- Export self-contained plans for all 79 corrected trajectories and collect all
  79 through the same dense Isaac path.
- Require zero failed captures, zero split leakage, finite values, bounded labels,
  and exact teacher-command reconstruction before offline BC.
- Do not start BC or any learned-policy rollout from the seven-case smoke.

## Round 14: phase-consistent playback and case-15 recovery

- The first resumable all-79 capture stopped correctly at case 15. Cases 1--14
  passed, but case 15 produced `0.209044 m` camera-position p95 against the
  unchanged `0.15 m` gate. Completion, balance, attitude, riser, proxy, IK, and
  saturation checks all passed.
- Root cause: the phase governor slowed phase time but continued to send the
  unscaled source feed-forward derivatives. Case 15 reached a progress scale
  near `0.25`, so the target advanced slowly while base velocity and yaw
  feed-forward remained at full trajectory speed.
- Contract fix: scale base linear/yaw, riser, and proxy derivatives by the same
  progress scale used to advance phase. Use these scaled values consistently in
  deterministic commands, observations, residual labels, and policy baselines.
- Tracking fix: promote `riser_phase_consistent_v2` with outer-loop gains
  `along=1.6`, `cross=1.5`, and `yaw=1.2`. Wheel-speed, yaw-rate, phase, riser,
  gimbal, and acceptance limits are unchanged.
- Plan fix: case 15 alone uses a manifest-recorded planning yaw cap of
  `0.20 rad/s` instead of `0.25 rad/s`. Its pure p95 improves from `0.110273 m`
  to `0.102839 m`; all other 78 playback NPZ hashes remain unchanged.
- Case-15 Isaac result: position p95 `0.140834 m`, maximum `0.143522 m`, peak
  pitch `5.1849 deg`, peak yaw error `3.9377 deg`, no termination, and zero
  action saturation.
- Regression result: representative cases 1, 15, 31, and 73 pass `4/4`.
  Repaired cases 18, 23, 24, 41, 50, 72, and 79 pass `7/7` over 34,002 steps.
  Repaired-family worst position p95 is `0.122759 m`, maximum is `0.127505 m`,
  attitude p95 is `0.199269 deg`, and pitch maximum is `4.876883 deg`.
- Rejected work: stronger yaw-loop gains, direct camera-point feedback, blended
  camera feedback, and larger cross-track gains all failed to beat the fixed
  gate and were not promoted.
- Decision: accept the v2 deterministic baseline. Discard the old cases 1--14
  captures because their phase/action contract is stale. Start a fresh all-79
  dataset stamp; BC and PPO remain blocked until it passes and merges.

## Next round after Round 14

- Run `20260716_residual_all79_phase_v2` from case 1 using only
  `20260716_all79_playback_inputs_v2`.
- Require every resumable gate to report `riser_phase_consistent_v2` and
  `derivatives_scaled_by_progress_v1`; record the plan-manifest SHA-256 in the
  merged summary.
- Only after `79/79` dynamic captures and a case-disjoint merge pass may the
  offline BC candidate be trained. PPO remains unauthorized.

## Round 15: upstream trajectory-integrity quarantine

- Upstream audit proved that `no_obstacle_episode_*_split_teacher_v2.npz`, the
  ancestor of the all-79 riser stage and v4 plans, contains truncated/resampled
  GIK trajectories rather than the complete authoritative source sequences.
- Concrete examples: episode 1 is 253 poses / 25.123689 s versus authoritative
  256 / 4.634756 s; episode 4 is 224 / 22.291202 s versus 723 / 14.042191 s;
  episode 7 is 174 / 17.247189 s versus 663 / 12.940941 s. Their path lengths
  are also reduced from 2.452/3.849/3.808 m to 1.092/2.007/2.260 m.
- The clean Isaac capture was paused after active case 65 completed. Case 66
  was never started. The complete 65 JSON gates and 65 NPZ files were preserved
  under `20260716_residual_all79_phase_v3_clean_QUARANTINED_UPSTREAM_TRUNCATED_SOURCE_20260717`.
- Decision: none of the quarantined labels is eligible for desired-trajectory
  BC, holdout, PPO, or policy promotion. Frozen-LQR, governor, residual-envelope,
  command-reconstruction, and dynamic-safety findings remain controller
  diagnostics only.
- Replacement contract: `exact_source_v1` requires N authoritative source
  poses/timestamps, N ordered retargeted waypoint states, N-1 transitions,
  explicit initialization separation, and a separate quality/safety gate.
- The episode 1/4/7 canary package passes transport integrity but is explicitly
  `valid_for_training=false`; it cannot unblock collection.

## Next round after Round 15

- Wait for a quality-qualified 79-case `exact_source_v1` teacher package with
  package-level and per-case `valid_for_training=true`.
- Run `validate_riser_exact_source_manifest.py`; reject resampling, missing
  timestamps, blended initialization, failed quality gates, or any case-count
  mismatch before retargeting.
- Regenerate plans and residual captures from empty exact-source namespaces.
  Do not resume v4 plans or the quarantined corpus. Keep BC and PPO blocked.

## Round 16: exact-source Gate A/B regeneration

- Gate A now distinguishes reference ingest from training admission. The
  SHA-pinned all-79 `exact_source_v1` package passes reference ingest and is
  correctly rejected from training.
- The Gate B exporter preserves original source timestamps, positions, and
  semantic DFR `xyzw` quaternions verbatim. Every plan carries a strictly
  increasing explicit execution schedule, a complete ordered anchor map, and
  a separate empty initialization segment.
- Ep1/4/7 canaries preserve `1,642/1,642` anchors and `10.109224 m` of path with
  zero mapped-position error. Heading-aware retiming improves pure kinematic
  quality, but ep1 and ep4 still fail at least one provisional quality check.
- The all-79 namespace preserves `71,038/71,038` anchors, `618.304657 m` of
  source path, and all original timestamps. Plan manifest SHA-256 is
  `940434d8caa5f85eb8c67d38d09a0894927a50b51fbb380b570d9e724fffe001`.
- Only `28/79` plans pass the provisional pure-kinematic gate. The remaining
  `51/79` are explicit rejects; no threshold was relaxed and no historical
  fallback was inserted.
- Decision: apply the `<70 accepted` stop rule. Gate C Isaac capture, residual
  labels, BC, and PPO remain unstarted and unauthorized.
- Evidence: `evidence_20260717_riser_exact_source_gate_a_b/` and
  `RISER_EXACT_SOURCE_GATE_A_B_STATUS_20260717.md`.
- Lesson: exact desired-trajectory identity is necessary but not sufficient.
  Retarget feasibility must be repaired structurally before dynamic learning
  data can be collected.

## Next round after Round 16

- Diagnose the 51 pure-kinematic rejects by failure family: nonholonomic
  position error, proxy-rate excess, riser bounds, and combinations thereof.
- Change planner structure or explicit execution timing only; never drop or
  reorder source anchors and never substitute quarantined plans.
- Rerun all 79 in a fresh Gate B namespace. Enter Gate C only if at least 70
  cases pass the unchanged pre-dynamic quality screen.
- Keep BC and PPO blocked.

## Round 17: preview/coupled Gate B recovery

- Added preview steering candidates, selected-plan execution retiming, and
  bounded constant vertical placement. Source anchors, source timestamps, and
  all acceptance thresholds remain unchanged.
- Bound the independent vertical-workspace audit SHA-256
  `51c2e60e11e53cf8b1884d0d01bec61df4f8cacec9d7cf35bb3b5b08f81447ab`.
  It confirms `78/79` sources are vertically compatible and ep27 is the sole
  irreducible span reject.
- All-79 v2 preserves `79/79` exact-source plans and raises the unchanged
  kinematic gate from `28/79` to `66/79`. Manifest SHA-256 is
  `c37ab4762e91492309f7c80a54df61379137282395f8bc3b482adb605ceca296`.
- Targeted proxy/preview recovery passes cases `52,69,74,75,76`. The recovery
  uses an explicit `6 deg/s` timing design margin while the acceptance gate
  stays at `24 deg/s`.
- The final hash-audited portfolio preserves the accepted 66, substitutes only
  those five passing recoveries, and reaches `71/79`. Portfolio manifest
  SHA-256 is
  `851a7b2751cd397ba35daf57d1a8c6971fb14ed0186683af48d3c6109090570a`.
- Explicit remaining rejects are `6,13,18,21,22,27,55,64`. Ep6/18 are
  terminal XY maximum-only; ep21/22 are paired p95-only; ep27 is vertical-span
  incompatible; ep13/55 are proxy-only; ep64 is mixed p95/proxy.
- Ep77's upstream seed was hash-verified and agrees with the riser source and
  base-path diagnosis, but remains reference-only with no learned actions.
- Decision: the Gate C candidate-count requirement is met. Gate C dynamic
  execution, residual capture, BC, and PPO have not started.

## Next round after Round 17

- Commit and freeze the exact Gate B code and portfolio identity.
- Run Gate C only on the 71 accepted cases with frozen LQR and unchanged
  dynamic safety gates.
- Collect residual labels only from dynamically passing cases, then recompute
  the residual envelope before freezing action scales.
- Keep BC/PPO blocked.

## Round 18: Gate C timing, gate order, and GPU ownership repair

- Frozen Gate B portfolio admission passed for cases `1,52,74,77` from clean
  pushed commit `abc87fd`; no dataset or learned policy was requested.
- Provisional dynamic cases 1 and 52 passed all physical checks. Case 52 ran
  `70,952` steps with position p95/max `0.108022/0.153767 m`, attitude p95/max
  `0.151288/0.211011 deg`, pitch max `5.569997 deg`, zero saturation, and no
  termination or dataset.
- Evidence audit found runtime `source_duration_s` contained retimed execution
  duration. Commit `af4a4da` preserves and validates both clocks and reports
  them separately in riser and whole-body runtime JSON.
- Case 74 exposed prospective normalized residual
  `[1.00200645,0.19818327,0.0356378]`. This was initially misclassified as a
  dynamic reject. The deterministic commands had already been computed and
  the normalized label was never applied.
- Commit `d4c2097` separates raw residual command, frozen-envelope diagnostics,
  and learned action normalization. Gate C physics continues without clipping
  or labels; dataset capture still rejects envelope overflow.
- The first v2 reseal was stopped because a differential run owned the GPU.
  Its partial case-1 log is retained as invalid shared-GPU evidence, with no
  completed runtime JSON.
- Commit `f08271e` enforces exclusive GPU ownership before namespace creation
  and before every case. A live test rejected the competing run with exit `5`
  and left the v3 namespace absent.
- Decision: do not infer case-74 dynamic quality yet. Hold all riser Isaac
  launches while the differential final exclusive rerun uses commit `3e820e8`.
  Keep residual capture, Gate D, BC, and PPO closed.

## Next round after Round 18

- Wait for explicit GPU release.
- Run `20260717_gate_c_canary_v3_exclusive_timing_resealed` for
  `1,52,74,77` from clean pushed `f08271e` or a verified descendant.
- Require exclusive GPU ownership, both timing fields, unchanged safety gates,
  no dataset, and independent dynamic versus label-envelope outcomes.
- If all four dynamically pass, expand Gate C across the 71 accepted plans.
- Recompute the raw residual envelope only from dynamically passing cases;
  do not widen frozen action scales before that audit.

## Round 19: final case-74 dynamic reject and continuous-yaw diagnosis

- After explicit differential GPU release, the exclusive guarded runner opened
  `20260717_gate_c_case74_77_v3_exclusive_timing_resealed` for cases 74 then
  77. Case 74 ran first; case 77 was fail-closed and never started.
- Case 74 is a final physical Gate C reject, not a label-envelope-only reject.
  It stopped at phase `46.961485/188.546638 s` after `13,659` steps with one
  forbidden-body contact. Position p95/max was `0.546011/1.842571 m`, pitch
  max `15.802575 deg`, attitude max `32.765379 deg`, and proxy servo error max
  `719.887930 deg`.
- Source and execution clocks are independently correct at `11.373883 s` and
  `188.546638 s`. The runtime JSON SHA-256 is
  `9bec49cf68d37d100b800e6505f5d0e5b6df2d1af30cd5f4e89bbe10d7794eb4`.
  Final status SHA-256 is
  `b6bbd2dc25783ddff8364bafea1a23b06555d7f2dfe089095dfad29304cde4ee`.
- Label-envelope admission independently failed: raw maxima
  `[0.4011729574,0.3081566074,0.0125294506]`, normalized maxima
  `[1.3372431914,0.7703915184,0.1252945059]`. These labels were not applied;
  residual action remained zero and no dataset or training was created.
- Trace evidence localized a continuous-yaw branch error. The unwrapped
  semantic proxy target reached about `+401.749 deg`, while PhysX reported the
  equivalent `-317.933 deg`. Raw subtraction created a false `~720 deg` servo
  error, saturated the proxy drive, and preceded chassis divergence.
- CPU-only repair keeps the unwrapped semantic target as the authoritative DJI
  attitude trajectory, maps only the PhysX target to its nearest equivalent
  `2*pi` branch, and wraps continuous-yaw servo diagnostics. It does not change
  source anchors, execution timing, LQR gains, physical gates, residual scales,
  clipping, or training admission.
- Added a hash-bound CPU scope audit over all 71 admitted plans. It finds that
  `45/71` require nearest-equivalent branch handling; all 45 contain canonical
  branch crossings. Case 71 can differ naively by `720 deg`. Cases 74/75/76
  share the largest valid unwrapped step at `178.000692 deg`, confirming case
  74 as the strongest next structural canary rather than an arbitrary retry.
- Scope summary SHA-256 is
  `7e357237237cb459a5b5c47f630852b4a179b955e022f4d55835c4949413fcbe`;
  CSV SHA-256 is
  `67bbfe164f6376189843517e9195437fb90b51e0186e8d1dfa32c4db15fc55cc`.
  All plan hashes, semantic continuity checks, and nearest-branch orientation
  equivalence checks pass. Multi-turn DJI attitude remains authoritative.
- The runner now seals dynamic-failure evidence even if Isaac returns process
  status zero while its JSON says `passed=false`. The full CPU-only
  riser/two-wheel suite is `119 passed`; no GPU rerun has been performed.

## Next round after Round 19

- Review, commit, and push the continuous-yaw and runner repair from one clean
  riser commit.
- Keep GPU launches stopped until explicit ownership release/authorization.
- Then run case 74 only in a new namespace with unchanged deterministic
  commands and physical gates. Require wrapped proxy error, both clocks, raw
  residual diagnostics, zero learned action, and no dataset.
- Start case 77 only if the repaired case 74 passes dynamic quality. Do not
  start the accepted-71 batch before that canary sequence passes.
- Keep residual capture, Gate D, BC, and PPO blocked; do not widen the frozen
  action envelope from this failed case.
