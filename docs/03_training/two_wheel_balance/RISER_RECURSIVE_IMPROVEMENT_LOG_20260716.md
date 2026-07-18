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
- Sealed `CPU_FAILURE_TIMELINE_AUDIT.json` at SHA-256
  `d4d85cd7f2b18a376a40e20d41bc92194536688310ba7e2313818f8bc45d424e`.
  Crossing `180 deg` at `17 s` was healthy. The first false branch error appears
  only when target yaw exceeds `360 deg` at `58 s`: raw error `719.580 deg`,
  proxy effort `10 Nm`, but position error still `0.128 m` and pitch only
  `0.275 deg`. Base XY fails at `59 s`, progress reaches zero at `61 s`, camera
  position fails at `64 s`, and forbidden contact follows at `68.29 s`.
- At the first fault, nearest-equivalent target is `-354.278 deg` relative to
  reported `-353.857 deg`, leaving a wrapped error of only `-0.420 deg` while
  preserving orientation. This establishes causal ordering but not a dynamic
  cure; only a corrected case-74 rerun can provide that proof.
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

## Round 20: reverse-recovery event ordering audit

- Independent verification fixed the final case-74 status identity at SHA-256
  `b6bbd2dc25783ddff8364bafea1a23b06555d7f2dfe089095dfad29304cde4ee`.
  Riser GPU work remained stopped.
- Added a hash-bound CPU audit that separates planned reverse motion from the
  post-fault recovery response. Its output
  `CPU_REVERSE_RECOVERY_AUDIT.json` has SHA-256
  `0a8e611a567640451103a80b05de955601718f5c63aa19d0e21fe9ab475df60a`.
- Before the first false yaw-branch error, `54` sampled reverse-motion points
  ran without command saturation. Maximum absolute `vx` reference was
  `0.233591 m/s`; maximum base and camera position errors were
  `0.144051/0.182615 m`.
- After the yaw fault, `7/11` sampled commands saturated at `0.4 m/s`, the
  command changed direction, and base error reached `1.361215 m`. This is
  downstream recovery evidence, not proof that reverse tracking caused the
  original failure.
- Decision: do not alter reverse gains, add clipping/hysteresis, or relax any
  gate from this 1 Hz trace. Preserve the continuous-yaw repair and require a
  corrected case-74 dynamic canary before opening a recovery-controller change.
- The CPU-only riser/two-wheel suite passes `147` tests. No Isaac, residual
  capture, BC, PPO, dataset, or learned action was started.

## Next round after Round 20

- Keep all riser GPU work stopped until explicit authorization.
- When authorized, run corrected case 74 alone in a fresh guarded namespace.
- Only if physical dynamic quality passes may case 77 start; the accepted-71
  batch and all learning stages remain closed until the canary sequence passes.

## Round 21: physical riser sizing and endpoint safety contract

- Re-audited the explicit `0.60--1.80 m` camera-height range against the
  `1.20 m` modeled riser stroke and the previously required end stopping zones.
  A `1.20 m` mechanical stroke cannot provide both the full software range and
  independent full-speed stopping reserve.
- Added pure stopping-envelope helpers. With `5 m/s^2` controlled emergency
  deceleration and `20 ms` total response delay, stopping from `1 m/s` requires
  `0.12 m`. Including a `0.03 m` hard margin at each end yields a recommended
  `1.50 m` mechanical stroke around the unchanged `1.20 m` software stroke.
- Camera working height remains strictly `0.60--1.80 m`; mechanical overtravel
  is safety space and does not authorize `1.9 m` or a higher target.
- Recomputed the 48 V/400 W servo sizing for `2:1` and `3:1` reductions. At the
  conservative 8 kg moving load, `2:1` requires `1.4917 N m` and fails the
  motor's `1.27 N m` rated torque. `3:1` requires `0.9945 N m` at `2571 rpm`
  and passes rated torque, speed, and power, so `3:1` is now preferred.
- The worst-direction emergency stop at 8 kg and `5 m/s^2` requires about
  `1.20 N m` with `3:1`, leaving only about 5.5% rated-torque margin. The
  calculation passes but keeps the 400 W choice provisional until bench tests.
- Sealed the deterministic report at
  `20260717_hardware_envelope_v1/summary.json`, SHA-256
  `ab9373780dad90a120aaddecb72a20d8c8419cff8bf8c7b80ad4d4f2191c9afa`.
  It is explicitly `valid_for_procurement=false` pending measured mass, duty,
  inertia, offset load, regeneration, temperature, and safety validation.
- This round changed no Isaac plant, trajectory plan, LQR gain, residual scale,
  or learned policy. The expanded CPU-only riser/two-wheel suite is
  `154 passed`; no GPU work, residual capture, BC, or PPO started.

## Next round after Round 21

- Obtain the measured moving assembly mass, camera offset moment, duty cycle,
  available mechanical stroke, and 48 V bus/regeneration limits.
- Bench-validate the `3:1` 400 W candidate with brake, independent height
  sensing, hard limits, and fail-safe anti-fall hardware before procurement.
- If mechanical travel remains `1.20 m`, integrate and validate the
  direction-aware endpoint velocity governor before permitting `1 m/s` near
  either end.
- Keep the corrected case-74 GPU canary sequence unchanged and separately
  blocked until explicit authorization.

## Round 22: portfolio-wide stateful continuous-yaw preflight

- Preserved the sealed v1 continuous-yaw scope audit byte-for-byte at summary
  SHA-256 `7e357237237cb459a5b5c47f630852b4a179b955e022f4d55835c4949413fcbe`
  and CSV SHA-256
  `67bbfe164f6376189843517e9195437fb90b51e0186e8d1dfa32c4db15fc55cc`.
- Strengthened the audit in a fresh v2 namespace by replaying each accepted
  semantic-yaw sequence statefully from five equivalent physics branches:
  `-2,-1,0,+1,+2` turns. The audit now verifies mapped delta continuity and
  orientation equivalence, not only pointwise canonical equivalence.
- All `71/71` accepted plans pass. The known `45/71` branch-crossing plans
  remain covered, and the maximum naive mismatch remains `720 deg`.
- Cases 74/75/76 have the tightest valid source step, `178.000692269 deg`, but
  retain a positive `1.999307731 deg` stateful margin below the half-turn
  ambiguity boundary in every branch-reference trial.
- The fresh v2 summary SHA-256 is
  `458e3f819c140d92d5664fa9118d0b1699b06ab55ac742ac3ee3ea7f48f2c2c6`;
  CSV SHA-256 is
  `88912658ecb77b1df462c65def2271d4ad566ad9fb1baac03b1fee3cae2d5786`.
- This round changes no runtime command, source anchor, threshold, LQR gain,
  residual scale, or learned policy. It creates no labels or dataset and starts
  no Isaac, residual capture, BC, or PPO.

## Next round after Round 22

- Keep case 74 as the only next dynamic canary and require explicit GPU
  authorization plus exclusive ownership before launch.
- If corrected case 74 physically passes, admit case 77 next. Otherwise stop
  at its first unchanged physical gate and diagnose structurally.
- Keep the accepted-71 batch and all learning stages closed until both canaries
  satisfy the unchanged Gate C contract.

## Round 23: corrected-yaw canary and motion-direction recovery

- With explicit authorization and an empty GPU ownership check, ran case 74
  alone from pushed commit `f07669c` in the fresh guarded namespace
  `20260717_gate_c_case74_continuous_yaw_fix_v4_exclusive`. Case 77 was not
  requested and no other playback owner overlapped the run.
- The continuous-yaw correction removed the original physical precursor:
  no forbidden contact occurred, proxy servo p95/max fell to
  `0.114312/0.500490 deg`, proxy saturation fell to `0.000044197`, pitch max
  was `7.645328 deg`, and attitude max was `0.458524 deg`.
- Case 74 still fails Gate C. The fixed `377.1 s` horizon ended at phase
  `159.901892/188.546638 s`; position p95/max was
  `1.065665/1.094696 m`. The run did not terminate, but it did not complete the
  reference. No labels were applied and no dataset or training was created.
- Sealed hashes are admission
  `37371f47cac9d4056877c4c468fa60598f8d263300c86da00ea7f8d496b27040`,
  case JSON
  `f5686d491cc3dff069a58f54fb974e718057993261a1924976e907b737fff65d`,
  log `8f02ad60c1d81fad32fa3a70274f1d8c00ce736950cab2df8d9631d93fa8f2cb`,
  and summary
  `3deb477ab7ee45cca9aafaac801b05ce4935523460d32f3cfd9e6b94cb37535f`.
- Added a hash-bound CPU recovery audit. Its summary SHA-256 is
  `39306409b323a8c6849d7d8de84e92641e20044e14a6c8b903330e3de5e524f5`.
  It finds feedforward/command direction conflict in `206/378` trace samples
  and `155/176` samples outside the `0.25 m` position gate.
- At peak error, feedforward was `+0.004799 m/s`, velocity command was
  `-0.054150 m/s`, cross-track error was `0.976531 m`, and yaw error was
  `-1.152521 rad`. Feedforward-sign steering commanded only
  `+0.107292 rad/s`; the bounded motion-direction candidate requests
  `-0.4 rad/s`.
- Implemented `riser_motion_direction_v3`: cross-track direction now follows
  commanded velocity with a smooth `0.05 m/s` zero-speed blend. Added direct
  trace diagnostics and regression tests. This candidate is CPU-only and not
  yet dynamically validated.
- The full CPU-only riser/two-wheel suite passes `157` tests. No threshold,
  plan, source anchor, LQR gain, residual scale, label, dataset, BC, or PPO was
  changed or started.

## Next round after Round 23

- Review, commit, and push the motion-direction candidate as one scoped change.
- Keep case 77 and all learning closed.
- Only after separate explicit GPU authorization, run case 74 alone in a fresh
  guarded namespace. Stop on the first unchanged dynamic gate and use the new
  direction diagnostics to accept or reject the candidate structurally.

## Round 24: recovery-gated steering and healthy-trace compatibility

- Counterfactual replay showed the broad `riser_motion_direction_v3` rule was
  not suitable for dynamic promotion. It altered `62/104` sampled case-1
  commands and also perturbed case 52 despite both sealed runs being healthy.
  The broad candidate is retained only as diagnostic history.
- Replaced it with `riser_recovery_direction_v4`. Legacy feedforward-direction
  steering is unchanged below `0.20 m` base-position error; recovery authority
  blends from `0.20--0.40 m` and is complete at `0.40 m`. Existing command
  limits and the `0.05 m/s` feedback-direction blend remain unchanged.
- Sealed the case-74 CPU recovery audit at
  `20260717_gate_c_case74_motion_direction_recovery_gate_v2/summary.json`,
  SHA-256
  `bb98816f1fad26e7c404080e5c5d00fe00f0be1fa253e8e6255ace53e8441935`.
  The gated candidate preserves the full bounded `-0.4 rad/s` correction at
  the peak recovery state.
- Added a hash-bound compatibility audit over the sealed passing case-1 and
  case-52 traces. Both remain entirely below recovery activation, with exactly
  zero candidate command delta. Summary SHA-256 is
  `74597d37f60c14a5b99ebffde5d4036fed05a1f1641853489e0b1cc98374286b`.
- Runtime traces now expose base-position error, feedforward direction,
  feedback motion direction, recovery blend, and final motion direction. The
  full CPU-only repository suite passes `234` tests.
- No Isaac/GPU work, source plan, threshold, LQR gain, residual scale, label,
  dataset, BC, or PPO was changed or started.

## Next round after Round 24

- Commit and push the recovery-gated controller, audits, tests, and durable
  evidence references as one scoped CPU-only change.
- Keep case 77, the accepted-71 batch, residual capture, BC, and PPO closed.
- Only after separate explicit GPU authorization and an empty ownership guard,
  run case 74 alone in a fresh namespace. Preserve all physical gates and stop
  on the first failure; dynamic success must be proven before case 77 starts.

## Round 25: case-74-only launch contract

- Added `run_riser_case74_recovery_canary.sh` as the only intended next GPU
  entrypoint. It exits before the shared runner unless
  `RISER_CASE74_GPU_AUTHORIZATION=AUTHORIZED_CASE74_RECOVERY_V4` is present.
- The wrapper hardcodes case `74`; it cannot advance to case 77. Its default
  namespace is
  `20260717_gate_c_case74_recovery_direction_v4_exclusive`, and the existing
  runner still rejects an existing namespace, dirty/unpushed code, a bad
  portfolio hash, or another playback owner.
- Strengthened runtime admission to require `structural_robust_v1`,
  `riser_recovery_direction_v4`, and recovery range `[0.20,0.40] m`.
  Summaries now report physical dynamic quality and runtime-contract admission
  independently, so evidence from the wrong controller cannot be sealed as a
  Gate C pass.
- This round is CPU-only preparation. It creates no Gate C namespace, starts no
  Isaac/GPU process, and creates no label, dataset, BC, or PPO run.
- The complete CPU-only repository suite passes `236` tests.

## Next round after Round 25

- Commit and push the launch contract after the CPU suite passes.
- Wait for a separate explicit case-74 GPU authorization.
- On authorization, verify empty ownership and run only the new wrapper.
- Admit case 77 only after case 74 passes both physical dynamic quality and the
  runtime contract. Keep the accepted-71 batch and all learning closed.

## Round 26: belt-drive and Isaac plant parity boundary

- Added a hash-bound CPU audit that connects the URDF, Isaac actuator config,
  riser build audit, and provisional motor-sizing envelope.
- The URDF and Isaac limits agree at `300 N / 1.0 m/s`. The selected 400 W,
  1.27 N m motor with `3:1` reduction and 70 mm/rev lead provides
  `292.397 N` continuous rated equivalent, `877.191 N` peak equivalent, and
  `1.1667 m/s` at rated motor speed.
- Therefore the current 300 N plant cap is a transient cap, not a continuous
  hardware rating. It is only 2.6% above the rated-force equivalent and below
  peak capability. The 8 kg emergency design force remains covered, but with
  only a `1.0557x` continuous-force margin.
- Counterbalance is not required for the present conservative sizing checks.
  If a 50--70 N constant-force spring is installed, its bidirectional force,
  friction, and fail-safe behavior must be added to the Isaac plant before
  hardware-transfer training.
- Sealed the CPU report at
  `20260717_hardware_sim_parity_v1/summary.json`, SHA-256
  `5d4247fc63e2875e1b2602753e58bdfbf6705187aff2e9674c0a307e637e7190`.
  Concept screening passes, but procurement, hardware transfer, residual
  capture, BC, and PPO remain false.
- No Isaac/GPU process or Gate C namespace was started.
- The complete CPU-only repository suite passes `239` tests.

## Next round after Round 26

- Keep the case-74-only canary ready behind its explicit authorization token.
- Before hardware transfer, measure carriage mass/friction and output force,
  then add a continuous-current/thermal governor and the final counterbalance
  model.
- Do not reinterpret the 300 N transient simulation cap as a continuous motor
  guarantee or relax any trajectory/safety gate.

## Round 27: trajectory-conditioned residual policy contract

- Audited the pre-training residual network contract before any valid capture.
  The original 26-dimensional instantaneous observation was reactive and could
  not distinguish trajectories with the same current state but different
  upcoming base turns, riser motion, or camera attitude.
- Replaced it with a versioned 65-dimensional observation contract: the
  original 26 executed-state features plus three 13-dimensional reference
  lookaheads at `0.25`, `0.50`, and `1.00 s` on the retimed execution clock.
  Endpoint queries clamp to the immutable final plan anchor.
- Each lookahead carries body-frame future base error, body-frame physical
  `cam_link` position error, physical-camera attitude error, riser target
  error, and progress-scaled base/riser feed-forward. Semantic DFR attitude is
  converted with `R_world_cam = R_world_DFR * Rz(+pi/2)` before comparison.
- Preserved the 3-dimensional action contract exactly: bounded residual
  `delta-vx`, `delta-wz`, and riser-target increment. Wheel torque remains under
  frozen LQR; DJI attitude adaptation remains deterministic; no arm or physical
  gimbal-joint action was added.
- Versioned per-case data, merged data, checkpoints, artifact namespaces, and
  downstream admissions to v2. Holdout and all-79 gates now independently
  reject checkpoints that do not declare the v2 policy schema, observation
  contract, and exact lookahead horizons.
- Historical v1 NPZs and checkpoints remain quarantined; they are neither
  upgraded nor silently mixed into v2. No Isaac/GPU run, residual capture,
  BC, PPO, learned rollout, source plan, threshold, LQR gain, or residual scale
  was changed or started in this round.
- The complete CPU-only repository suite passes `240` tests.

## Next round after Round 27

- Keep the case-74-only deterministic canary separate and authorization-gated;
  this DNN contract change does not make its physical failure pass.
- Do not open residual capture until the deterministic dynamic admission is
  satisfied under the unchanged safety and quality gates.
- When capture is eventually authorized, start only in the empty lookahead-v2
  namespace and prove one bounded per-case artifact before any all-79 batch or
  BC run.

## Round 28: shared-horizon residual network architecture

- Audited the actual v2 policy network before capture. The previous flat
  `65-256-256-128-3` MLP was bounded and deployable, but assigned unrelated
  first-layer weights to the same 13 physical features at the three lookahead
  horizons. That is weak structure for a corpus of only 79 trajectories.
- Replaced it with `state_shared_lookahead_fusion_v1`: a `128,128` encoder for
  the 26 executed-state features, one weight-shared `64,64` encoder applied to
  each ordered 13-dimensional lookahead, and a `256,128` fusion encoder before
  the unchanged three-channel tanh action head.
- Preserved train-only observation normalization, orthogonal initialization,
  LayerNorm, SiLU, deterministic training, case-balanced loss, TorchScript
  export, the 65-dimensional observation contract, and the bounded 3D action
  contract.
- Versioned the BC report to v2 and added fail-closed architecture admission
  to BC, holdout, and all-79 scripts. This also corrects a pre-training schema
  mismatch where downstream gates had compared the BC report schema against
  the checkpoint schema; no historical policy had been admitted through that
  impossible check.
- The sealed case-74 trace was re-read without changing it. Its main reverse
  recovery incident is localized to approximately `89--166 s` wall time and
  `52.1--59.8 s` plan time. Recovery-v4 remains the only pending controller
  candidate; no second controller tweak was introduced before its canary.
- No Isaac/GPU run, residual capture, BC, PPO, learned rollout, source plan,
  physical threshold, LQR gain, or residual action scale was changed or
  started in this round.
- The complete CPU-only repository suite passes `241` tests, including eager
  versus TorchScript parity and repeated-seed BC prediction equality.

## Next round after Round 28

- Run the complete CPU suite and publish this architecture as one scoped
  pre-capture change.
- Keep case 74 as the only next deterministic GPU canary behind its explicit
  authorization token; do not infer authorization from this CPU work.
- Only after deterministic admission may the empty lookahead-v2 capture
  namespace, BC, holdout, and learned all-79 sequence open in order.

## Round 29: riser motor thermal-force admission

- Closed a known hardware-transfer gap without changing the pending controller
  candidate. The Isaac riser drive retains its existing 300 N transient cap,
  but dynamic evaluation now integrates the measured applied force through
  `leadshine_400w_first_order_monitor_v1`.
- The provisional model uses the selected drive's `292.397 N` continuous and
  `877.191 N` peak linear-force equivalents. It integrates normalized
  force-squared load with a documented provisional `30 s` first-order thermal
  time constant at every 200 Hz physics step.
- Dynamic admission now fails closed unless applied force is observed for every
  completed step, maximum thermal load remains at or below `1.0`, and no peak
  force violation occurs. Runtime JSON exposes the force/thermal maxima, final
  load, sample count, and model parameters.
- Gate C sealing independently requires the thermal-force contract and all
  three thermal checks. A dynamically unsafe run cannot become a dataset even
  if camera tracking metrics pass.
- This is a qualification monitor, not an active force clamp. It prevents
  training admission from relying on sustained over-continuous force while
  preserving recovery-v4 commands for an unconfounded canary. Bench-identified
  motor temperature/current constants and an active hardware current derater
  remain required before hardware transfer.
- Sealed the CPU parity report at
  `20260717_hardware_sim_parity_v2/summary.json`, SHA-256
  `ed2d332fd8caef27368f57255db9c62af2aeb4b53670f76c0a32968115675f8f`.
  It proves drive/monitor parameter parity and runner gate wiring while
  explicitly keeping active derating, hardware transfer, and training false.
- No Isaac/GPU run, residual capture, BC, PPO, learned rollout, source plan,
  tracking threshold, LQR gain, or residual action scale was changed or
  started in this round.
- The complete CPU-only repository suite passes `246` tests.

## Next round after Round 29

- Run the complete CPU suite and publish the thermal admission as a scoped
  hardware-realism change.
- Keep case 74 as the only next GPU canary behind its explicit authorization;
  its result must now pass both the unchanged physical tracking gates and the
  new thermal-force admission.
- Replace the provisional 30 s model only from measured bench evidence, never
  by tuning it to make a trajectory pass.

## Round 30: smoothed-plan tranches and bounded camera lever-arm candidate

- Replaced the obsolete long-duration exact-anchor execution portfolio with
  the explicit smoothed-plan contract. The current sealed CPU portfolio admits
  `70/79` cases while preserving immutable source arrays, separate source and
  execution clocks, endpoint/order provenance, path/deviation bounds, and the
  `2.0x` duration ceiling.
- Deterministic Gate C qualification has passed cases
  `[10, 11, 12, 19, 23, 24, 25, 26, 28, 52, 53, 66, 70, 74, 77]` under the
  current smoothed-plan lineage. This is `15/70`, not corpus completion.
- The third tranche stopped correctly at case 68. Case 68 completed its full
  `13.562891 s` execution clock with no termination, saturation, thermal,
  attitude, riser, proxy, or balance failure. Its only physical rejection was
  camera-position p95 `0.172847 m` against the unchanged `0.15 m` limit;
  maximum error `0.175031 m` remained below `0.25 m`.
- Comparison with the dynamically passing case 66 localized the gap. Both
  cases use effectively the same base path and timing, while case 68 carries
  the camera higher. The existing outer loop tracks planned chassis XY/yaw and
  uses camera error only to slow phase progress; it does not cancel the
  camera-to-base XY offset caused by chassis pitch at the taller lever arm.
- Added an opt-in `measured_camera_to_base_xy_offset_v1` controller candidate.
  It compares planned and measured camera-to-base XY lever vectors and offsets
  the commanded chassis XY target in the opposite direction. The correction
  is norm-bounded to `0.05 m`; source targets, planned chassis yaw, execution
  clocks, LQR gains, physical gates, and residual scales remain unchanged.
- Added policy-rate aggregate and 1 Hz trace telemetry for raw/bounded
  correction, saturation, commanded-versus-planned base targets, and fail-fast
  CLI validation before `AppLauncher`. The candidate is disabled unless an
  explicit runtime flag enables it.
- A read-only calculation over existing 1 Hz traces estimates case 68 camera
  p95 changing from `0.172907 m` to `0.124078 m` under ideal translation, and
  case 66 from `0.139314 m` to `0.093109 m`. This is geometric canary
  justification only; it is not a dynamic result because it does not replay
  the altered closed loop.
- No Isaac/GPU process, new namespace, source plan, label dataset, residual
  capture, BC, PPO, or other training was started in this CPU-only round.

## Next round after Round 30

- Run the authoritative CPU suite and publish the opt-in controller candidate
  as a scoped change only if the diff proves no source-plan, threshold, LQR,
  or residual-scale mutation.
- Prepare a fresh hash-bound, ownership-guarded wrapper for case 68 followed by
  the previously passing case 66. Stop at case 68 if any physical, thermal, or
  runtime-contract check fails.
- Treat case 68 as recovered only from sealed dynamic evidence. Case 66 must
  independently prove non-regression before this candidate can be considered
  for the remaining corpus.
- Keep residual capture, BC, PPO, and all other training closed.

## Round 31: camera lever-arm recovery and healthy-case regression

- Published the opt-in controller and evidence contract at commit
  `c03305a00a589171663072458cbdcb995d8b6252`. The final authoritative CPU
  suite passed `320` tests with two existing pytest-configuration warnings.
- Ran only cases `[68, 66]` in the fresh exclusive namespace
  `20260718_gate_c_smoothed_case68_66_camera_lever_arm_v1_exclusive`.
  Case 68 had to pass before the wrapper could start the previously healthy
  case 66 regression. Both source plans, clocks, LQR gains, thresholds, and
  the 28 kg robot USD remained unchanged.
- Case 68 passed every physical, thermal, controller-evidence, and runtime
  check. Camera-position p95/max improved from `0.172847/0.175031 m` to
  `0.123103/0.125201 m`; pitch max was `6.705359 deg`; action saturation was
  zero; and no termination occurred. Its residual-label envelope also passed
  with normalized absolute maxima `[0.847078, 0.189232, 0.128190]`.
- Case 66 independently passed the non-regression gate. Camera-position
  p95/max improved from `0.139627/0.141763 m` to `0.091068/0.121144 m`;
  pitch max was `6.614879 deg`; action saturation was zero; and no termination
  occurred. Its residual-label envelope passed with normalized absolute
  maxima `[0.834357, 0.187343, 0.129256]`.
- The bounded correction reached `0.05 m` and was saturated for `98.2407%` of
  case 68 and `97.7772%` of case 66. This is safe under the candidate contract
  and improved both cases, but it is strong evidence that expansion must stay
  bounded rather than treating two passes as universal portfolio proof.
- Evidence hashes:
  - admission JSON: `eda824d9a4f27898c4cc68bbb1c5e24c8fdbbf065629cbf25408405a9bc3f804`;
  - case 68 JSON: `d49443e8cf3111f07661802738ca0abda1eaafc8f900a480226b8c723e6d4b56`;
  - case 66 JSON: `3adafeaae66c0f5941ddfe79a3d443e5cb2c61c63ac1e7d9510979fa61f00a77`;
  - final summary: `953b864ad282490aafcdd7ad4b76e10b8ccebd435b767585705531b331613026`.
- The final summary records dynamic, thermal, controller-evidence, runtime,
  and residual-envelope outcomes as independently true. It also records no
  residual dataset, BC, PPO, or valid-for-training admission. GPU ownership
  was empty after closure.

## Next round after Round 31

- Keep the bounded candidate controller profile and unchanged `0.05 m` bound
  fixed.
- Run only the previously unstarted tranche cases `[67, 7]` in a fresh,
  hash-bound, ownership-guarded namespace and stop on the first rejection.
- If both pass, continue the remaining admitted corpus in small fail-fast
  tranches. Do not globally infer success from cases 68 and 66.
- Recompute the raw residual envelope only after deterministic qualification;
  keep capture, BC, PPO, and all training closed meanwhile.

## Round 32: tranche-tail expansion and case-7 dynamic-margin diagnosis

- Extended the same hash-bound camera-lever-arm wrapper to cases `[67, 7]` at
  commit `8f36bcb7880033fe434be33675dd925be2e87fcf`. The fresh exclusive namespace
  was `20260718_gate_c_smoothed_case67_7_camera_lever_arm_v1_exclusive`.
- Case 67 passed every dynamic, thermal, controller-evidence, runtime, and
  residual-label-envelope check. Camera-position p95/max were
  `0.114029/0.115406 m`; pitch max was `6.658292 deg`; action saturation was
  zero; no termination occurred. This raises the uniquely qualified set to
  `17/70` admitted plans.
- Case 7 completed all `6093` physics steps and the full
  `13.582122/12.940941 s` execution/source clocks. It stopped the wrapper only
  on `position_p95_bounded`: `0.150670894 m` against the unchanged `0.15 m`
  gate, an excess of `0.000670894 m`. Position max `0.168787626 m`, pitch,
  attitude, thermal force, servo, rate, saturation, and termination gates all
  passed.
- The 1 Hz trace localizes the case-7 error to the first three seconds. The
  parent plan has execution/source ratio `1.049547` and rapidly reaches the
  frozen `-0.4 m/s` base feed-forward limit. At approximately `2 s`, camera
  error peaks near `0.169 m` while the base lags about `0.215 m`; the remainder
  of the trace returns below approximately `0.131 m`. This is a startup dynamic
  margin problem, not source geometry, camera-height geometry, balance, or a
  failed lever-arm correction.
- Evidence hashes:
  - admission JSON: `bc692660e13955cbf5ed03413cc5eeb1e9122b5d9bed7404002c3d6364913b7b`;
  - case 67 JSON: `df2ad0e85ecdd04395b2f1c520eedb3b1f6246d72cb3e645fe89ec1c5e190066`;
  - case 7 JSON: `32a9bec2c4d375907888e6f01fd627f17d272eb3841b60bf318e3c7c43c0f31d`;
  - final summary: `435d8ad4b4eb955a824219cd9771bc680fb0734d77f31e90c86199c825b54045`.
- No residual dataset was written; residual commands remained zero; capture,
  BC, PPO, and training remained false. GPU ownership was empty after closure.

## Next round after Round 32

- Derive one separately labeled, CPU-only uniform execution retime for case 7.
  Preserve every source anchor, source timestamp, target state, endpoint,
  ordering, and all physical gates; alter only the execution clock and derived
  feed-forward rates.
- Use target execution/source ratio `1.4`. This bounds peak feed-forward near
  `0.30 m/s` (`1.049547 * 0.4 / 1.4`) while retaining margin below the frozen
  `2.0x` duration ceiling and the `1.5` portfolio-median ceiling.
- Run the duration, path, transition, kinematic, hash, and array-identity gates
  before preparing any runtime wrapper. Do not launch Isaac from this step.
- Keep capture, BC, PPO, and all training closed. A CPU pass authorizes only a
  reviewable case-7 canary candidate, not dynamic admission.

## Round 33: case-7 source-preserving dynamic-margin retime

- Extended the CPU derivation contract with an explicit
  `completed_position_p95_only` mode. It requires the sealed case to have
  completed both clocks, failed only the p95 gate, passed thermal/runtime/
  controller evidence, terminated normally, applied no residual action, and
  kept all learning stages closed. The prior completion-only mode remains the
  default.
- Derived case 7 at execution/source ratio `1.4` from the unchanged v7 plan.
  The execution clock changed from `13.582122` to `18.117317 s`; all `663`
  source poses, authoritative source timestamps, target states, base/riser/
  proxy states, endpoints, ordering, and initialization arrays are byte-equal
  to the parent. Only the execution clock and its derived feed-forward rates
  changed.
- Static maximum rates dropped from approximately `0.4` to `0.299870 m/s`
  base linear, `0.299870 rad/s` base yaw, `0.685982 m/s` riser, and
  `0.314024 rad/s` proxy. Every duration, path, transition, kinematic, camera
  height, and training-closure check passed.
- Composed the fresh all-79 v8 portfolio with exactly this one replacement.
  It retains `70/79` admitted plans and accepted duration median
  `1.499794`, below the unchanged `1.5` ceiling. The case-7 static Gate C
  admission passes, but `valid_for_training` remains false.
- CPU artifact hashes:
  - derived plan: `a83934dab6e4293cd830397d3c2ffb41d4f4d78545dddec7fdfa630fa0d22f41`;
  - derivation manifest: `f2557bc31afd0f15aa5f57bb340d1e6d0bad81b0514a264f408da28664b0db2d`;
  - v8 portfolio manifest: `0a6a9361095e3045b2835f2ea96520f2b6e1c378df4feaa394fb87627bc165b2`;
  - v8 portfolio summary: `ad4f509941653faa70778d70ee4684e30f383e30b2ba056976600890534565ce`;
  - case-7 static admission: `86a0d66d41d39f809463aefdac51a90850de46c9dc78a9c77b8ef084cb349c93`.
- The linked-worktree portfolio composer was repaired to resolve its committed
  Git directory correctly under Windows Python. The authoritative suite at
  the preceding derivation commit passed `323` tests with two existing pytest
  configuration warnings.
- No Isaac/GPU process, label dataset, residual capture, BC, PPO, or training
  was started in this round.

## Next round after Round 33

- Run one hash-bound case-7-only canary from the v8 portfolio in a fresh
  namespace, with the same LQR, camera lever-arm compensation, thermal model,
  physical thresholds, and ownership guard.
- A dynamic pass adds case 7 to the qualified set; a failure stops immediately
  and must be classified from the sealed evidence. Do not tune during the run.
- Keep residual capture, BC, PPO, and all training closed regardless of the
  canary outcome.

## Round 34: case-7 retimed dynamic qualification

- Published the one-case guarded runtime contract at commit
  `7f9ce0aa949ce5da48616de59d8d337d22cea23b` and ran only case 7 in
  `20260718_gate_c_smoothed_case7_dynamic_retime_v1_exclusive`. The wrapper
  pinned the v8 portfolio/plan, source package, LQR gains, 28 kg USD, playback,
  controller, loader, validator, summarizer, and wrapper identities.
- The retimed case completed all `6607` physics steps and both clocks:
  `12.940941 s` authoritative source time and `18.117317 s` execution time.
  Camera-position p95/max passed at `0.130904/0.142948 m`, improving from the
  parent run's `0.150671/0.168788 m` without changing the `0.15/0.25 m` gates.
- Attitude p95/max passed at `0.151167/0.260295 deg`; action saturation was
  zero; there was no termination. Riser effort max was `17.185259 N` and
  thermal load max `0.001086`, with every thermal and controller-evidence
  check true.
- The bounded camera lever-arm correction remained unchanged at `0.05 m` and
  saturated for `91.9479%` of policy-rate samples. The raw residual was only
  audited prospectively: normalized maxima
  `[0.766613, 0.403088, 0.126284]` passed the frozen envelope, but residual
  action remained `[0,0,0]`, was not applied to commands, and no dataset was
  written.
- The canary summary independently records dynamic, thermal, controller,
  runtime, and residual-envelope outcomes as true while capture, BC, PPO, and
  training remain false. This raises the uniquely dynamically qualified set
  to `18/70` CPU-admitted plans.
- Evidence hashes:
  - runtime admission: `2ec64cef491fab5a86ceef2769fc93e5046035f6976997e6cc893814cb2feabb`;
  - case-7 gate JSON: `28b1117eb1dede740e70348f48eb06aa35d87ce86ae3572a740737a83c75605f`;
  - case-7 log: `18e7b9290205667214c600349103dd22dac66da9560eb6080a033bc17d8325a1`;
  - final summary: `6765e51ddb964e6e92ad7b32de0fdadc6f90dad5d9767788b1da3049a62b65d0`.
- The wrapper exited `0`; GPU and playback ownership were empty after closure.
  A transient SSH monitoring interruption did not affect the process or the
  sealed result.

## Next round after Round 34

- Keep the v8 portfolio and the accepted case-7 retime immutable.
- Continue the remaining CPU-admitted corpus in small, ordered, fail-fast
  deterministic tranches under the same controller and gates. Do not infer
  corpus qualification from `18/70` passes.
- On the first rejection, stop and classify it before proposing any bounded
  plan/controller change. Keep residual capture, BC, PPO, and training closed.

## Round 35: v8 corpus expansion through cases 2 and 3

- Published the next ordered fail-fast tranche at commit
  `c2b896a3d8a047cef7b470d0900cf02dd02845d2` and ran only cases `[2,3]` from
  the immutable v8 portfolio. Case 3 could start only after case 2 passed.
- Case 2 completed `9199` steps and its full `18.241928/9.439314 s`
  execution/source clocks. Position p95/max passed at
  `0.139830/0.153514 m`; attitude p95/max were
  `0.137239/0.175777 deg`; action saturation was zero; no termination
  occurred. Riser effort/thermal maxima were `20.321682 N` and `0.000931`.
- Case 3 completed `8478` steps and its full `19.343648/11.873175 s`
  execution/source clocks. Position p95/max passed at
  `0.093561/0.095761 m`; attitude p95/max were
  `0.123540/0.225890 deg`; action saturation ratio was `0.000236`; no
  termination occurred. Riser effort/thermal maxima were `21.518591 N` and
  `0.001359`.
- Both cases passed dynamic, thermal, controller-evidence, runtime, and frozen
  residual-envelope checks. The camera correction remained bounded to
  `0.05 m`; residual actions remained exactly zero and were not applied; no
  dataset, capture, BC, PPO, or training stage opened.
- This raises the uniquely dynamically qualified set to `20/70` CPU-admitted
  plans. It does not qualify the remaining 50 plans or open learning.
- Evidence hashes:
  - runtime admission: `75f6de15b41e49fbd3f8eae02bc12b4b8aec87e4f7bf9aa92d2a49404ad0f582`;
  - case-2 gate/log: `a6479130ef95d0cc1ea48d723f35c858638f2ca776fc6069863019de9b086e68` /
    `2b0eda22b7f3f037c1662c03e3bff75b3eeb08200f3a9b6f3d7640b3162335b3`;
  - case-3 gate/log: `56a0cf3d61f943b9f5d92a51112c11e43f920b5fdc34fbfd9d7fad173df79ddc` /
    `9fc19f840e2d9300dcabf3e1c339668afa1edc5b095b2a60f5a560982e01fd95`;
  - final summary: `0eb9ca95f76487805d25164afdda014ea73ebc6460dc426e479e3bdbf8206a7b`.
- Both wrappers exited `0`; GPU and playback ownership were empty after the
  tranche closed.

## Next round after Round 35

- Continue in source order with cases `[4,5]` under the unchanged v8 plan,
  controller, LQR, camera correction, thermal monitor, and gates.
- Preserve the same ordered fail-fast rule. If case 4 rejects, case 5 must not
  start; classify the first failed check before any change.
- Keep residual capture, BC, PPO, and all training closed.

## Round 36: paired cases 4 and 5 dynamic qualification

- Published the `[4,5]` fail-fast tranche at commit
  `9813ec22ea8aebf44a08f8198c152bd54aa234db` and ran only those two v8
  plans. Case 5 started only after case 4 passed.
- Case 4 completed `6172` steps and its full `21.514453/14.042191 s`
  execution/source clocks. Position p95/max passed at
  `0.128736/0.151724 m`; attitude p95/max were
  `0.183879/0.217912 deg`; action saturation was zero; no termination
  occurred. Riser effort/thermal maxima were `18.323277 N` and `0.001204`.
- Case 5 completed `6169` steps and its full `21.514453/14.042191 s`
  execution/source clocks. Position p95/max passed at
  `0.128780/0.152655 m`; attitude p95/max were
  `0.183783/0.212018 deg`; action saturation was zero; no termination
  occurred. Riser effort/thermal maxima were `18.269590 N` and `0.001204`.
- Both cases passed every dynamic, thermal, controller-evidence, runtime, and
  frozen residual-envelope check. The camera correction remained bounded to
  `0.05 m`; residual action remained exactly zero and no dataset or learning
  stage opened.
- The close but independently measured results confirm the paired trajectory
  structure without treating one camera-height variant as evidence for the
  other. The uniquely dynamically qualified set is now `22/70`.
- Evidence hashes:
  - runtime admission: `1316bb80d7f14a5069d84b3cbc63c636fac8ea1e7b73dd95deb0116d11f88a66`;
  - case-4 gate/log: `b1dc4b6e970b7fba1c22769f7087fcdc8e71305511b11479ddaadb5e0b64b992` /
    `51a5d1329e5893793895345de7b6260bcc226c74a1caec0a2be2b1414f0f9cbd`;
  - case-5 gate/log: `e3b46ceb405bad9e3a6ee00d482dbc9da90e9f3f2097dfdbd0a8729e471e4ce0` /
    `d97aeb5b9f81b27fff1c397c72f299d62ed6a17348d6206963361123a7199705`;
  - final summary: `90980a0b0df103b48a8529398590479e31795dd51306caf226382df41412f582`.
- Both wrappers exited `0`; GPU and playback ownership were empty after the
  tranche closed.

## Next round after Round 36

- Continue in source order with `[6,8]`; case 7 is already separately
  qualified from its v8 retime.
- Case 8 retains the aggressive `1.049547x` execution ratio and
  `0.915 m/s` static riser rate, so treat it as an independent dynamic test,
  not as implied by case 7. Preserve fail-fast ordering and all existing
  thresholds.
- Keep residual capture, BC, PPO, and all training closed.

## Round 37: case-6 pass and case-8 p95-only rejection

- Published and ran the ordered `[6,8]` tranche at commit
  `b7917da1ba864647a252410ae06165815240aeb5`. Case 8 started only after case
  6 passed.
- Case 6 completed `7968` steps and the full `17.737275/15.942736 s`
  execution/source clocks. Position p95/max passed at
  `0.118125/0.127080 m`; attitude p95/max were
  `0.157198/0.457533 deg`; action saturation ratio was `0.003263`; no
  termination occurred. Every thermal, controller, runtime, and residual
  envelope check passed.
- Case 8 completed `6108` steps and the full `13.582122/12.940941 s` clocks.
  It failed only `position_p95_bounded`: `0.150575598 m` against the unchanged
  `0.15 m` gate, an excess of `0.000575598 m`. Position max
  `0.168787626 m`, attitude, balance, thermal, riser/proxy, saturation,
  completion, and residual-envelope checks all passed; no termination or
  residual application occurred.
- The wrapper stopped on case 8 with no later case. Case 6 raises the uniquely
  qualified set to `23/70`; case 8 remains unqualified at this checkpoint.
- Evidence hashes:
  - runtime admission: `1f5b88dd1a7462f0a49ffd2a1b2ef060eae4548f948ef02d37208aed1f42b643`;
  - case-6 gate/log: `140a080b1daab4985fe353bfee9e3f86186dfe3c319841d5a8e08f98e8d6b7c7` /
    `237b2afc7141e852bcfb08d2bda75cd98cb9445e25b71705db190a9384e522af`;
  - case-8 gate/log: `c5ea974ebebea42665112eb985b17af902b33d9fdf802a83c5d57ab33a777787` /
    `5ef232728388527800861bf4a903d7576759fd9eb47ec8bfcf84ff4205842297`;
  - final summary: `135c8f453daa311921c3c966a02f0ce20a0ee7c33608071bab57e4c2c571cdb2`.
- GPU and playback ownership were empty after the fail-fast stop. Capture, BC,
  PPO, and training remained closed.

## Round 38: case-8 source-preserving retime candidate

- Applied the same evidence-bound `completed_position_p95_only` CPU contract
  proven by case 7. The case-8 candidate preserves every source pose,
  timestamp, target/base/riser/proxy state, endpoint, ordering, and
  initialization array; only execution time and derived feed-forward rates
  change.
- The candidate execution/source ratio is `1.4`, giving execution duration
  `18.117317 s`, maximum base linear speed `0.299870 m/s`, and maximum riser
  speed `0.685982 m/s`. Every static duration, path, transition, kinematic,
  camera-height, array-identity, and learning-closure gate passes.
- Composed the fresh v9 all-79 portfolio with both prior case-7 and new case-8
  retimes preserved. It remains `70/79` CPU-admitted with accepted-duration
  median `1.499794`; its case-8 static runtime admission passes while training
  remains false.
- CPU artifact hashes:
  - case-8 retimed plan: `f07ff020128dee70ea9c8c2d806dc75c8e0ef3964dccb4e0aabfd1b0048f3655`;
  - derivation manifest/summary: `3d70721d5525ab1a4ee2ebb537c99f8de64ab4220bcb3dcce46bdbf38ad6557b` /
    `9a9161e94802abed48431ce7617f59839bb6d2d9b20e05e6dad3a172fddf35da`;
  - v9 portfolio manifest/summary: `ac5da6ce721bd0af51b9b851ada86b08f587f190440c9de23172b115bad3c748` /
    `b1c2fffd4a620c2e5c89303702d809c108bcd3554ceeaa4ccd59392c3d531285`;
  - case-8 static admission: `5c89760bd220e1684394a73e5fa63b5c3d1b1e35bcb246f656074b817ea87db1`.
- No Isaac/GPU process, dataset, capture, BC, PPO, or training was started by
  the retime/composition step.

## Next round after Round 38

- Run one v9 case-8-only dynamic canary under the unchanged LQR, camera
  correction, thermal monitor, physical gates, and ownership guard.
- Count case 8 only from sealed dynamic evidence. Regardless of outcome, do
  not open residual capture, BC, PPO, or training.

## Round 39: case-8 retimed dynamic qualification

- Published the v9 one-case runtime contract at commit
  `122c0981fde1a3f822e30843a91c7506397a2206` and ran only retimed case 8 in
  `20260718_gate_c_smoothed_case8_dynamic_retime_v1_exclusive`.
- The case completed `6605` physics steps and both clocks:
  `12.940941 s` source and `18.117317 s` execution. Position p95/max improved
  from `0.150576/0.168788 m` to `0.131254/0.143331 m`, passing the unchanged
  `0.15/0.25 m` gates.
- Attitude p95/max passed at `0.148800/0.223093 deg`; action saturation was
  zero; no termination occurred. Riser effort max was `17.041342 N` and
  thermal load max `0.001085`; every controller-evidence and runtime check
  passed.
- The `0.05 m` camera correction remained unchanged and saturated for
  `92.0061%` of policy-rate samples. The frozen prospective residual envelope
  passed with normalized maxima `[0.768307, 0.407379, 0.125314]`, but residual
  action remained exactly zero, was not applied, and no dataset was written.
- This raises the uniquely dynamically qualified set to `24/70`. Capture, BC,
  PPO, and training remain closed.
- Evidence hashes:
  - runtime admission: `fbe2ac787022309ead89846a7c9905b929ba51fd60567950427cf310f09edde7`;
  - case-8 gate JSON: `19506045f9b6ec04cee58efa1b5d2d5600824ce166b1534db05a4895596cf1e0`;
  - case-8 log: `c2979794691ebbacbd92b9192335f5ec0841bde82ce55834405c792dad211617`;
  - final summary: `b0591a9f928e8e3b052fe51a19c5f3d22936cf1f83bb7c1b27c796439b04759f`.
- The wrapper exited `0`; GPU and playback ownership were empty after closure.

## Next round after Round 39

- Continue in source order with case 9, then case 13 only if case 9 passes.
  Case 9 shares the aggressive original timing family with cases 7 and 8 but
  still requires independent dynamic evidence.
- Preserve the v9 portfolio, fail-fast ordering, LQR/controller/gates, and all
  training-closure invariants.

## Round 40: case-9 transport interruption and sealed p95-only retry

- The first `[9,13]` namespace started case 9 but its SSH-bound wrapper exited
  `255` after Isaac initialization, wrote no gate JSON, and correctly left
  case 13 unstarted. It is preserved as infrastructure-interrupted evidence,
  not a trajectory result.
- Live cleanup found no remaining Isaac/Python/GPU owner and Windows memory had
  recovered to approximately `21 GB` free. The retry used a fresh namespace
  and a `nohup`-detached wrapper reparented to init so SSH transport could not
  terminate the simulation.
- The detached case 9 completed all `6100` steps and both
  `13.582122/12.940941 s` execution/source clocks. It failed only
  `position_p95_bounded`: `0.150626389 m` versus `0.15 m`, an excess of
  `0.000626389 m`. Position max `0.168787626 m`, attitude, balance, thermal,
  riser/proxy, saturation, completion, controller evidence, runtime, and
  residual-envelope checks all passed. Case 13 remained unstarted.
- Residual actions remained zero and unapplied; no dataset, capture, BC, PPO,
  or training stage opened.
- Retry evidence hashes:
  - runtime admission: `74749e4357d33a60cdc9187272189c754715af968c2a0b0cf60bfd610c897dce`;
  - case-9 gate/log: `b46a5e689f3cca12478b69bd2fbfb6c02b17c20bd5e87607ba2dd9641ba504e1` /
    `d629ec2f4c1b9c9314b2a962763ec6ced2a4d9c3ccc68b549efea3bac50d76b6`;
  - final summary: `bc5b58a82e99306c9cbcedba1b35bbe36edeb504fe23e6ec13e5f54de96bea1a`;
  - detached outer log: `cd7578eaa48de4ab31e0fcd77336586d5cae4655b0d93cdc68200e1b4e696ebf`.
- Corrected the CPU derivation admission to bind the failing case row,
  first-reject runtime evidence, dynamically passed prefix, and unstarted
  suffix independently from aggregate summary health. The complete suite
  passes `324` tests at commit `c82e3658a7769a424a26eb22a203cd0b4ae39d52`.

## Round 41: case-9 source-preserving retime candidate

- Derived case 9 at execution/source ratio `1.4` from the sealed retry. Every
  source pose, timestamp, target/base/riser/proxy state, endpoint, ordering,
  and initialization array is unchanged; only execution time and derived
  feed-forward rates change.
- Execution duration is `18.117317 s`; maximum base/riser speeds are
  `0.299870/0.685982 m/s`. All duration, path, transition, kinematic,
  camera-height, array-identity, and training-closure checks pass.
- Composed the v10 all-79 portfolio, preserving the prior case-7 and case-8
  retimes and adding only case 9. It remains `70/79` CPU-admitted with median
  ratio `1.499794`; static case-9 runtime admission passes and training remains
  false.
- CPU artifact hashes:
  - case-9 retimed plan: `195249929b363e49fcc73a2600c2d7de9dc9d9fedf0bb9ed0718a44e76bf3fd3`;
  - derivation manifest/summary: `06652c6b74ff22beb62b840bac26f215da0a2156e044f31158614ad71bd991c7` /
    `d2a2ef8db7091cfc973720e005d13aa4616cdb8fe2c1bd51c75782a11951bf72`;
  - v10 portfolio manifest/summary: `229a76e3003b2e31a0d1a7a7cd34cda208b292638e7039e79198c951e034cda1` /
    `58a480469ef4dcfbd93b90c4f1aa27426dfdd1a75afc66ba4e4cee5f90966fbd`;
  - case-9 static admission: `1ac875218dbcaaf4c694250cc19be8938543fbb6c7a95bc09dc822a5bace7222`.
- No Isaac/GPU run or learning stage was started by this CPU retime step.

## Next round after Round 41

- Run one detached, hash-bound v10 case-9 canary under unchanged physical and
  safety gates. Count it only from sealed dynamic evidence.
- Keep case 13 unstarted until case 9 passes. Keep residual capture, BC, PPO,
  and training closed.

## Round 42: case-9 retimed dynamic qualification

- Published the detached v10 case-9 runtime contract at commit
  `ecff98034194fe9fc6d0e4362d872c591715ed7c` and ran only retimed case 9 in
  `20260718_gate_c_smoothed_case9_dynamic_retime_v1_exclusive`.
- The case completed `6608` physics steps and both clocks:
  `12.940941 s` source and `18.117317 s` execution. Position p95/max improved
  from `0.150626/0.168788 m` to `0.131352/0.143312 m`, passing the unchanged
  `0.15/0.25 m` gates.
- Attitude p95/max passed at `0.150890/0.219478 deg`; pitch max was
  `6.171498 deg`; action, proxy, and riser saturation were all zero. No
  termination occurred, internal attitude IK had zero failures, and all
  controller-evidence, runtime-contract, and thermal-admission checks passed.
- Riser effort max was `17.154804 N` and thermal load max `0.001085`. The
  `0.05 m` camera correction remained unchanged and saturated for `91.8130%`
  of policy-rate samples.
- The frozen prospective residual envelope passed with normalized maxima
  `[0.768161, 0.403818, 0.125279]`. Residual action remained exactly zero,
  was not applied to the deterministic commands, and no dataset was written.
- This raises the uniquely dynamically qualified set to `25/70`. Residual
  capture, BC, PPO, and training remain closed.
- Evidence hashes:
  - runtime admission: `e495ea1d80ac4293eda113a26ac0b2f1f2d2c3e1c14fa54331aee3b75f551c20`;
  - case-9 gate JSON: `3881e31b4b08ac1be6a8ac51aeb80b465ba44558080bf247851ce7da34d46461`;
  - case-9 log: `e13264326f35066a1bd9e1bedb9a9f77f84685742566d4815218e59604d03284`;
  - final summary: `aa4adcaf7fc27d48b5fae574655dc425eaddd2332705f6a0d9dc67071657d089`.
- The wrapper closed fail-safe with the complete summary marked passed; GPU
  and playback ownership were empty after closure.

## Next round after Round 42

- Continue in source order with one fresh, hash-bound v10 case-13-only dynamic
  canary. Use a detached wrapper and the same exclusive ownership guard.
- Do not alter the LQR, deterministic controller, source anchors, v10 plans,
  physical gates, camera correction, thermal model, or residual scales.
- Count case 13 only from sealed dynamic evidence. Keep residual capture, BC,
  PPO, and training closed regardless of outcome.

## Round 43: case-13 fixed wall-time bound exhaustion

- Published the v10 case-13-only runtime contract at commit
  `e566c137bf8c6c38bb9cf2716d37e25ecddb1b76` and launched it detached in
  `20260718_gate_c_smoothed_case13_v10_camera_lever_arm_v1_exclusive`.
- Static admission passed and bound only case 13 plan
  `0451bc312420b1d1a026afb89c23ddb0b325a8b9da10246918e42a067494a228`.
  The plan preserves `1713` source anchors with `33.821283 s` source and
  `40.269133 s` execution clocks.
- The generic `480 s` shell wall timeout expired with exit code `124` after
  Isaac initialization and dynamic execution had started. No gate JSON was
  written, so this is an infrastructure-bound `missing_runtime_json` result,
  not a trajectory-quality rejection or pass.
- The wrapper wrote a fail-closed summary, started no later case, and left no
  playback or GPU owner. No residual action, dataset, capture, BC, PPO, or
  training stage was admitted.
- Evidence hashes:
  - runtime admission: `71dc7d747f76329fd4292bf32db3717a0fc9c38cbd657f135701b0a12433cbcc`;
  - case-13 log: `0756363039ed0d9e6f15f5310fd9ea36e647b723ec040dd1ed740b0c9493b7a0`;
  - exit-code file: `ca2ebdf97d7469496b1f4b78958f9dc8447efdcb623953fee7b6996b762f6fff`;
  - fail-closed summary: `10834ccd7cf17ab7de3570a583cc907371b6e737b071000d14e592e3f240d5fb`.
- Prior successful retimed cases required approximately `244-304 s` wall
  time for about `33 s` simulated time. Case 13 permits up to
  `3 * 40.269133 = 120.807399 s` simulated time, so the fixed wall cap was not
  sufficient to observe its existing simulation horizon.

## Next round after Round 43

- Use a fresh case-13-only retry namespace with a bounded `1600 s` shell wall
  timeout recorded in the hashed runtime admission. Preserve the simulation's
  existing `3.0` duration-scale horizon and every physical/quality gate.
- This retry is an orchestration/evidence correction only. Do not change the
  source plan, LQR, controller, camera correction, thermal model, residual
  scales, or deterministic commands.

## Round 44: case-13 bounded retry dynamic qualification

- Published the fresh retry contract at commit
  `df0ee39d99e82013f57a8c870a3e070e0911c25c` and ran only case 13 in
  `20260718_gate_c_smoothed_case13_v10_camera_lever_arm_retry_v2_exclusive`.
  The hashed admission records the `1600 s` shell wall bound; the simulation's
  own `3.0` duration-scale horizon and every controller/gate remained
  unchanged.
- The case completed `19617` physics steps and both clocks:
  `33.821283 s` source and `40.269133 s` execution. Position p95/max passed at
  `0.149032/0.160031 m` against the unchanged `0.15/0.25 m` gates.
- Attitude p95/max passed at `0.135536/0.416216 deg`; pitch max was
  `6.929054 deg`; internal attitude IK had zero failures, proxy rate max was
  `44.246564 deg/s`, and no termination occurred.
- Action/proxy saturation ratios were `0.000535/0.000187`; riser saturation
  was zero. Riser effort max was `19.971910 N`, thermal load max was
  `0.001568`, and all thermal, runtime, and controller-evidence gates passed.
- The frozen prospective residual envelope passed with normalized maxima
  `[0.784293, 0.267138, 0.108635]`. Residual action remained exactly zero and
  unapplied, and no dataset was written.
- This raises the uniquely dynamically qualified set to `26/70`. Residual
  capture, BC, PPO, and training remain closed.
- Evidence hashes:
  - runtime admission: `073428109be413f5286e193b453f5419e979c9852e9a69d161495242fa9b03d6`;
  - case-13 gate JSON: `28c861bd8a550363c06973087872bf75f965d060133a98f4714748c69d66241c`;
  - case-13 log: `8cdace55e4e451378ad382f82312a4fb3009a82234d69badf8b73549c36973c9`;
  - final summary: `fd6624ed5c75974fbf713b5ce7be9909fa62a03316514c904f3afd1677ef26c8`.
- The case exit code is `0`; GPU and playback ownership were empty after
  closure.

## Next round after Round 44

- Continue in source order with a fresh, hash-bound v10 case-14-only canary.
  Determine and record a sufficient bounded wall timeout from its execution
  duration before launch rather than reverting to a universal `480 s` cap.
- Keep source anchors, LQR, deterministic controller, camera correction,
  physical gates, thermal model, and residual scales unchanged. Keep residual
  capture, BC, PPO, and training closed.

## Round 45: case-14 dynamic qualification

- Published the case-14-only runtime contract at commit
  `23dd27cc63a85b15a31fba92770e202661ad9b37` and ran it in
  `20260719_gate_c_smoothed_case14_v10_camera_lever_arm_v1_exclusive`. The
  hashed admission records the `2100 s` shell wall bound and only plan
  `e863db5bc93c25bf91f31ac6dbcbd11fa091830290aaf64c58a4a3982d5cae58`.
- The case completed `25777` physics steps and both clocks:
  `33.754608 s` source and `52.432971 s` execution. Position p95/max passed at
  `0.076035/0.135208 m` against the unchanged `0.15/0.25 m` gates.
- Attitude p95/max passed at `0.123322/0.239985 deg`; pitch max was
  `6.822419 deg`; internal attitude IK had zero failures, proxy rate max was
  `49.104633 deg/s`, and no termination occurred.
- Action saturation was `0.0000776`; proxy and riser saturation were zero.
  Riser effort max was `19.640121 N`, thermal load max was `0.001763`, and all
  thermal, runtime, and controller-evidence gates passed.
- The prospective residual envelope passed with normalized maxima
  `[0.909717, 0.323437, 0.127760]`. Residual action remained exactly zero and
  unapplied, and no dataset was written.
- This raises the uniquely dynamically qualified set to `27/70`. Residual
  capture, BC, PPO, and training remain closed.
- Evidence hashes:
  - runtime admission: `15d4671cabe75006a8fbd9d4b53dabc968f21fc97a1ac4d33672442cd269d33f`;
  - case-14 gate JSON: `b3d3774c60f04163ca235bf55707fdccf52b4f4bb982dc6afade5fd0482c23cf`;
  - case-14 log: `2593d142e2fc6136cf6de0b3ff940d44401a88ef35cecdead2653584d72dcd37`;
  - final summary: `6dd9a2978cd1e68e12a31b8e0b9fca4f94595d18a5e86217f232ef50f307db4d`.
- The case exit code is `0`; GPU and playback ownership were empty after
  closure.

## Next round after Round 45

- Inspect v10 case 15 CPU admission and execution duration. If admitted, add
  a fresh case-only contract with a duration-derived bounded wall timeout.
- Keep the exact-source/derived provenance boundary and all learning closures
  unchanged.

## Round 46: case-15 dynamic qualification

- Ran only case 15 at commit `4722a915aa391d4a7ceee691576716cfdee64025`
  in `20260719_gate_c_smoothed_case15_v10_camera_lever_arm_v1_exclusive`,
  bound to plan `8626af7d6d2feeb22d0eb4b2136f0617f91f1fbd3dc87c639d0f459f3c38c25f`
  and an `1100 s` wall timeout.
- It completed `12978` steps and `17.277396/25.052774 s` source/execution
  clocks. Position p95/max passed at `0.065091/0.067957 m`; attitude p95/max
  passed at `0.122486/0.230283 deg`; pitch max was `6.871187 deg`.
- IK failures were zero, proxy rate max was `43.741906 deg/s`, action
  saturation was `0.000308`, and proxy/riser saturation were zero. Riser
  effort/thermal maxima were `22.957657 N` and `0.001314`.
- Dynamic, thermal, runtime, controller, and residual-envelope gates all
  passed. Residual actions stayed zero and unapplied; no dataset, capture, BC,
  PPO, or training was started. The uniquely qualified count is now `28/70`.
- Evidence hashes: admission `e81bc0dc671fb82e539c818c353ef37f6a832d35bf9537980443e27e37c29907`,
  gate `481188488d236ac0abbbf40488e02149e20197bacb97862b6639d79215aaf384`,
  log `dbe38ed29fd6155f99c132dc7b17957757e32adb73a31a5acd93ecbf911077b8`,
  summary `2818516a93c2c5d6b061bb1748eb1ae93378ed3838ce8827e3dc225b589940be`.
- GPU and playback ownership were empty after closure.

## Next round after Round 46

- Continue source-order CPU admission with case 16, then use one fresh bounded
  case-only canary if admitted. Keep all learning stages closed.

## Round 47: case-16 p95 rejection and capped CPU retime

- Case 16 completed `11446` steps and both `17.548706/33.648216 s`
  source/execution clocks at commit `a84d13ea006ebc7c6053a2ba5a63a287aa7fbd53`.
  It failed only position p95 at `0.170346 m`; position max `0.186027 m`,
  attitude, balance, rates, thermal, runtime, controller evidence, and the
  residual envelope all passed. No dataset or learning stage started.
- Evidence hashes: admission `4cab04086c83724758772b863b6fe7a85563f102695659317e06873fc7f77349`,
  gate `878c7612608539136abc745d22fd72e5bd652eec8e83c0252a7106918e7fb93c`,
  log `3ea6605f7d3e14cbd4907d50a149a0d84ae8e0cae26d0ead65956db7bf9ff35f`,
  summary `4abf933e548c689e788c18cc7a57dbe2a23ca7b0c637f7264df8af6915709ae6`.
- Derived the only allowed uniform timing candidate at the unchanged `2.0`
  execution/source cap. All 896 source anchors, timestamps, geometry,
  attitudes, ordering, and initialization arrays remain unchanged; execution
  duration becomes `35.097412 s` and feed-forward rates scale consistently.
- The CPU candidate and v11 portfolio pass all integrity, transition,
  kinematic, duration, and median gates. The portfolio remains `70/79`
  admitted with median `1.499794`; all artifacts remain invalid for training.
- CPU hashes: retimed plan `8bcf14454ce4b087973e0c0d2c6efb3858edf75209e195dec7fc09fe7111c821`,
  derivation manifest/summary `23d822018be803d65d3de4cf21f4357678c0425e797f6a89e157231eafef7f40` /
  `b1ca295565a4c03808a239f7d32112a8391d4277177bf81dd5fa204d3fd4441e`,
  v11 manifest/summary `56670dd0ecbdf0157361bef65af50f8d688a9e86bc3e0ff50768472b17474032` /
  `b711bfdbbc4fb15497732c3813161a2164a4dc9d888afadd977f9047ae4e4aa3`.

## Next round after Round 47

- Add one fresh hash-bound v11 case-16-retime canary and run it under unchanged
  LQR/controller/physical gates. Do not advance to case 17 or learning first.

## Round 48: case-16 duration-cap exhaustion and structural localization

- Ran the v11 `2.0x` retimed case 16 at commit
  `e76ba071d8b267b3bd1540681670ec9c033bcb10`. It completed `11603` steps and
  both `17.548706/35.097412 s` clocks, but again failed only position p95:
  `0.168950 m`. Position max `0.184844 m` and all attitude, balance, rate,
  thermal, runtime, controller, and residual-envelope gates passed.
- Uniform slowing improved p95 by only `0.001396 m` from the baseline
  `0.170346 m`; the frozen `2.0` execution/source cap is now exhausted.
- Evidence hashes: admission `e4b44535ace27b5673d654b07568c386d2eed52c9d6df736d27b773a3f766f12`,
  gate `9a8186d4467b52639b8dbac455681174ce5624d5f14229035bac4c2d4c81fc5e`,
  log `2c94ad446fa7a8cb528ac125f9b9482a5afe90cccf70156e6eae331fe9361939`,
  summary `78345a0f1ab15b43f29d9ec0d0b18c3b2a3d87dfb21176939b39c59b17c50d89`.
- The 1 Hz trace localizes the peak to forward-motion phase `7.2-12.0 s`,
  dominated by camera cross-track error while the unchanged `0.05 m`
  lever-arm correction is saturated. This is not an endpoint, reverse-motion,
  attitude, riser, or thermal failure.
- The parent static planner selected its first passing `0.10 m` lookahead
  candidate and did not evaluate `0.15 m`. No correction cap, threshold,
  controller gain, source anchor, or learning flag was changed.

## Next round after Round 48

- Add a CPU-only, case-16-specific explicit-lookahead derivation path and test
  `0.15 m` while preserving the source camera path and all duration/transition
  gates. Do not launch case 17, widen the `0.05 m` correction, or start learning.

## Round 49: case-16 explicit-preview recovery and dynamic qualification

- Added a CPU-only, hash-bound explicit-preview derivation. Its first two
  fail-closed invocations exposed and fixed evidence-path defects before any
  artifact or runtime existed: binary NPZ parents are now hashed without JSON
  decoding (`5ffe01fda72d67250d54b7a3715c9cc186bbf5a7`), and the authoritative
  source count is validated using `episode_count` plus the item count
  (`27574668625e55f858fac72df401d6165775b948`). The authoritative remote suite
  remained `325 passed` after both repairs.
- The explicit `0.15 m` lookahead and unchanged `2.75` heading gain produced a
  statically admitted case-16 plan at source/execution clocks
  `17.548706/26.028630 s`, ratio `1.483222`. All 896 source positions,
  attitudes, timestamps, anchor order, path length, and initialization fields
  remain immutable. Static position p95/max are `0.042568/0.058537 m`.
- CPU artifact hashes: plan
  `742d1f705d3559916c3e1d7d35caffd5ea9e7200b6e321d1f9f70c8e5a7dad16`,
  derivation manifest/summary
  `cc4c00dcd77deeaed1c02bec3fe3f2b86d75d25dd58d1fef164ce2d84bbb2ff9` /
  `5eb852ad1a1bdd59c0fe5b6c28e09c7d29af654184d73ff5441cf125a2949828`.
- Composed the v12 all-79 portfolio without altering any other case. It remains
  `70/79` statically admitted; its accepted duration-ratio median is
  `1.496427 < 1.5`. Manifest/summary hashes are
  `59e572712879e25a687bebd17be94b8464e7f0de08ef3d5ce2102bb9303a5581` /
  `e97da6859c9d3da5fb8a1ed3d7b842b5880005c25c56e7e26f6f73833a7e4ff3`.
- Published the one-case runtime route at
  `0d0939bdbc80a8a927e8c6017206b03fcf25a732` and ran only case 16 in
  `20260719_gate_c_smoothed_case16_explicit_preview015_v12_exclusive`.
  It completed `10445` steps and both clocks. Dynamic position p95/max passed
  at `0.080600/0.081492 m`, improving from the capped-retime reject's
  `0.168950/0.184844 m` without changing source geometry, LQR, controller
  gains, the `0.05 m` camera correction bound, or physical thresholds.
- Attitude p95/max passed at `0.119408/0.214863 deg`; pitch max was
  `6.030922 deg`; internal attitude IK had zero failures; proxy rate max was
  `43.660585 deg/s`; action, proxy, and riser saturation were all zero.
  Riser effort/thermal maxima were `22.748663 N` and `0.001462`.
- Dynamic, thermal, runtime, controller-evidence, and residual-envelope gates
  all passed. Normalized prospective residual maxima were
  `[0.212769, 0.352328, 0.127748]`; residual action remained exactly zero and
  unapplied, and no dataset, capture, BC, PPO, or training was started. The
  uniquely dynamically qualified set is now `29/70`.
- Dynamic evidence hashes: admission
  `7065aff16249cc0a6f4022c5ac63c34788cf9455a23ff57cfd95adf87b79827a`,
  gate `b51e7e8b65a0da12e792d93d31c897fa6e2d5dd2a9a2d030d98dbbfb76634bec`,
  log `a29e6a4063b3f017b023cc9dd761d5659c5b5a5f63d14e51239e092bc04becc9`,
  summary `408edb317561c002d9524f95c34fb1775475bae3054509a8c4f1201deed33362`.
  GPU and playback ownership were empty after closure.

## Next round after Round 49

- Continue in source order with CPU inspection of v12 case 17. If it remains
  admitted and unchanged, add one fresh hash-bound case-only route with a
  duration-derived wall timeout and run only that canary.
- Keep exact-source provenance, LQR/controller settings, all physical gates,
  and the learning closure unchanged.

## Round 50: case-17 dynamic qualification

- CPU inspection confirmed that v12 case 17 is unchanged, statically admitted,
  and hash-bound to plan
  `e38228121caf797546ac0936fc522e84f61f04cd3740438e0b93469665fa938d`.
  Its source/execution clocks are `17.376415/31.450542 s`; static position
  p95/max are `0.093125/0.238651 m`, with no failed timing, path, transition,
  or kinematic check.
- Published the fresh case-only runtime route at
  `2b06d9a7d4803326b4aceae266be96b3200c7691`, using the unchanged v12
  portfolio and a `1500 s` wall bound. The authoritative remote suite passed
  all `325` tests before launch.
- Ran only case 17 in
  `20260719_gate_c_smoothed_case17_v12_camera_lever_arm_v1_exclusive`. It
  completed `16305` steps and both clocks. Dynamic position p95/max passed at
  `0.143606/0.216037 m`; attitude p95/max passed at
  `0.131835/0.234283 deg`; pitch max was `7.226670 deg`.
- Internal attitude IK had zero failures, proxy rate max was
  `42.988322 deg/s`, action saturation was `0.000123`, and proxy/riser
  saturation were zero. Riser effort/thermal maxima were `26.933525 N` and
  `0.001342`; no termination occurred.
- Dynamic, thermal, runtime, controller-evidence, and residual-envelope gates
  all passed. Normalized prospective residual maxima were
  `[0.911113, 0.420583, 0.147586]`; residual actions remained zero and
  unapplied, and no dataset, capture, BC, PPO, or training was started. The
  uniquely dynamically qualified set is now `30/70`.
- Evidence hashes: admission
  `dc1307559e4ea3ae4fcf283c3d4f1dc35cf538a24905495b2d60ff4b17c6ec9f`,
  gate `eb9da572b528d74119b957200ad21461e94637ac4a83a069bcb7ca0b7e12ee71`,
  log `65d5f2cb2aaa1c5254aa50fe3da5499a309ea64119285e6ed064caa077782703`,
  summary `c437491992e671c101aae2c5b03b9b02a50cb656c2b33994089f29f6a5a0278c`.
  GPU and playback ownership were empty after closure.

## Next round after Round 50

- Inspect v12 case 18 on CPU and compare its static margin and execution clock
  with the now-qualified cases. If admitted, create a fresh case-only runtime
  route and run no other case in parallel.
- Preserve all controller, source, correction, gate, and learning-closure
  invariants.

## Round 51: case-18 long-duration dynamic qualification

- CPU inspection confirmed that v12 case 18 is unchanged and statically
  admitted, with plan hash
  `121b0f336dd1e236aaee2b9bf0b158466636624507c107e2d90935339edf2517`,
  source/execution clocks `35.888335/56.216910 s`, and static position p95/max
  `0.080594/0.203560 m`. No timing, path, transition, or kinematic check failed.
- Published the fresh case-only runtime route at
  `abfe84f258cad9b50ed46465191ed9fc602865d6`, with a duration-derived
  `2200 s` wall bound. The authoritative remote suite passed all `325` tests
  before launch.
- Ran only case 18 in
  `20260719_gate_c_smoothed_case18_v12_camera_lever_arm_v1_exclusive`. It
  completed `26966` steps and both clocks. Dynamic position p95/max passed at
  `0.124703/0.211728 m`; attitude p95/max passed at
  `0.127902/0.223177 deg`; pitch max was `7.665609 deg`.
- Internal attitude IK had zero failures, proxy rate max was
  `43.581408 deg/s`, action saturation was `0.0000742`, and proxy/riser
  saturation were zero. Riser effort/thermal maxima were `22.537685 N` and
  `0.001525`; no termination occurred.
- Dynamic, thermal, runtime, controller-evidence, and residual-envelope gates
  all passed. Normalized prospective residual maxima were
  `[0.982979, 0.357060, 0.143591]`; residual actions remained zero and
  unapplied, and no dataset, capture, BC, PPO, or training was started. The
  uniquely dynamically qualified set is now `31/70`.
- Evidence hashes: admission
  `53ca74b8480227cbf42a4331e0c637af61afb80ffe7893989d49a7b08e20e11e`,
  gate `bdf6799e7aa663e7990ac54456317dbae381738e4f54550da1b8bef3dec227c6`,
  log `6e8730ab433fb975f61426218cd1d18fd47fff03e439562177647128b9f61893`,
  summary `cb42585916d4fbd3be696198f0c1a23a306260137a7765ff1f971e8993931e7c`.
  GPU and playback ownership were empty after closure.

## Next round after Round 51

- Inspect v12 case 19 on CPU. If admitted, create a fresh case-only route with
  a duration-derived wall cap and preserve source-order fail-fast execution.
- Do not start residual capture or learning until deterministic qualification
  and the subsequent raw residual-envelope recomputation are complete.

## Round 52: case-19 dynamic qualification

- V12 case 19 was unchanged and statically admitted with plan hash
  `8cf8bde298c73d1809c3dc7c0dae249446d7554ba77275a490d32fc1a6004b37`,
  source/execution clocks `8.338212/12.028271 s`, and static position p95/max
  `0.039550/0.059349 m`.
- Published the case-only route at
  `ea27a43`; the authoritative remote suite passed all `325` tests before
  launch. Ran only case 19 in
  `20260719_gate_c_smoothed_case19_v12_camera_lever_arm_v1_exclusive`.
- It completed `6022` steps. Dynamic position p95/max passed at
  `0.085048/0.088555 m`; attitude p95/max passed at
  `0.141937/0.214087 deg`; pitch max was `7.035737 deg`. IK failures and all
  action/proxy/riser saturation were zero; no termination occurred.
- Proxy rate max was `41.349092 deg/s`; riser effort/thermal maxima were
  `21.486368 N` and `0.000961`. All dynamic, thermal, runtime,
  controller-evidence, and residual-envelope gates passed. Normalized
  prospective residual maxima were `[0.838843, 0.168118, 0.118204]`; no
  residual action, dataset, capture, BC, PPO, or training was started. The
  uniquely dynamically qualified set is now `32/70`.
- Evidence hashes: admission
  `4717c2ccded67e47970f01dbbc379be54e6a526d940604453b6cf58a8a5f9299`,
  gate `d1b103a3f723d02e9b33b9e458479bac9437bd2bda46e34ce44fafda96c9e8a6`,
  log `7fc2049ff64a04fdc498c2c66f8a2c9af3ce00b1815d947dceb7aec816adf627`,
  summary `adfaa597d3f586e1e97a4b436c9f6005317cf2414d1b6b3767315983a3abb87b`.
  GPU and playback ownership were empty after closure.

## Next round after Round 52

- Continue source-order CPU inspection with v12 case 20 and create a fresh
  case-only route only if all static gates remain admitted.

## Round 53: case-20 narrow dynamic and residual-envelope rejection

- V12 case 20 was statically admitted with plan hash
  `ec0bb2845c948d17daec8abef6b00b205f6f56fe6cb9e4c42aa9395c6b66336d`,
  source/execution clocks `7.261100/14.469192 s`, ratio `1.992700`, and static
  position p95/max `0.139940/0.165549 m`.
- Published the fresh case-only route at
  `d2280c002f13c4c5e72ff1c32a255fb3d9f36f03`; all `325` authoritative tests
  passed before the exclusive launch. Case 20 completed all `7460` steps in
  `20260719_gate_c_smoothed_case20_v12_camera_lever_arm_v1_exclusive`.
- The physical run failed only position p95: `0.154389 m` against the unchanged
  `0.15 m` limit. Position max passed at `0.163111 m`; attitude p95/max passed
  at `0.135357/0.239113 deg`; pitch max was `7.873634 deg`; IK failures,
  action/proxy/riser saturation, and termination were zero.
- Thermal and controller evidence passed. Proxy rate max was
  `39.716284 deg/s`; riser effort/thermal maxima were `29.560146 N` and
  `0.001248`.
- The residual-label envelope independently failed: raw vx residual reached
  `0.302311 m/s`, normalized `1.007705` against the frozen scale. It was not
  clipped or applied; residual action stayed zero and no dataset, capture, BC,
  PPO, or training started.
- The 1 Hz trace localizes the largest XY errors around execution/phase
  `12.0/5.97 s` and `28.0-29.0/11.36-11.68 s`. Camera lever-arm correction was
  saturated for `98.78%` of samples. The plan already consumes the frozen
  near-`2.0` duration cap, so further uniform retiming is not admissible.
- Evidence hashes: admission
  `87ebaab1dcb5d8bc2283a0e216befeb40cd0392c3600dc6dda47a1b7c42b10ce`,
  gate `9dd45696d26b8aa1ccf4f6025a9646e8cb5f1c9c25ee50eb0c159eae91d5d14e`,
  log `8d9c31844be2667c218f574fcd6c77a4a7f19cfc471f89862fb454c094ff4968`,
  summary `d0d09404f99f6d69cf84f3e0e0dc56c276cb916f327f4429c20165d71026a79a`.
  GPU and playback ownership were empty after closure. The uniquely qualified
  count remains `32/70`.

## Next round after Round 53

- Perform CPU-only case-20 preview recovery analysis while preserving the
  already-selected smoothed target geometry, source arrays, duration cap,
  controller, `0.05 m` correction cap, and all physical thresholds.
- Do not run case 20 again or advance to case 21 until a fresh candidate passes
  duration, path, transition, and kinematic gates. Keep learning closed.

## Round 54: geometry-preserving preview bracket exhausted

- Added a hash-bound CPU derivation path that reuses the parent NPZ's selected
  smoothed target positions byte-for-byte while changing only preview. It pins
  parent smoothing sigma/blend/reset-yaw identity and proves source arrays,
  parent smoothed geometry, controller, and thresholds unchanged. Implementation
  commit: `3a28480`; explicit playback strategy integration: `8cb57f4`.
- Repaired the derivation to preserve rejected candidates as machine-readable
  CPU evidence rather than raising before output (`53b6188`). The authoritative
  remote suite is now `327 passed`.
- Evaluated a bounded bracket around the parent `0.90 m` preview, all with
  heading gain `1.0`, sigma `64`, blend `0.1276273593606172`, and forward-path
  reset yaw:
  - `0.875 m`: ratio `2.002022`, p95/max `0.183379/0.205045 m`; failed duration
    and position p95. Manifest/summary
    `c276309c0bdd2fd9e32ecd1dd9356491247d63fff69802014128ab890129e31a` /
    `0f33466562b7c9f10af908a09d3e20bc60e97c20e325fe30608e6686f1da0ad7`.
  - `0.95 m`: ratio `1.993704`, p95/max `0.190707/0.211959 m`; failed position
    p95. Manifest
    `696d8efbb78ebbdd95f802ea6f3742dfd8ce88449b32ebf67362931e40c2a0e5`.
  - `1.00 m`: ratio `1.988732`, p95/max `0.200773/0.216067 m`; failed position
    p95. Manifest/summary
    `c0ddda3917a9025d5c069542dd24e40b88f7df9a94c36c7ee51a19dadb27d46c` /
    `d504ec8221d37a4bada86c47646139de04ff9044d9872fa005b948f9db432565`.
- Every candidate preserved parent smoothed geometry and all source arrays;
  Isaac, residual capture, BC, PPO, and training remained closed. The original
  `0.90 m` parent remains the best admissible preview, so preview-only recovery
  is exhausted and no dynamic retry is authorized.

## Next round after Round 54

- Perform CPU-only localization of case 20's base-path/yaw allocation around
  the two sealed XY error intervals. Evaluate one bounded geometry-preserving
  base-state/heading allocation change with explicit command-delta, duration,
  transition, and kinematic evidence.
- Do not widen the camera correction cap or residual scales, relax thresholds,
  launch Isaac, or advance to case 21 before that CPU candidate passes.

## Round 55: fixed-geometry base-allocation recovery rejected as a no-op

- Added a fixed-geometry batch-unicycle derivation path that was required to
  preserve target/source geometry and clocks while reporting parent-relative
  base-state and command deltas. The first artifact reported an internal
  optimizer command delta of `0.05`, but a direct NPZ audit proved every
  emitted array was byte-identical to the parent, including `base_xy_yaw`,
  `feedforward_v_wz`, riser/proxy states, target positions, and execution time.
- Added a fail-closed no-op guard at `b012993`. The resealed CPU artifact
  `20260719_case20_fixed_geometry_batch_unicycle_v2_noop_guard_cpu` exits `6`
  with all actual parent-relative deltas zero and
  `base_allocation_changed=false`. Manifest/summary hashes are
  `0cce3a1b409c9d4ca8a20285a7196dd9cd3fb363c5e9c057da4301fb9843ef5c` /
  `cd8e8e542971e695a9f2edd768963c0abc67ea61594aa2e2c7e058d8f4cb08bc`.
- This closes the current static base-allocation route honestly. No candidate
  was composed, no Isaac run was authorized, and learning remained closed.

## Round 56: bounded camera-error phase-recovery candidate, CPU only

- The sealed case-20 trace shows that the existing governor already consumes
  `37.300 s` of a `43.407577 s` runtime bound. The position-p95 reject is only
  `4.389 mm` over threshold and is localized near wall/phase times
  `12.0/5.967 s` and `28.0-29.0/11.360-11.680 s`; therefore neither global
  retiming nor a broader gain/correction change is justified.
- Added an opt-in continuous phase cap that activates only when measured camera
  correction is saturated and physical camera-position error exceeds
  `0.13 m`. It reaches a frozen minimum scale of `0.20` at `0.155 m`:
  `g(e)=1` below `0.13`; `g(e)=1-(e-0.13)/0.025*0.8` in the transition; and
  `g(e)=0.20` at or above `0.155`. Runtime phase progress is the minimum of
  the existing tracking-error governor, balance governor, and this new cap.
- This changes only how quickly the immutable plan phase advances. It does not
  alter source/plan arrays, LQR, controller or tracking gains, fixed-phase
  commands, the `0.05 m` camera-correction cap, the `3.0x` horizon, physical
  thresholds, or residual scales. The feature is disabled by default.
- A 1 Hz counterfactual over the sealed trace would reduce progress at only two
  recorded rows: `0.499167 -> 0.20` at wall `12 s`, and
  `0.314514 -> 0.201219` at wall `28 s`. Its coarse additional-wall estimate
  is `2.058879 s`, below the existing `6.107577 s` runtime margin. This is a
  CPU diagnostic, not evidence that the dynamic gate will pass.
- Added fail-closed identity and telemetry fields for profile
  `riser_recovery_direction_v4_camera_lever_arm_error_governor_v1`. A future
  summary must prove exact thresholds, policy-rate sample parity, bounded
  progress scales, nonzero activation, unchanged learning closure, and the
  existing camera/LQR/runtime contracts. No runtime authorization route or
  namespace was created; Isaac, case 21, capture, BC, PPO, and training remain
  closed.

## Next round after Round 56

- Run the authoritative CPU suite and diff-audit this candidate. Only after a
  clean pushed commit, explicit review, and exclusive GPU release may a fresh
  case-20-only canary route be authorized.
- If the bounded canary fails completion or any physical/controller gate, stop
  and preserve the first reject. If it passes physical dynamics, keep residual
  admission independent and do not advance to case 21 until the evidence is
  sealed.

## Round 57: case-20 bounded camera-error governor passes Gate C

- Published the fresh case-only, hash-bound runtime route at `7ab2959`; the
  authoritative `.98` CPU suite passed `334/334` tests before launch. The route
  pinned v12 plan
  `ec0bb2845c948d17daec8abef6b00b205f6f56fe6cb9e4c42aa9395c6b66336d`,
  the `0.13/0.155/0.20` governor, unchanged `0.05 m` correction cap and
  physical gates, clean `HEAD==upstream`, and exclusive GPU ownership.
- Ran only case 20 in
  `20260719_gate_c_smoothed_case20_v12_camera_error_governor_v1_exclusive`.
  Both source/execution clocks completed in `7486` steps. Position p95/max
  passed at `0.149128/0.155961 m`, improving from the sealed reject's
  `0.154389/0.163111 m` without changing the plan, LQR, gains, correction cap,
  horizon, or thresholds.
- Attitude p95/max passed at `0.135390/0.257223 deg`; pitch max was
  `7.851979 deg`. Dynamic, thermal, controller-evidence, and runtime-contract
  outcomes all passed. The governor activated for `21.6805%` of policy-rate
  samples with minimum/mean cap `0.20/0.914336`; total wall duration was
  `37.43 s`, still below the unchanged `43.407577 s` bound.
- The independently evaluated prospective residual envelope also passed:
  normalized maxima were `[0.988451, 0.541243, 0.153122]`. Residual actions
  remained zero and unapplied; no dataset, capture, BC, PPO, or training was
  started. The uniquely dynamically qualified set is now `33/70`.
- Evidence hashes: admission
  `2c0baa1d4dd1a5056db920a52575d6448ec30ecfb0de93b66712a7ed5449a9b0`,
  gate `ffbd741b8a4423b988ab7a13c075fb8726edc8d2cd47b41fe8def9442a96f792`,
  log `29af97c04354b6833676873bb081062e6f627718a7b600df092b6cb6a1d748f1`,
  and summary
  `c00896e0b13eaab5fb18cbf338f01c78e431707e49177618d9c04633d2e13860`.
  GPU and playback ownership were empty after closure.

## Next round after Round 57

- Resume source-order CPU inspection with v12 case 21. Do not automatically
  apply the case-20 recovery profile: first inspect its static margin and only
  use the default accepted controller unless case-specific evidence justifies
  otherwise.
- Keep fail-fast dynamic execution and all learning stages closed. The case-20
  result qualifies one deterministic trajectory; it does not authorize
  residual capture or establish a universal teacher transformation.

## Round 58: case-21 reversal transient is the next fail-fast reject

- CPU inspection admitted unchanged v12 case 21 with plan hash
  `85029afbbcce435ec8df27770b521b0ab57eae8d98ab4a2dc7f7b7680efaa9ba`,
  source/execution clocks `13.495780/18.753269 s`, ratio `1.389565`, and static
  position p95/max `0.090662/0.119344 m`. Published the default-profile,
  case-only route at `0cfb6bc`; `.98` passed `334/334` CPU tests before launch.
- Ran only case 21 in
  `20260719_gate_c_smoothed_case21_v12_camera_lever_arm_v1_exclusive`. It
  completed both clocks in `5368` steps and failed only dynamic position p95:
  `0.156808 m` against `0.15 m`. Position max passed at `0.195109 m`; attitude
  p95/max passed at `0.192667/0.220782 deg`; pitch max was `6.317007 deg`;
  termination, IK failures, and all physical/controller/thermal checks passed.
- The prospective residual envelope independently failed at normalized maxima
  `[1.018811, 0.537389, 0.113703]`; it remained unapplied and no dataset or
  learning process started. The uniquely qualified count remains `33/70`.
- Trace localization shows a forward-to-reverse transition around execution
  phase `6.35-6.39 s`: planned feedforward changes from approximately
  `+0.013` through zero to `-0.40 m/s`, with local finite-difference peaks up
  to `16.88 m/s^2`. At wall `9-10 s`, base error grows from `0.157` to
  `0.227 m`, while the command reaches `-0.40/-0.346 m/s`. This is a
  longitudinal reversal transient, not a camera lever-arm saturation problem.
- Applying the case-20 governor formula to the sealed trace would affect only
  one 1 Hz row (`0.205 -> 0.20`) and add a coarse `0.023 s`; therefore it is
  not an evidence-based recovery for case 21 and was not run.
- Evidence hashes: admission
  `7946840a02ab914901dc9fb2c5397d879c08098808ce435598f0bfe8ca2f0350`,
  gate `22bf703472f4b0786120a420622f5af3e47067c61f3f44f018c6e3298e03bed0`,
  log `5c297e10cb236be91f4bb748bad64b52ce4b084b5e80a432cb130ff208910a8e`,
  and summary
  `623c872ec80c5b13de855f741280363250246ce8adbd92ce0649bc1ac0cd53d8`.
  GPU and playback ownership were empty after closure.

## Next round after Round 58

- Stay CPU-only and derive one localized case-21 time reparameterization around
  the `6.35-6.39 s` reversal. Preserve source anchors/order and target
  geometry, keep initialization separate, and remain within the frozen `2.0x`
  duration ratio while adding explicit longitudinal acceleration/transition
  evidence. Do not change LQR, correction cap, gates, or residual scales.
- Do not launch a retry until that candidate passes duration, path, transition,
  and kinematic checks and demonstrates a nonzero parent-relative timing and
  command change. Case 22 and every learning stage remain closed.

## Round 59: localized case-21 reversal retime passes CPU admission

- Added a reusable local execution-time scaling primitive at `9871991`. It
  applies a smooth endpoint-neutral interval taper, never speeds an interval,
  preserves all states/source arrays, recomputes every feedforward derivative,
  rejects duration overflow, and reports command-transition evidence. The
  authoritative `.98` suite passed `340/340` tests.
- Derived case 21 over intervals `230:340` with peak time scale `4.0`. All 693
  source anchors, target geometry, base/riser/proxy states, source timestamps,
  initialization separation, LQR/controller settings, and physical thresholds
  remained unchanged. Execution duration changed from `18.753269` to
  `20.284171 s`, ratio `1.503001 < 2.0`.
- In the reversal window, maximum linear acceleration fell from
  `16.881930` to `4.891088 m/s^2` (`0.289723x`), command-transition norm from
  `0.162606` to `0.108473` (`0.667093x`), and maximum linear velocity from
  `0.400000` to `0.237283 m/s`. Every duration/path/transition/kinematic and
  nonzero parent-delta check passed.
- CPU candidate plan hash is
  `81c0da4be22d5b800978d1d46ca9705912f72007f7c615b31715c672dd86a1d4`;
  derivation manifest hash is
  `4dcf6151e8ec6d247adcbddc2de41add08d647217de5b067597f52dab8484ac0`.
- Added the replacement schema to the fail-closed portfolio composer at
  `823db97`; `.98` passed `341/341` tests. V13 portfolio manifest hash is
  `40611139cb50c4431c238994f311e578c6b43f754ad07b700ec54576a8574e3e`.
  It retains `70/79` CPU-admitted plans and accepted duration median
  `1.499794 <= 1.5`. No Isaac or learning stage ran during derivation.

## Round 60: case-21 localized reversal plan passes Gate C

- Published the fresh v13 case-only route at `91fb909`; the authoritative
  `.98` suite again passed `341/341` tests before launch. Ran only case 21 in
  `20260719_gate_c_smoothed_case21_v13_localized_reversal_v1_exclusive` under
  the unchanged default tracking/LQR profile.
- Both clocks completed in `5486` steps. Dynamic position p95/max passed at
  `0.102706/0.108222 m`, improving from the parent's
  `0.156808/0.195109 m`. Attitude p95/max passed at
  `0.192802/0.215200 deg`; pitch max was `6.300506 deg`; no termination or
  controller/thermal failure occurred.
- The independently evaluated prospective residual envelope also passed:
  normalized maxima changed from the rejected parent's
  `[1.018811, 0.537389, 0.113703]` to
  `[0.573797, 0.537389, 0.113652]`. Residual action remained zero and
  unapplied; no dataset, capture, BC, PPO, or training started. The uniquely
  dynamically qualified set is now `34/70`.
- Evidence hashes: admission
  `c216d80eadd4c354549a60c8166a53e88db78cdf6ce57b91451e14d0ecca4658`,
  gate `5f71526f4b4d4e2055a4c1153b9e104d1d1eb88617d33fe6a67c730171ed1fb8`,
  log `bed79df285467ef0e24c1f24464fd5a843404c3cf1ca9614a75fc26f4371bc76`,
  and summary
  `f5555c95a17eb57802254bcf0121afe9f1fd4a4f6e0314cfcca12175ad704f83`.
  GPU and playback ownership were empty after closure.

## Next round after Round 60

- Resume source-order CPU inspection with v13 case 22. Use its unchanged
  default plan/profile unless its own evidence demonstrates a reversal or
  camera-error recovery need; do not generalize the case-21 retime blindly.
- Keep fail-fast Gate C execution and all residual capture/learning stages
  closed until deterministic qualification is complete and the raw residual
  envelope is recomputed over the final accepted portfolio.
