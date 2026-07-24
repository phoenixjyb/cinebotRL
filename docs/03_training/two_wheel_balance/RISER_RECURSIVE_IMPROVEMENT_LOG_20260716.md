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

## Round 61: case-22 default plan reproduces the paired reversal reject

- Published the fresh v13 case-only default route at `2e3f769`; the
  authoritative `.98` CPU suite passed `341/341` tests before launch. A first
  launch attempt correctly stopped before Python with exit `73` because a
  separate Isaac renderer owned the machine. The admitted run started only
  after Windows and WSL ownership were clear.
- Ran only case 22 in
  `20260719_gate_c_smoothed_case22_v13_camera_lever_arm_v1_exclusive`. Both
  clocks completed in `5640` steps at source/execution durations
  `13.495780/18.753269 s`. Dynamic position p95 alone failed at
  `0.156460 m`; position max passed at `0.201943 m`. Attitude p95/max passed
  at `0.191535/0.213396 deg`, pitch max was `6.529206 deg`, and every
  completion, termination, thermal, controller-evidence, runtime, IK, and
  saturation check passed.
- The 1 Hz trace reproduces the case-21 forward-to-reverse transient: phase
  advances from `5.531609 s` with `+0.035649 m/s` reference to `6.436596 s`
  with `-0.383448 m/s`, after which position error reaches `0.199612 m`.
  This confirms the paired XY path's reversal mechanism despite case 22's
  `+0.45 m` camera-height offset.
- The prospective residual envelope independently failed at normalized maxima
  `[1.178149, 0.529025, 0.125254]`; residual action remained zero and
  unapplied, with no dataset, capture, BC, PPO, or training.
- Evidence hashes: admission
  `d858b58cc079cfda0802aaf59c2b53a905ad31e5d75b9d0777e53a877180ed9d`,
  gate `d2b7678d81a7710818abf3ab2c8af536d484871630a2f8f458b060ac20050460`,
  log `0bbe00fc1a993f9473fbc58df1b1de904555f2fcff05a9a28c7da1d2313004b0`,
  and summary
  `94112f7f5b71a1a70c53332f2d65956a1c0680adf7e75babc59df75f4d1b118b`.

## Round 62: bounded case-22 localized reversal recovery passes

- Reused the already-tested localized retime primitive, but bound the
  derivation to case 22's own sealed v13 plan and Round-61 reject. A peak scale
  of `4.0` was rejected CPU-side because it would raise the accepted-set
  duration median above the frozen `1.5` ceiling. The strongest evaluated
  compliant scale, `3.9`, preserves that global invariant without relaxing it.
- The candidate changes only execution intervals `230:340`. All `693` source
  anchors/timestamps, target geometry, base/riser/proxy states, initialization,
  LQR, controller gains, correction cap, and physical thresholds are
  unchanged. Execution duration is `20.233141 s`, ratio `1.499220`.
  Reversal-window acceleration falls from `16.881937` to
  `4.993260 m/s^2` (`0.295775x`) and command-transition norm from
  `0.162607` to `0.109724` (`0.674780x`). Camera height remains
  `1.23934-1.80000 m`.
- The candidate plan hash is
  `8f1638cd771cfac32ca251906e2c095bd7091edb2561974f12ae09b0a65d4a79`;
  derivation manifest hash is
  `2b5237a041a7a671678aab772a94ee07a05f6a117b4a84804b5f72fc1d0807da`.
  V14 portfolio manifest hash is
  `369e3294a45ef468979a81a8bf34b9012f9ec4f77a1d4489c4514930f2d79dab`;
  it retains `70/79` CPU admissions with median `1.499904 <= 1.5`.
- Published the v14 case-only route at `a50e2d6`; `.98` again passed
  `341/341` CPU tests. The exclusive retry namespace was
  `20260719_gate_c_smoothed_case22_v14_localized_reversal_v1_exclusive`.
  Both clocks completed in `5721` steps. Dynamic position p95/max passed at
  `0.099300/0.113607 m`; attitude p95/max passed at
  `0.190204/0.216710 deg`; pitch max was `6.529206 deg`; all physical,
  thermal, controller, runtime, IK, and saturation checks passed.
- The prospective residual envelope also passed at normalized maxima
  `[0.595563, 0.529025, 0.117504]`, but remained unapplied. No dataset,
  capture, BC, PPO, or training started.
- A fresh union over all sealed Gate C result rows proves `34/70` unique
  dynamic passes after this run. This corrects the running count from Round 52
  onward: case 19 was already present in Round 30's initial 15-case qualified
  set, so Round 52 must not have incremented the unique count. No trajectory
  admission changed; this is bookkeeping correction from immutable evidence.
- Retry evidence hashes: admission
  `c46b1eabebfb66427454177bbd0324fa0c40187dc2dd78f7337acda918a28323`,
  gate `911855b08f4688431f5f4b346cebd42d4a9d388b98c1b131470f3b3e546f72a4`,
  log `39f2f704e3382ebee65b0fae6ffe405ec027db86deeb64112e4234f3c8447ddf`,
  and summary
  `60e23bcd2cd10834f6764b1396a4b470b783a9688cd71b0909907ce8adc11e07`.
  GPU and playback ownership were empty after closure.

## Next round after Round 62

- Recompute the union of sealed dynamic passes and continue with the first
  source-order v14 CPU-admitted case that is not already qualified; do not
  rerun previously sealed passes merely because their case number follows 22.
- Use the unchanged default profile first. Keep fail-fast execution and all
  residual capture/learning stages closed until deterministic qualification is
  complete and the raw residual envelope is recomputed over the final set.

## Round 63: case-30 full-rate riser plan passes Gate C

- Recomputed the dynamic-pass union from sealed result rows rather than the
  historical running counter. Cases `23-26` and `28` were already qualified,
  while cases `27` and `29` are not in the CPU-admitted v14 set; therefore
  case 30 was the first eligible unqualified source-order case.
- V14 case 30 is hash-bound to plan
  `1722bfdc7c1aeabc5a9d3920cf6a47bc789afbc96e6ef5c8e540695dc3c97dcb`.
  Its source/execution clocks are `18.144412/29.222488 s`; static position
  p95/max are `0.135337/0.168890 m`; target camera height is
  `1.256005-1.800000 m`; and its planned riser rate reaches the frozen
  `1.0 m/s` limit without exceeding it.
- Published the fresh case-only route at `ed5c57d` with a duration-derived
  `1500 s` wall bound. The authoritative `.98` suite passed `341/341` tests
  before launch. Ran only case 30 in
  `20260719_gate_c_smoothed_case30_v14_camera_lever_arm_v1_exclusive`.
- Both clocks completed in `11495` steps. Dynamic position p95/max passed at
  `0.141848/0.169449 m`; attitude p95/max passed at
  `0.153157/0.226069 deg`; pitch max was `7.034872 deg`. Riser servo p95/max
  were `0.011870/0.014594 m`; effort max was `32.964031 N`; thermal load max
  was `0.001203`; and every completion, termination, physical, thermal,
  controller, runtime, IK, rate, and saturation check passed.
- The prospective residual envelope passed with normalized maxima
  `[0.804779, 0.412764, 0.172961]`. Residual action remained zero and
  unapplied; no dataset, capture, BC, PPO, or training started. The corrected
  uniquely dynamically qualified set is now `35/70`.
- Evidence hashes: admission
  `5e4b74e1d32144bc2cee11769d1e208262db7be79cde8df8d8fc49e28da67411`,
  gate `76274cd5373c296910089d6c139ac3384e54ffa1e921673dcbf3f588f4e857e9`,
  log `aa0332b27a7636931cbecde3fdd225ff8725c69c01f2e131eaa8637dc4944f0f`,
  and summary
  `a5cf2bed55a1e4d2ba993df622bbbe5228c0369961a8f7d59fcbb2cddfad3fa4`.
  GPU and playback ownership were empty after closure.

## Next round after Round 63

- Inspect v14 case 31 on CPU and, if its frozen timing/path/transition and
  kinematic gates remain admitted, create one fresh default-profile case-only
  canary route with a duration-derived timeout.
- Preserve exact-source geometry, the v14 portfolio, LQR/controller settings,
  `1.8 m` camera-height ceiling, all physical gates, and learning closure.

## Round 64: case-31 paired lower-height plan passes Gate C

- CPU inspection confirmed that v14 case 31 is statically admitted and
  hash-bound to plan
  `8ebc938eeb53b8f7dbf4382a085d3667ea38d5ea52e535dc3be409767737aefb`.
  It shares case 30's source/execution clocks `18.144412/29.222488 s` and
  full `1.0 m/s` planned riser-rate limit, while its target camera-height
  range is lower at `1.168515-1.712510 m`. No static gate failed.
- Published the fresh case-only route at `d031b4f` with the unchanged default
  controller and a `1500 s` wall bound. The authoritative `.98` suite again
  passed `341/341` tests before exclusive launch. Ran only case 31 in
  `20260719_gate_c_smoothed_case31_v14_camera_lever_arm_v1_exclusive`.
- Both clocks completed in `11536` steps. Dynamic position p95/max passed at
  `0.136473/0.168665 m`; attitude p95/max passed at
  `0.152725/0.229012 deg`; pitch max was `6.982104 deg`. Riser servo p95/max
  were `0.011914/0.014094 m`; effort max was `32.296131 N`; thermal load max
  was `0.001202`; and every completion, physical, thermal, controller,
  runtime, IK, rate, saturation, and termination check passed.
- The prospective residual envelope passed with normalized maxima
  `[0.805935, 0.420867, 0.168425]`. Residual action stayed zero and unapplied;
  no dataset, capture, BC, PPO, or training started. The evidence-derived
  unique dynamic count is now `36/70`.
- Evidence hashes: admission
  `7d73e2bb2ea726b4048ac281869f4cf75c85152340c30cc443e9c5fce3bbfc17`,
  gate `a990fce3ebaff929cfb38b29ce1ac453caccab570151064c6646eaf65e2c10ac`,
  log `4b757ea3f68f7469e0bbd77b2035b829c5ddc51075fe7ce9e88340e70f8bac9d`,
  and summary
  `cdb4f0c12dfd5d10aeb7d45aae81b44e60c471649915e9849c32d8b2ca54f001`.
  GPU and playback ownership were empty after closure.

## Next round after Round 64

- Continue source-order CPU inspection with v14 case 32. Use the unchanged
  default profile first and authorize only one fresh case-specific canary after
  static admission and duration-derived timeout review.
- Keep residual capture, BC, PPO, and training closed. The current deterministic
  qualification is `36/70`, not corpus completion or teacher admission.

## Round 65: case-32 localized camera-lag reject and bounded recovery choice

- V14 case 32 was statically admitted with plan hash
  `45040c19379c0f56f68f44e6391033d2342769f3c034cc281d12f4e5f0cb35a1`,
  `1099` source anchors, source/execution clocks `21.648708/32.092669 s`,
  static position p95/max `0.129187/0.185137 m`, camera height
  `1.350000-1.486568 m`, and riser rate max `0.059092 m/s`.
- Published the default case-only route at `e541668`; the authoritative `.98`
  suite passed `341/341` tests before launch. Case 32 ran exclusively in
  `20260719_gate_c_smoothed_case32_v14_camera_lever_arm_v1_exclusive` and
  completed both clocks in `13447` steps.
- The only failed physical check was position p95: `0.171752 m` against the
  unchanged `0.15 m` gate. Position max passed at `0.192684 m`; attitude
  p95/max passed at `0.148761/0.221794 deg`; pitch max was `7.392641 deg`;
  no termination, IK, rate, saturation, thermal, controller, or runtime check
  failed. The residual envelope independently passed at normalized maxima
  `[0.862471, 0.400925, 0.137795]` but remained unapplied.
- The 1 Hz trace localizes all samples above `0.15 m` to wall seconds `20-24`
  and phase `9.882854-11.491540 s`. Camera correction is saturated, motion
  direction does not change, and position error rises to `0.192294 m` before
  recovering. This is a localized camera-lag hump, not reversal, terminal,
  balance, or riser behavior.
- The already-reviewed camera-error governor would affect exactly those five
  trace rows, capping phase progress from `0.359677-0.482954` to `0.20`. Its
  coarse additional-wall estimate is `5.513073 s`, below the existing
  `29.043008 s` runtime margin. This justifies one unchanged-plan retry; it is
  not proof of a pass and does not authorize broader tuning.
- Evidence hashes: admission
  `18a7161567b88a9b3ada2d91bb6e15ee45348625913f370904931e57953317bc`,
  gate `319a663352bebeb88a5b234a527854dba47f0274845c011462b5c5271294e2f7`,
  log `bb126c8809e250a753458cbe3f55772a14d7e63bc1b812035869f699b6d5b474`,
  and summary
  `2904537707061a2515b7fffacae0482d590c85be79301b9ae9a2603d10114047`.
  No dataset, capture, BC, PPO, or training started; the unique count remains
  `36/70`.

## Next round after Round 65

- Run one case-32-only retry with the frozen `0.13/0.155/0.20` camera-error
  governor, unchanged v14 plan, LQR, gains, `0.05 m` correction cap, horizon,
  and physical thresholds. Stop and seal either outcome.
- Do not advance to case 33 or any learning stage until the retry closes.

## Round 66: case-32 governor retry rejects slowdown as the recovery mechanism

- Published the bounded governor route at `797734e`; the authoritative `.98`
  CPU suite passed `341/341` tests before launch. Ran only case 32 in
  `20260719_gate_c_smoothed_case32_v14_camera_error_governor_v1_exclusive`
  with the unchanged v14 plan, LQR, gains, `0.05 m` correction cap, physical
  thresholds, and frozen `0.13/0.155/0.20` governor profile.
- The retry completed both clocks in `14222` steps and again failed only
  position p95. P95 worsened from `0.171752 m` to `0.177584 m`, while position
  max improved slightly from `0.192684 m` to `0.184828 m` and remained below
  its gate. Every completion, attitude, balance, rate, thermal, controller,
  runtime, IK, saturation, and termination check passed.
- The governor was active for `12.6213%` of policy steps, reached its frozen
  minimum phase scale `0.20`, and had mean scale `0.906603`. At its peak-error
  trace row, chassis XY error was only `0.028700 m`, but camera error remained
  `0.184264 m`; the camera-to-base lever-arm mismatch was about `0.188600 m`
  and the `0.05 m` correction was saturated. Slowing phase therefore prolonged
  the camera-error plateau instead of repairing the base/yaw allocation.
- The prospective residual envelope independently passed at normalized maxima
  `[0.862471, 0.399056, 0.137504]`. Residual action remained zero and
  unapplied; no dataset, capture, BC, PPO, or training started. The unique
  dynamically qualified count remains `36/70`.
- Evidence hashes: admission
  `1c0ec491cb5ceacf196bcb026067669e943dc8a2bb2c2af5c62765410570a492`,
  gate `06cd9d93239ca4c8858f54c5479ffd897d764f9353a7a4dd4f80a45aed01edfa`,
  log `b7ef7e9bcf102d0f0c28deb7c4697bc852c5ee90e79e7e6d97946466d9a8dfdf`,
  and summary
  `412ee7d0e42f09293abf7f561c57d1677dd3c6ae8a5dba6a2fb1080db9500e77`.
  GPU and playback ownership were empty after this case-32 run closed.

## Next round after Round 66

- Do not retry case 32 with another phase governor or a wider correction cap.
  Audit base/yaw allocation on CPU using a hash-bound explicit-preview
  candidate that preserves all source arrays and the parent smoothed camera
  geometry. Bind the derivation to both sealed case-32 rejects.
- The existing parent search contains no usable alternate: `0.10 m` lookahead
  is the only admitted attempt, while its `0.05 m` attempt has static position
  p95/max `0.679576/0.715013 m`. Evaluate a bounded preview/heading grid, admit
  at most one candidate through the unchanged duration, path, transition, and
  kinematic gates, and do not create a runtime route yet.
- Keep case 33, residual capture, BC, PPO, and all training closed.

## Round 67: dual-reject-bound case-32 preview candidate passes CPU gates

- Extended the explicit-preview derivation contract so one candidate can be
  hash-bound to a second corroborating dynamic reject. The new contract
  validates that both evidence pairs are completed position-p95-only rejects
  with runtime, thermal, and controller evidence intact, and that neither run
  opened residual capture, BC, or PPO. The authoritative `.98` suite passed
  `341/341` tests at pushed commit `6bc1ad8`.
- A bounded CPU sweep preserved the parent camera geometry and evaluated
  lookahead distances `0.11-0.25 m` at heading gain `2.75`, followed by gains
  `2.25-3.25` around the best `0.15/0.175 m` previews. The selected `0.175 m`,
  `2.75` candidate has the lowest static position p95 at `0.066916 m`; max is
  `0.109337 m`. The parent values were `0.129187/0.185137 m`.
- Derived the single candidate in
  `20260719_smoothed_case32_explicit_preview0175_dual_reject_cpu`. It is bound
  to default-reject gate/summary hashes
  `319a663352bebeb88a5b234a527854dba47f0274845c011462b5c5271294e2f7` /
  `2904537707061a2515b7fffacae0482d590c85be79301b9ae9a2603d10114047`
  and governor-reject gate/summary hashes
  `06cd9d93239ca4c8858f54c5479ffd897d764f9353a7a4dd4f80a45aed01edfa` /
  `412ee7d0e42f09293abf7f561c57d1677dd3c6ae8a5dba6a2fb1080db9500e77`.
- All `1099` source anchors, source timestamps, positions, attitudes, order,
  path length, initialization separation, and the parent smoothed camera
  geometry remain unchanged. Execution duration is `29.592866 s`, ratio
  `1.366958`; all path, duration, transition, rate, workspace, and kinematic
  gates pass. Candidate plan hash is
  `71b1986633613fdb13585ac4c12870addc553ad12e895b05cc424a83cf4e037f`;
  derivation manifest/summary hashes are
  `1af96b793cdae719a5d2a3b79cb77412fe5ee5b726d0d06e9db542d11e5eaa1b` /
  `2c58a71b6c880b162082ab7260b037eddfecb5c083de53dc4c67292cb07ef313`.
- Composed v15 at
  `20260719_smoothed_plan_all79_v15_case32_explicit_preview0175_cpu`. It keeps
  `70/79` static admissions and accepted duration-ratio median
  `1.499904 <= 1.5`. Manifest/summary hashes are
  `ef084a77e9f9fe633d8f6918d4e29808d7b339fe2e7db939c4aa826d597f1977` /
  `438adf783461c8954522cefe86fb0a3ea445f93481aced9a42d06b70a6c221a4`.
  Isaac, differential work, residual capture, BC, PPO, and training remain
  false; neither artifact is valid for training.

## Next round after Round 67

- Add one fresh, hash-bound case-32-only v15 route using the unchanged default
  camera-lever-arm controller, LQR, correction cap, and physical gates. Do not
  use the failed phase governor. Run it only after the shared `.98` ownership
  guard proves no other Isaac/GPU owner.
- Seal the first dynamic outcome. Advance to case 33 only if case 32 passes;
  otherwise stop for evidence-driven diagnosis. Keep all learning closed.

## Round 68: case-32 v15 canary quarantined after cross-session GPU overlap

- Published the v15 route at `36186c7`; `.98` passed `341/341` CPU tests.
  Case 32 started at `19:37` in
  `20260719_gate_c_smoothed_case32_v15_explicit_preview0175_v1_exclusive`.
  Startup completed, but the log immediately warned that another Kit process
  held the key-value database. At `19:40`, a separate
  `evaluate_cascade_robustness.py` process became visible while this canary was
  still live, invalidating exclusive GPU ownership.
- Stopped only the riser process tree and left the later robustness process
  untouched. No runtime gate JSON or labels were written. The fail-closed
  summary classifies case 32 as `missing_runtime_json`; it is not a physical
  or tracking reject and cannot update the `36/70` dynamic-pass count.
- Preserved interruption hashes: admission
  `9a51d6b2102357bec05f2ea29222b5d67180f45703b8ad85468c90cf553b6475`,
  partial log
  `3675bbbfa116c9b46ddcae5e8f44052438085373f0604121b550f403578f641c`,
  exit-code file
  `9d9b18720961e9b4689fd763b85e7b6f36160ccd3a8a1c9ddc5103bb0f66c396`,
  and summary
  `9b9235957bb16cd3cfa3d26986f9c0cbbb54efe03a880ac5f343a845179ff167`.
  Residual capture, BC, PPO, and training remain false.
- The old ownership check could miss Windows-side Isaac/Kit processes when
  they were absent from WSL `ps` and the WSL `nvidia-smi` compute list. The
  riser runner must additionally query Windows process command lines and
  reject Kit, playback, or robustness owners before any future launch.

## Next round after Round 68

- Add the Windows-process ownership check and a fresh v2 namespace, then run
  CPU tests. Do not launch while the robustness process remains active.
- After an explicit clean ownership read, run exactly one v2 case-32 canary.
  Treat the v1 namespace only as quarantined overlap evidence. Keep case 33 and
  all learning stages closed.

## Round 69: Windows guard validated; v2 later invalidated by shard 2

- Added a Windows `Win32_Process` ownership query and excluded the query's own
  PowerShell PID. Commit `e737c0c` passed `341/341` authoritative CPU tests.
  While robustness shard 1 was active, the v2 authorization correctly exited
  `5` before creating a namespace and reported both Windows wrapper/worker
  command lines. This validates pre-launch rejection for Windows-side owners.
- After a sustained clean window, v2 started at `21:10` in
  `20260719_gate_c_smoothed_case32_v15_explicit_preview0175_v2_exclusive`.
  It remained the only observed owner initially, but robustness shard 2 entered
  at `21:13` while the canary was still running. Stopped only the riser process
  tree; the robustness shard was not changed.
- No runtime gate JSON or labels were written. The fail-closed summary is a
  `missing_runtime_json` interruption, not a physical reject or dynamic pass.
  Evidence hashes: admission
  `dc91e4147364d118c0e561c3aea9b4e70dcbb70269bffb2234544c9489213685`,
  partial log
  `23a3a21fd0debd3a65f71803a293c75efaa4b487a4615f1d39877cf72d242112`,
  exit-code file
  `9d9b18720961e9b4689fd763b85e7b6f36160ccd3a8a1c9ddc5103bb0f66c396`,
  and summary
  `122d849957799623b6b25a1deed2a50c4cf9379545f86007357aa1ac8a5e2448`.
  The dynamic count remains `36/70`; learning remains closed.

## Next round after Round 69

- Do not race the remaining robustness queue. Prepare a fresh v3 namespace,
  but launch it only after shard 3 of 4 has been observed and then completed,
  followed by an empty Windows-owner query.
- Keep the v1/v2 namespaces quarantined as overlap evidence. Do not advance to
  case 33, residual capture, BC, PPO, or training first.

## Round 70: v3 confirms that idle snapshots are not an exclusive lease

- Synced exact pushed commit `c06a46f` to `.98` and passed the authoritative
  `341/341` CPU suite. Both WSL and Windows ownership queries were empty before
  v3 started in
  `20260719_gate_c_smoothed_case32_v15_explicit_preview0175_v3_exclusive`.
- A dedicated monitor then detected the separate minimal-controller process
  `candidate_speedki0p3_yaw10_5_provisional_full_v20` entering after v3 had
  started. This reproduces the same cross-session violation as v1/v2: a clean
  preflight snapshot cannot prevent a different task from claiming the GPU
  during the several-minute playback.
- Stopped only the riser process tree. No runtime gate JSON or labels were
  written. The normal runner exited during its bounded GPU-release wait because
  the later process remained active, so the preserved namespace was sealed
  CPU-side with the standard summarizer. Its classification is
  `missing_runtime_json`, not a physical reject or pass.
- Evidence hashes: admission
  `3525ddfd43698cf539f61530d6663dea9ef2bb3e7e216f9131b202e1945a433a`,
  partial log
  `c20da588672d81e754746f394d9ff3b9f5a7807767afbc7c4259d18379b72534`,
  exit-code file
  `9d9b18720961e9b4689fd763b85e7b6f36160ccd3a8a1c9ddc5103bb0f66c396`,
  and summary
  `7ff6aeec0443115ae2ecb92c201303ff4cfda17da12c9e26219c1044ee61e8c3`.
  Dynamic qualification remains `36/70`; residual capture, BC, PPO, and
  training remain closed.

## Resume condition after Round 70

- Do not issue a v4 runtime token based only on another momentary empty-owner
  read. The minimal-controller task in
  `/mnt/g/wSpace/cinebotRL-two-wheel-minimal` must explicitly release `.98` for
  the full riser playback window, or both tasks must adopt one shared atomic
  GPU lease before any new Isaac launch.
- Once that external condition is satisfied, create one fresh namespace bound
  to the unchanged v15 case-32 plan and rerun exactly once. Keep case 33 and all
  learning stages closed until a valid exclusive case-32 result exists.

## Round 71: resumed ownership audit permits one fresh v4 route

- On 2026-07-20, both the WSL process audit and the Windows `Win32_Process`
  query were empty for playback, robustness, training, and Kit owners. Local
  and remote riser branches remained clean; the remote retained only the
  previously documented preserved untracked evidence directories.
- Prepare exactly one fresh namespace,
  `20260720_gate_c_smoothed_case32_v15_explicit_preview0175_v4_exclusive`,
  bound to unchanged v15 manifest
  `ef084a77e9f9fe633d8f6918d4e29808d7b339fe2e7db939c4aa826d597f1977`
  and case-32 plan
  `71b1986633613fdb13585ac4c12870addc553ad12e895b05cc424a83cf4e037f`.
  The default camera-lever-arm profile, LQR, gains, `0.05 m` correction cap,
  timeouts, and physical thresholds remain unchanged.
- V4 is a runtime authorization only. It does not admit v1-v3 evidence, create
  labels, open residual capture, BC, PPO, or training, or advance case 33.

## Round 72: exclusive case-32 v4 dynamically qualifies the preview recovery

- Published the fresh v4 route at `f124970`; the authoritative `.98` suite
  passed `341/341` tests. Both WSL and Windows owner probes were empty before
  launch, and a dedicated five-second monitor observed the full playback close
  without any competing robustness process.
- Case 32 ran only in
  `20260720_gate_c_smoothed_case32_v15_explicit_preview0175_v4_exclusive`.
  Both source/execution clocks completed at `21.648708/29.592866 s` in `13037`
  steps. Dynamic position p95/max passed at `0.102419/0.133996 m`; attitude
  p95/max passed at `0.140728/0.219045 deg`; pitch p95/max were
  `6.341652/7.418519 deg`. Riser servo p95/max were
  `0.012923/0.013585 m`, effort max was `21.440975 N`, and thermal-load max
  was `0.001576`. Every physical, completion, runtime, thermal, controller,
  IK, rate, saturation, and termination check passed.
- The prospective residual envelope independently passed with raw maxima
  `[0.264953, 0.158985, 0.013684]` and normalized maxima
  `[0.883177, 0.397463, 0.136840]`. Residual action stayed exactly zero and
  unapplied; no dataset, capture, BC, PPO, or training started.
- A fresh union over all sealed dynamic gate rows proves `37/70` unique passes:
  `[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,30,31,32,52,53,66,67,68,70,74,77]`.
- Evidence hashes: admission
  `eb41fee8882cb1cffea276ca4fbca0cfd9be008dbf9f134e28eb2a2132bde9ea`,
  gate `da9003bbf089b28da8b19238d99efb4ad1beecc267e529b70a51221382fb5cf2`,
  log `d9974f4a4f19770995418493683d3e2b210334ea99f36630eeaee1898916abf2`,
  and summary
  `96dcb03e9ab46ba2e99a0adca2452689b5ae19468d067e9fa7b4dbf5e6bafa1a`.
  GPU and playback ownership were empty after closure.

## Next round after Round 72

- Inspect v15 case 33 on CPU. If its unchanged source, duration, path,
  transition, rate, workspace, and kinematic gates pass, add exactly one fresh
  default-profile case-only route with a duration-derived timeout.
- Keep residual capture, BC, PPO, and training closed until deterministic
  qualification and the final raw residual-envelope audit are complete.

## Round 73: case-33 CPU admission and bounded route

- V15 case 33 is statically admitted and hash-bound to plan
  `052b828587efef44e8f17bc6c8a4d73dcbfc2d35466ae02f5dd1a60f64af8d00`.
  It preserves all `1253` source anchors and has source/execution clocks
  `24.515451/36.622524 s`, ratio `1.493855`. Static position p95/max are
  `0.111536/0.137112 m`; camera height is `0.600000-1.637535 m`; planned riser
  rate max is `0.404418 m/s`.
- Every source-integrity, path, duration, transition, initialization, base/riser
  rate, proxy-rate, workspace, and kinematic check passes. Prepare exactly one
  default-profile route in
  `20260720_gate_c_smoothed_case33_v15_camera_lever_arm_v1_exclusive` with a
  `1600 s` wall bound. No controller or threshold changes are authorized.
- Case 33 remains invalid for training until its exclusive dynamic result is
  sealed. Residual capture, BC, PPO, and training remain closed.

## Round 74: exclusive case-33 playback passes Gate C

- Published the case-only route at `c1b3859`; the authoritative `.98` suite
  passed `341/341`. A corrected monitor ignored the runner's short-lived
  PowerShell ownership query and observed no actual robustness Python process
  during the full playback.
- Both clocks completed at `24.515451/36.622524 s` in `13727` steps. Dynamic
  position p95/max passed at `0.128966/0.166919 m`; attitude p95/max passed at
  `0.160480/0.244486 deg`; pitch p95/max were `6.205591/6.715392 deg`.
  Riser servo p95/max were `0.012256/0.012368 m`, effort max was
  `31.324106 N`, and thermal-load max was `0.001245`. All dynamic, runtime,
  thermal, controller, IK, rate, saturation, and termination checks passed.
- The residual envelope independently passed with raw maxima
  `[0.267019,0.165038,0.012629]` and normalized maxima
  `[0.890062,0.412594,0.126288]`; action remained zero and unapplied. No
  dataset, capture, BC, PPO, or training started. The unique dynamic union is
  now `38/70`.
- Evidence hashes: admission
  `44d28179a1cb9de572273f3b6d3b9f9eaed36ee7cbd4dcafa12fdb6853aef9a5`,
  gate `7abbcb0abe0f8bd760ceeabe3d3b1c9ba593a940551bcfe545a99177a2e205a8`,
  log `fe9b714927e5f8dcd79867aeebbc49609a05e021ccf313ced8c57e3f7e738b98`,
  and summary
  `2ecc682f51d22628eb09477ac856baa130e516611ceb45cb3b3e0278cfb88917`.

## Round 75: case-34 CPU admission and bounded route

- V15 case 34 is statically admitted with plan hash
  `e2b170f649f9e90542bfaa463c74fa802c0247273d7fad8c26f24922c212b9d4`,
  `1400` source anchors, source/execution clocks `27.384062/49.749541 s`,
  ratio `1.816733`, and static position p95/max `0.081825/0.142634 m`.
  Camera height is `1.344500-1.563395 m`; riser rate max is `0.114369 m/s`.
- Every source-integrity, path, duration, transition, initialization, rate,
  workspace, and kinematic gate passes. Prepare one unchanged default-profile
  route in `20260720_gate_c_smoothed_case34_v15_camera_lever_arm_v1_exclusive`
  with a `2000 s` wall bound. Learning remains closed.

## Round 76: exclusive case-34 playback passes Gate C

- Published the case-only route at `9e5d8f9`; the authoritative `.98` suite
  passed `341/341`. A minimal-controller robustness run owned the GPU at the
  first launch attempt, so the riser runner remained closed until that process
  exited. Case 34 then started through the unchanged Windows/WSL ownership
  guard with no competing playback or robustness process.
- Both source/execution clocks completed at `27.384062/49.749541 s` in `23778`
  steps. Dynamic position p95/max passed at `0.099618/0.158989 m`; attitude
  p95/max passed at `0.127229/0.221809 deg`; pitch p95/max were
  `6.139166/7.020725 deg`. Riser servo p95/max were
  `0.011750/0.012059 m`, effort max was `21.068913 N`, and thermal-load max
  was `0.001630`. All completion, physical, runtime, thermal, controller, IK,
  rate, saturation, and termination checks passed.
- The prospective residual-label envelope independently passed with raw maxima
  `[0.285401,0.143052,0.012274]` and normalized maxima
  `[0.951336,0.357630,0.122740]`. The residual action stayed exactly zero and
  was not applied; no dataset, capture, BC, PPO, or training started.
- The unique dynamically qualified union is now `39/70`:
  `[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,30,31,32,33,34,52,53,66,67,68,70,74,77]`.
- Evidence hashes: admission
  `7fd8e2e5077bbed64cfdd0e3d4c5cd4a5da51f695c5595156bc66aa2aece7f23`,
  gate `b7ad17005cd5257169abdbe4680331e626baad707dc967c605b693c93ab92032`,
  log `f623a95e6e366ba256e6717c784cfec19fe17c376720c60a79b49c33222a7628`,
  and summary
  `47f6b053108147121bafbe0e6ad8bc7cdaecbe017914034f1ca91473eee75633`.

## Next round after Round 76

- Preserve case 35 as the existing v15 static reject. Inspect v15 case 36 on
  CPU and authorize at most one fresh case-only Gate C route if all immutable
  source, timing, path, transition, workspace, and kinematic checks pass.
- Keep deterministic commands and all thresholds unchanged. Residual capture,
  BC, PPO, and training remain closed until the deterministic portfolio is
  qualified and the final raw residual-envelope audit is complete.

## Round 77: skip case-35 reject and admit case 36 on CPU

- Preserve v15 case 35 as a static reject: position p95/max are
  `0.288929/0.364426 m`, so it fails both immutable position gates despite
  passing source integrity, timing, path, transition, rate, attitude, and
  workspace checks. It is not routed to Isaac.
- V15 case 36 is statically admitted with plan hash
  `16b8d492571794b057a6747235ce37ce26173058c776773c2eaf717e38f1fe95`.
  It preserves all `797` source anchors and has source/execution clocks
  `15.694569/27.210186 s`, ratio `1.733733`. Path-length drift is `-2.754%`,
  static position p95/max are `0.121207/0.225160 m`, camera height is
  `1.425330-1.704093 m`, and planned riser rate max is `0.151173 m/s`.
- Every immutable source, path, duration, transition, initialization, rate,
  workspace, and kinematic check passes. Prepare exactly one default-profile
  route in `20260720_gate_c_smoothed_case36_v15_camera_lever_arm_v1_exclusive`
  with a `1500 s` wall bound. No controller, plan, threshold, or learning-stage
  change is authorized.

## Round 78: default case-36 playback is a position-p95-only reject

- Published the case-36 route at `0391190`; the authoritative `.98` suite
  passed `341/341`. Both clocks completed at `15.694569/27.210186 s` in
  `15362` steps with no termination. Position max passed at `0.210480 m`, but
  p95 failed at `0.193973 m`; this was the only failed dynamic check.
- Attitude p95/max passed at `0.134712/0.238552 deg`; pitch p95/max were
  `6.563231/6.664334 deg`. Riser servo p95/max were
  `0.011681/0.011932 m`, effort max was `22.072868 N`, and thermal-load max
  was `0.001282`. Completion, attitude, balance, riser, thermal, controller,
  IK, proxy-rate, saturation, and runtime gates all passed.
- The residual-label envelope independently passed with raw maxima
  `[0.131102,0.111716,0.012025]` and normalized maxima
  `[0.437007,0.279289,0.120250]`; action remained zero and unapplied. No
  dataset, capture, BC, PPO, or training started.
- Evidence hashes: admission
  `a1d2a95ab48ac26ca21c2f34e2b1255cb0199ab3e9b18a658f82fefbe24dbbc8`,
  gate `a271fa32e494faa7f000025ca4d73410d497681e43ebb499da892168e989befe`,
  log `148ad9d5174d25fcbcef61808a0c16bb0494690acf1da632cb0fcb69cfeb89fa`,
  and summary
  `51113c58b906e0ccd5b4fd893178f4bbdfb275ed5e84394c202b789805518344`.
  The dynamic union remains `39/70`; case 37 is not started.

## Round 79: bounded case-36 preview replacement restores static margin

- The failed p95 window is localized to execution phase `15.90-17.88 s`.
  Base XY error remains only `0.038-0.069 m`, while the physical camera
  lever-arm mismatch reaches about `0.24 m` during reorientation and the
  unchanged `0.05 m` correction cap saturates. This is a plan-allocation
  margin problem, not a balance, attitude, riser, thermal, or rate failure.
- A CPU-only grid over the unchanged parent smoothed geometry selected
  lookahead `0.55 m` and heading gain `1.25`. The hash-bound derived plan
  `d1e4da8ea73a26a8ac9f7b3d7063d2272569a7375f5ec8feed6e9a238a3c08ed`
  preserves all `797` source anchors, exact source arrays, camera geometry,
  endpoints, and source order. Source/execution clocks are
  `15.694569/26.525347 s`; static position p95/max improve to
  `0.111912/0.129099 m`. Every static gate passes without changing the
  controller, gains, correction cap, or thresholds.
- Compose v16 in
  `20260720_smoothed_plan_all79_v16_case36_explicit_preview055_g125_cpu`.
  Its manifest hash is
  `8351514a361d3be4e5fbf57f2dbb019a7d8d2f5b86e89cea2553a1cfda3c64a1`;
  it retains `70/79` CPU admissions and accepted-duration median `1.499904`.
  The derivation manifest hash is
  `46b2b1ca8e0ae24956390235ad764bd2152ffb864ee2f9568555b455e7e566d1`.
  All artifacts remain invalid for training.
- Prepare exactly one v16 case-36 retry in
  `20260720_gate_c_smoothed_case36_v16_explicit_preview055_g125_v1_exclusive`
  with the unchanged default runtime profile and `1500 s` wall bound.

## Round 80: v16 case-36 preview recovery passes Gate C

- Published the v16 recovery route at `ca7cea5`; the authoritative `.98`
  suite passed `341/341`. Both clocks completed at
  `15.694569/26.525347 s` in `14898` steps with no termination.
- Dynamic position p95/max passed at `0.130809/0.147473 m`; attitude p95/max
  passed at `0.135354/0.220562 deg`; pitch p95/max were
  `6.562593/6.608260 deg`. Riser servo p95/max were
  `0.011676/0.011750 m`, effort max was `22.072868 N`, and thermal-load max
  was `0.001277`. Every completion, physical, runtime, thermal, controller,
  IK, proxy-rate, saturation, and termination check passed.
- The prospective residual-label envelope independently passed with raw maxima
  `[0.131102,0.102489,0.011813]` and normalized maxima
  `[0.437007,0.256223,0.118127]`. Residual action remained zero and unapplied;
  no dataset, capture, BC, PPO, or training started.
- The unique dynamically qualified union is now `40/70`:
  `[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,30,31,32,33,34,36,52,53,66,67,68,70,74,77]`.
- Evidence hashes: admission
  `5cedb5d1d4251020c4b2059a0355af9f68ab69d41ad29f37806d1054a14d9293`,
  gate `46027d28ce2e1d443f6292be2ea5bf63a7e9223e7902aed3a01dc2029e22ada8`,
  log `ebae68627724ac3dd7cdce97c5ed71e6101974352fa44948382884e009c2e572`,
  and summary
  `11453bda0bc55eef0b750ce9bb68ee7aa2ac30a3b4420442e5b4c57d2e0c6302`.

## Next round after Round 80

- Inspect v16 case 37 on CPU. Route at most one fresh case-only canary if its
  source, timing, path, transition, workspace, and kinematic gates remain
  admitted. Do not revisit case 35 or launch statically rejected case 38.
- Keep residual capture, BC, PPO, and training closed pending completion of
  deterministic qualification and the final raw residual-envelope audit.

## Round 81: case-37 CPU admission and bounded route

- V16 case 37 is statically admitted with plan hash
  `3bc3119fa210f1fd190d7fba11b9571caa74dc1bc4de02fb98296ecc9e8d2c1e`.
  It preserves all `1458` exact-source anchors with unchanged geometry and has
  source/execution clocks `28.823874/46.347559 s`, ratio `1.607957`.
- Static position p95/max are `0.036262/0.059870 m`; camera height is
  `1.342686-1.666355 m`; planned riser rate max is `0.042155 m/s`. Every
  source, path, duration, transition, initialization, rate, workspace, and
  kinematic gate passes with substantial position margin.
- Prepare exactly one unchanged default-profile route in
  `20260720_gate_c_smoothed_case37_v16_camera_lever_arm_v1_exclusive` with a
  `2000 s` wall bound. Preserve case 38 as a static reject because its
  execution/source ratio is `2.190027`, above the immutable `2.0` limit.
  Learning remains closed.

## Round 82: exclusive case-37 playback passes Gate C

- Published the case-only route at `e511169`; the authoritative `.98` suite
  passed `341/341`. Both clocks completed at `28.823874/46.347559 s` in
  `22139` steps with no termination.
- Dynamic position p95/max passed at `0.104540/0.106517 m`; attitude p95/max
  passed at `0.126702/0.223847 deg`; pitch p95/max were
  `6.527206/6.789710 deg`. Riser servo p95/max were
  `0.013522/0.014111 m`, effort max was `21.501381 N`, and thermal-load max
  was `0.002026`. Every dynamic, runtime, thermal, controller, IK, rate,
  saturation, and termination check passed.
- The residual-label envelope independently passed with raw maxima
  `[0.220701,0.167322,0.014145]` and normalized maxima
  `[0.735670,0.418304,0.141450]`; action remained zero and unapplied. No
  dataset, capture, BC, PPO, or training started. The unique dynamic union is
  now `41/70`.
- Evidence hashes: admission
  `6ae2b65fd697c15491ac09c5807b51751dce9115caa0074aeabd463569b90fe5`,
  gate `4ae9d488f643568f43ddb31bd276301694fd6fb81c609ff2ed88f4d7b92fa7bf`,
  log `caaee0777f3b033c761cac08a3b857d0df1ffa3cd6c419bb000510431de31d6d`,
  and summary
  `0dc58d1b89b4aa6aa7bc75da2be5eeed78ffd9adf264a77f63e4e82e0b772a4f`.

## Round 83: skip cases 38-40 and admit case 41 on CPU

- Preserve v16 cases 38, 39, and 40 as static rejects because each exceeds the
  immutable `2.0` execution/source duration ratio. None is routed to Isaac.
- V16 case 41 is statically admitted with plan hash
  `cf3c1f35fbf20377c23dbc7ff3d24fbca8cdc9ef833cf1eff925d585295a4679`.
  It preserves all `2023` source anchors and has source/execution clocks
  `39.991917/70.739720 s`, ratio `1.768850`. Static position p95/max are
  `0.084243/0.123881 m`; camera height is `1.199417-1.500219 m`; planned
  riser rate max is `0.107360 m/s`. Every static gate passes.
- Prepare exactly one unchanged default-profile route in
  `20260720_gate_c_smoothed_case41_v16_camera_lever_arm_v1_exclusive` with a
  duration-derived `2800 s` wall bound. Learning remains closed.

## Round 84: exclusive case-41 playback passes Gate C

- Published the case-only route at `c587817`; the authoritative `.98` suite
  passed `341/341`. Both clocks completed at `39.991917/70.739720 s` in
  `25298` steps with no termination.
- Dynamic position p95/max passed at `0.085697/0.131343 m`; attitude p95/max
  passed at `0.157019/0.251114 deg`; pitch p95/max were
  `6.377008/6.415854 deg`. Riser servo p95/max were
  `0.013107/0.014853 m`, effort max was `25.502213 N`, and thermal-load max
  was `0.001688`. Every completion, physical, runtime, thermal, controller,
  IK, proxy-rate, saturation, and termination check passed.
- The prospective residual-label envelope independently passed with raw maxima
  `[0.109987,0.145519,0.014965]` and normalized maxima
  `[0.366623,0.363797,0.149654]`. Residual action remained zero and unapplied;
  no dataset, capture, BC, PPO, or training started. The unique dynamic union
  is now `42/70`.
- Evidence hashes: admission
  `24aaa9dfb0b9f1c686f61590f2e53f7b2a45a87229a757f791150a8da3b4e181`,
  gate `e4a8ad303aedba34c3c03b60e94f360b29d3f7b25c99ceb9ca824bf73d4ad593`,
  log `b54a37d78b7965eac2d3ecb3293cf7078f37848dc1021823f9124a92daf288ad`,
  and summary
  `61461418c83cb071ffce9300838140dd99ddff6fe333d834a3785b6ec895b016`.

## Next round after Round 84

- Review v16 case 42 on CPU before routing it. It is statically admitted, but
  its position p95 is `0.148363 m`, only `0.001637 m` below the immutable
  `0.15 m` gate; static max is `0.197051 m`. Preserve its exact source and
  determine whether a geometry-preserving preview allocation can provide
  meaningful static margin under the existing duration and transition gates.
- Do not launch case 42 merely because it narrowly passes static admission.
  Keep case 43 and all learning stages closed until this margin review is
  complete and one bounded case-42 route is explicitly selected.

## Round 85: case-42 CPU preview margin review

- Run a read-only CPU grid over the exact same unsmoothed case-42 camera path.
  The default `0.10 m / 2.75` preview has static position p95/max
  `0.148363/0.197051 m` and execution/source ratio `1.619662`; it is admitted
  but has only `0.001637 m` p95 margin.
- Every tested geometry-preserving preview from `0.125 m / 2.75` through
  `0.55 m / 1.25` passes all immutable static gates. The strongest bounded
  candidate is `0.55 m / 1.25`: static p95/max improve to
  `0.051280/0.081235 m`, duration ratio improves to `1.488203`, and maximum
  pre-densification base branch step is `0.020546 rad`.
- This read-only grid is diagnosis, not a deployment artifact. Do not route
  case 42 from an ad hoc result. The next bounded CPU task is to bind the
  selected candidate to the exact source, v16 parent manifest/plan, unchanged
  smoothed geometry, planner commit, and fresh output namespace under a
  proactive static-margin derivation contract. That contract must keep Isaac,
  residual capture, BC, PPO, and training closed before any route is added.

## Round 86: hash-bound case-42 margin derivation and v17 route

- Add the dedicated proactive static-margin derivation contract at `8e78e6c`.
  It admits only a closed CPU parent whose position p95 lies in
  `0.14-0.15 m`, requires at least `0.03 m` improvement in both p95 and max
  error, preserves exact-source arrays and parent smoothed geometry, and keeps
  Isaac, capture, BC, PPO, and training closed. The authoritative `.98` suite
  passes `342/342`.
- Derive case 42 in
  `20260720_smoothed_plan_case42_static_margin_preview055_g125_cpu`. The
  replacement plan hash is
  `b2057b0a23c9b5172f09a5373a020b18583cf118d55886624332f1d4e861a298`;
  source/execution clocks are `32.453601/48.297533 s`. Static p95/max improve
  from `0.148363/0.197051 m` to `0.051280/0.081235 m`; all immutable static
  gates pass. Derivation manifest hash is
  `b37e61c34a74e179ab095de04fb3da2b0cc94225e9b2a3cac094a00252bed20a`.
- Compose v17 in
  `20260720_smoothed_plan_all79_v17_case42_static_margin_preview055_g125_cpu`.
  It contains all 79 cases, retains 70 static admissions, and improves the
  accepted-duration median to `1.499110`. Its manifest hash is
  `57f6eab7453d0d1d3f2c244b7d429bb5ac1fa95184f63b4212770dc9fefb1a51`.
  All runtime and learning flags remain false.
- Prepare exactly one unchanged default-profile case-42 route in
  `20260720_gate_c_smoothed_case42_v17_static_margin_preview055_g125_v1_exclusive`
  with a `2000 s` wall bound. Case 43 and all learning stages remain closed.

## Round 87: v17 case-42 preview is a reversal-recovery dynamic reject

- Published the v17 case-only route at `a059fb0`; the authoritative `.98`
  suite passed `342/342`. Both clocks completed at
  `32.453601/48.297533 s` in `25779` steps with no termination.
- Dynamic position p95/max failed at `0.385169/0.496077 m`; these were the only
  physical quality failures. Attitude p95/max passed at
  `0.138777/0.217521 deg`; pitch p95/max were `6.479553/6.643589 deg`.
  Riser servo p95/max were `0.012583/0.013783 m`, effort max was
  `28.722269 N`, and thermal-load max was `0.001714`. Balance, attitude,
  riser, thermal, controller, IK, proxy-rate, saturation, completion, and
  runtime gates all passed.
- The label envelope independently failed: raw maxima were
  `[0.440000,0.149075,0.013813]`, normalized to
  `[1.466667,0.372687,0.138125]`. The prospective label was never applied;
  residual action stayed zero and no dataset, capture, BC, PPO, or training
  started. The dynamic union remains `42/70`.
- Evidence hashes: admission
  `1f62bb1bc6b9a689334dd4f48c332ab0e0b36bcb67171493a348a47b9903ff9e`,
  gate `aa7d535dbec839d25783c2e6e0b844818c1cebb0bcb85f57c9040700e9d8828c`,
  log `30f60edab6f21ef38241b541b1648f8f47ff85894e01a650236a0002bcf21e43`,
  and summary
  `10c36a40505643598d644ccdaaf5b0e42f56d457bba504e5e5b10a5edc6738f3`.

## Round 88: localized reversal retime produces v18 recovery route

- Trace localization shows the error grows during execution phase
  `3.47-4.77 s`: the feedforward direction reverses while the chassis remains
  about `0.52 m` behind, recovery commands stay at `-0.4 m/s`, and phase
  progress remains at the `0.10` minimum. This is a reversal-transition lag,
  not a camera-attitude, balance, riser, or thermal failure.
- Extend the evidence-bound retime contract at `4d5db5a` to accept exactly a
  completed p95+max rejection; the authoritative suite passes `343/343`.
  Locally retime immutable intervals `105-291` with peak scale `6.0`. The
  derived plan preserves all source arrays, targets, base/riser/proxy states,
  and geometry while reducing local acceleration and command-jump severity to
  `0.043914` and `0.329382` of the parent values.
- The replacement plan hash is
  `df694d8e6702ac0712ff7e1ce597c79ac30e3fc08f072caf63245fe8740e6669`;
  source/execution clocks are `32.453601/57.348774 s`, ratio `1.767100`.
  Derivation manifest hash is
  `308eeb7a27f4b53bcf97bfef1c0b0013a07e62243e22692a7305d10327be6872`.
- Compose v18 with 79 cases, 70 static admissions, and accepted-duration
  median `1.499904`. Its manifest hash is
  `b296a32f64a3e1f22b3a2fb51db2dd426baa2e1374d36cdf0eaaf5a5cdecd5a1`.
  Prepare exactly one retry in
  `20260720_gate_c_smoothed_case42_v18_localized_reversal_retime_v1_exclusive`
  with a `2300 s` wall bound. Learning remains closed.

## Round 89: v18 rejects and the lineage audit selects the v16 baseline

- Published the v18 route at `134aef5`; the authoritative `.98` suite passed
  `343/343`. The exclusive case-42 run exhausted its bounded simulated horizon
  at phase `37.191666/57.348774 s` after `34411` steps. Position p95/max failed
  at `5.074297/5.308876 m`; no termination occurred.
- All non-position physical gates remained healthy: attitude p95/max were
  `0.124800/0.218509 deg`, pitch p95/max were `7.330061/8.026940 deg`, riser
  servo p95/max were `0.010540/0.012393 m`, thermal-load max was `0.001429`,
  IK failures were zero, and action/riser/proxy saturation remained zero.
  The label envelope independently failed at normalized raw maxima
  `[1.417943,0.376901,0.126241]`; labels were not applied and no dataset,
  capture, BC, PPO, or training started.
- Sealed v18 hashes are admission
  `ecf8d6209389b45f8e8487a5ef2552de6e51de7f3906b89dcca900c4271cb978`,
  gate `e1621e47f56492129642a66bec306b51b3add9755737d9a299c116662a0e1309`,
  log `8ad1a97117bc07f6e5073bf14edad7dbf19b2cb40425cd487810db448043962f`,
  and summary
  `a466bf0da5777d4c35e7f1b3391d9c48b935246d85c1fde2cf6e91896b30908d`.
- Added the hash-bound lineage audit at `99c35c2`; the authoritative suite now
  passes `346/346`. Its real artifact SHA is
  `1ad420538b0d64b961b004c7c3e0bd952c3f2b9c5196ea4924c811daf5de2c8d`.
  It proves all authoritative source arrays are identical across v16/v17/v18,
  v18 only retimed the rejected v17 allocation, and v17 changed base/riser/proxy
  allocation by up to `0.673676 m`, `0.031866 m`, and `0.729410 rad` from v16.
- Select the pre-existing v16 case-42 plan, SHA
  `f737f0b2e1fe4877685ae2bc4a976c2179dce5ce8c30491146d14b3994eb4343`,
  as the sole next candidate. It already passes every duration, path,
  transition, and kinematic gate with static position p95/max
  `0.148363/0.197051 m`, but remains dynamically unvalidated and training
  ineligible.
- Prepare exactly one fail-closed rollback canary in
  `20260720_gate_c_smoothed_case42_v16_baseline_rollback_v1_exclusive` with a
  `2200 s` wall bound. No controller, threshold, source geometry, or learning
  stage changes are authorized; case 43 remains closed until case 42 passes.

## Round 90: v16 rollback completes but disproves the baseline allocation

- Published the hash-bound v16 rollback route at `b9f7d81`; the authoritative
  `.98` suite passed `346/346`, and WSL, Windows, and GPU ownership checks were
  empty before launch. The exclusive canary completed all
  `52.563872/52.563872 s` in `28578` steps with no termination.
- Dynamic position p95/max failed at `1.276124/1.468091 m`. The remaining
  physical gates passed: attitude p95/max were `0.136711/0.224623 deg`, pitch
  p95/max were `6.465981/7.005437 deg`, riser servo p95/max were
  `0.012703/0.014659 m`, thermal-load max was `0.001717`, IK failures were
  zero, and action/riser/proxy saturation remained zero.
- The residual label envelope independently failed at normalized maxima
  `[1.466667,0.297448,0.147136]`. Prospective labels were never applied,
  residual action remained zero, and no dataset, capture, BC, PPO, or training
  started. The dynamic union remains `42/70`.
- Sealed hashes are admission
  `7ecc155c0038753f9b5fc388f3e266f7c6e4c5180c0cba8f5b6a4088a1d4265d`,
  gate `b1b7ff22cfb7640b4a0eab83379388cb9412b901fa895ffa65bd16d9ca8772d8`,
  log `3f325f3acb5c55a06dd70c5056aced246eba967be15125cf7239ffadc50f4138`,
  and summary
  `dfc9ef8503229de3dccdd4fe98f56bb0b4e3c552efa835695949ebad9afcc59d`.
- Cross-run localization shows v16, v17, and v18 first exceed `0.15 m` at
  elapsed `1 s`; their dominant failure is the first reversal around phase
  `4.2-4.9 s`. V17 remains the best allocation at dynamic p95/max
  `0.385169/0.496077 m`; reverting its allocation or stretching its reversal
  is rejected.
- Stop GPU work. The next CPU-only candidate must use explicit initialization
  separation: add a bounded, unscored pre-roll that reaches the immutable
  first source anchor with compatible chassis velocity before source time
  starts. It must preserve all `1643` source anchors/timestamps, keep source
  and execution clocks separate, prove continuity and all existing kinematic
  limits, and remain dynamically unvalidated/training-ineligible. No runtime
  route may be added until that contract and focused negative tests pass.

## Round 91: explicit initialization schema and case-42 pre-roll candidate

- Added first-class optional initialization support at `d62c3bf`: a separate
  increasing clock plus `K x 7` state array ordered as base x/y/yaw, riser,
  and three continuous proxy axes. Non-empty initialization must end exactly
  at the first execution state; malformed, half-specified, or discontinuous
  initialization fails closed. Existing empty plans remain compatible.
- Added independent initialization interpolation and round-trip tests without
  shifting either source or execution time. The authoritative `.98` suite
  passed `349/349` after this schema change.
- Added the hash-bound smooth rest-to-first-derivative derivation at
  `53fabe4`. The first real attempt produced a valid plan but failed before
  manifest creation because a NumPy boolean was not JSON serializable. The
  partial namespace is preserved as
  `20260720_smoothed_plan_case42_v17_initialization_preroll2s_cpu_FAILED_JSON_SERIALIZATION`.
  The serialization regression was fixed at `5d3d87e`; the authoritative
  suite passes `352/352`.
- Generated the clean CPU-only candidate in
  `20260720_smoothed_plan_case42_v17_initialization_preroll2s_cpu`. It contains
  `401` pre-roll samples over `2.0 s`, starts within `1.6e-6` state units per
  second of rest, and matches the first execution derivative within
  `6.3e-6` relative error.
- Initialization maxima are base linear `0.079664 m/s`, lateral
  `0.007871 m/s`, yaw `0.218336 rad/s`, riser `0.066576 m/s`, and proxy
  `0.248524 rad/s`; every existing rate bound passes. The candidate plan SHA
  is `ea2e54273c42efa3980eaa3ea9b161109702047467df131d4ad1d2604f063984`;
  manifest SHA is
  `6bb3f04b802761fcae0675430dd820819180296c73f86ddfd83b19b2288f26ce`.
- All scored execution arrays, authoritative source arrays, and both source
  and execution clocks are byte-identical to v17. No controller, threshold,
  runtime route, Isaac process, dataset, capture, BC, PPO, or training was
  opened. The candidate remains dynamically unvalidated and invalid for
  training.

## Next round after Round 91

- Add fail-closed playback support that executes the separate pre-roll before
  source phase zero, excludes it from source tracking metrics, preserves LQR
  state continuity into the scored run, and reports initialization completion
  and terminal state/rate evidence independently.
- Add focused CPU tests proving source phase and metric arrays remain empty
  during initialization, malformed initialization cannot launch, and plans
  without initialization retain their current behavior.
- Do not add a GPU authorization token or runtime namespace until the runner
  contract and authoritative CPU suite pass. Case 43, residual capture, BC,
  PPO, and training remain closed.

## Round 92: initialization playback and evidence contracts are implemented

- Added initialization execution and evidence support at `be0e874`. The runner
  executes the separate `2.0 s` pre-roll before source phase zero, retains the
  LQR/controller state into scored execution, and excludes initialization from
  source tracking metrics, residual-label observations, and datasets.
- Initialization now has independent completion, terminal error, saturation,
  effort, thermal, and sample-count evidence. Summary admission fails closed
  if required initialization evidence is absent or if any source/residual
  sample leaks into the pre-roll. Plans without initialization retain their
  historical evidence contract.
- Added composable replacement support at `83212c2`. The first composed v19
  portfolio reached `70/79`, but audit found that its case-42 row inherited the
  obsolete parent assertion `initialization_separate_empty=true`. V19 is
  superseded and must not be used as runtime evidence even though its NPZ data
  are valid and training remained closed.
- Replaced that stale assertion at `5a66e3d` with explicit evidence that the
  parent initialization was empty, the new initialization uses a separate
  clock, remains unscored, and passes its kinematic gate. The composer now
  rejects mutated source/scored arrays, advanced source/execution clocks,
  malformed pre-roll metadata, failed initialization checks, and stale empty
  assertions. Focused tests pass `9/9`; the authoritative `.98` suite passes
  `360/360`.

## Round 93: corrected case-42 pre-roll portfolio passes full CPU audit

- Regenerated the corrected case-42 derivation in
  `20260720_smoothed_plan_case42_v17_initialization_preroll2s_v3_cpu`.
  The plan SHA remains
  `ea2e54273c42efa3980eaa3ea9b161109702047467df131d4ad1d2604f063984`;
  manifest and summary SHAs are
  `44854a904f494014b44dfdc911d7bbfbf6406a542ba9600aa4c07a0e8f06b821`
  and `03cdeeb8fb1cb3e536b6a04add31485ecf862319db86a8a2240e7a709a50e5ca`.
- Composed the corrected all-79 CPU portfolio in
  `20260720_smoothed_plan_all79_v20_case42_initialization_preroll2s_cpu`.
  It admits `70/79`; honest rejects remain
  `[1,27,29,35,38,39,40,45,71]`, and accepted-duration median is
  `1.499110`, within the unchanged `1.5` gate. Manifest and summary SHAs are
  `3d7f9650a4f701f80a11948364a53ecd34641160bffb6bc3ed697d038d559b72`
  and `4546f29d35b4b6a7a69c38e10122f80b8ff00a1c32eb50fa79b8878b7215155c`.
- Read-only audit verified all `79` NPZ hashes, all `79` per-case JSON rows,
  and `78` byte-identical parent plans. Case 42 is byte-identical to the v3
  replacement and differs from its v17 parent only in `metadata_json`,
  `initialization_time_s`, and `initialization_state`; every scored/source
  array and both clocks remain identical. Case 42 is the sole plan with a
  non-empty initialization (`401 x 7`).
- No Isaac process, runtime namespace, authorization token, residual capture,
  dataset, BC, PPO, or training was opened. V20 remains dynamically
  unvalidated and invalid for training.

## Next round after Round 93

- Add one hash-bound, fail-closed case-42-only Gate C authorization for the v20
  portfolio and the existing default controller profile. Pin commit `5a66e3d`,
  v20 manifest SHA, case-42 plan SHA, source/gains/USD identities, exclusive
  ownership, and a fresh namespace.
- Run the focused route tests and the full authoritative CPU suite before any
  launch. Then verify WSL, Windows, and GPU ownership immediately before one
  bounded case-42 canary. Initialization must complete independently before
  scored phase zero; any initialization or dynamic rejection stops the route.
- Do not launch case 43, residual capture, BC, PPO, training, or differential
  work. A case-42 dynamic pass would move the demonstrated union from `42/70`
  to `43/70`; a reject must be sealed and diagnosed CPU-first.

## Round 94: initialization passes, but reversal tracking still rejects case 42

- Added the v20 case-42-only route at `0d7d540`; the authoritative suite
  passed `360/360`, and HEAD/upstream, tracked state, fresh namespace, WSL,
  Windows, and GPU ownership checks were clean before launch.
- The initialization contract passed independently: exactly `400` policy
  steps over `2.0 s`, completion true, zero source-metric and residual-label
  samples, zero action saturation, `400` thermal samples, thermal-load max
  `0.000101`, and riser effort max `19.872696 N`. Terminal base position/yaw,
  riser, and proxy errors were `0.156443 m`, `1.877245 deg`, `0.011067 m`, and
  `0.075022 deg`; all are finite evidence and the existing initialization
  gates pass.
- The scored reference completed all `48.297533 s` in `25772` steps with no
  termination. Dynamic position p95/max still fail at
  `0.407855/0.520446 m`; thermal and controller evidence pass, and the only
  physical quality failures are position p95/max. Residual-label admission
  independently fails, labels were not applied, residual action remained zero,
  and no dataset, capture, BC, PPO, or training started.
- The pre-roll solved the startup symptom but not the structural reversal:
  first trace error above `0.15 m` moved from elapsed `1 s` in v17 to `13 s`
  in v20, and the first-five-second trace mean improved from `0.056194 m` to
  `0.036496 m`. The dominant error still grows over phase `3.5-4.9 s`; peak
  trace error is `0.519999 m` at phase `4.202720 s` while the requested
  velocity remains `-0.4 m/s` and the measured chassis displacement reverses.
- Sealed hashes are admission
  `d12eba1967868b0557342fec27855aee6a5e89593a4ad5db1fe7f69225bf411f`,
  gate `c00323ebc21045d7fee56bb9a37538fb86553fcf545ddd0e1874789a954da025`,
  log `776821772ca9b37e18bf944d84486c3ac6737cb70fd0574de9eb9bb392c3b1d6`,
  and summary
  `07f0fb0f94d4961e393e329a6833cf0cc4398eb69446edb601b5858d1ceba357`.
- The physical process exited zero and wrote the complete gate. The wrapper's
  post-run checker then raised `IndexError` because its embedded Python was not
  passed the newly required fifth argument; the rejection summarizer still
  produced a valid runtime-contract pass. Fixed the wrapper at `502b2be`, added
  regression coverage, and retained `360/360` authoritative tests. Do not
  rerun v20 merely to repair wrapper presentation.

## Next round after Round 94

- Stay CPU-only and inspect the velocity-feedback boundary. The outer cascade
  currently computes longitudinal error from wheel-derived velocity while the
  quality gate scores axle/root displacement. The v20 trace shows requested
  `-0.4 m/s` while finite-difference root motion slows, stops, and reverses;
  this is now the leading structural hypothesis.
- Add policy-rate aggregate and existing 1 Hz trace evidence for root-frame
  longitudinal velocity, wheel-derived longitudinal velocity, their mismatch,
  effective/governed velocity reference, pitch reference/bias, and common wheel
  action. Do not change commands in that evidence patch.
- Use CPU tests and the sealed v20 trace to define one bounded optional
  root-velocity outer-loop candidate while retaining wheel velocity in the
  frozen inner LQR. Do not authorize another GPU canary until the evidence
  contract, backward-compatible default behavior, and synthetic sign/reversal
  tests pass. Case 43 and all learning stages remain closed.

## Round 95: root-velocity feedback is measured and rejected as the case-42 fix

- Added policy-rate root-versus-wheel velocity evidence at `05c63dc`. The
  evidence compares root-link longitudinal velocity, wheel-derived velocity,
  their mismatch, both reference errors, direction disagreement, pitch
  reference/bias, and common wheel action without changing commands. Added the
  optional outer-loop-only root-velocity selector at `95a6303`; the wheel state
  remains in the frozen inner LQR and the default selector preserves legacy
  behavior. The authoritative CPU suite passed `365/365`.
- Added the case-42-only root-velocity route at `b6f6c64`, with the same v20
  plan SHA `ea2e54273c42efa3980eaa3ea9b161109702047467df131d4ad1d2604f063984`,
  v20 manifest SHA
  `3d7f9650a4f701f80a11948364a53ecd34641160bffb6bc3ed697d038d559b72`,
  source, gains, USD, controller, position gates, and `2.0 s` initialization.
  WSL, Windows, and GPU ownership checks were empty before the exclusive run.
- The scored reference completed all `48.297533 s` in `25787` steps with no
  termination. Dynamic position p95/max still fail at
  `0.407546/0.519863 m`; all other physical checks, initialization, thermal,
  controller, runtime, and telemetry contracts pass. This is only
  `0.000309/0.000583 m` better than wheel-feedback v20 and remains worse than
  v17 at `0.385169/0.496077 m`.
- Root/wheel mismatch is only `0.020294 m/s` RMS and `0.110495 m/s` maximum.
  Root and wheel reference-error RMS are `0.159325/0.171710 m/s`, opposite
  direction occurs for `1.6675%` of policy samples, and false wheel tracking is
  exactly zero. The selected feedback source is `root_link_vx`, schema
  `riser_root_vs_wheel_velocity_policy_rate_v1`, and its `25787` samples equal
  completed steps. Root-vs-wheel estimation is therefore not the structural
  case-42 bottleneck.
- The separate initialization again passes: `400` steps over `2.0 s`, zero
  source/residual samples and saturation, thermal-load max `0.000102`, and
  terminal base/yaw/riser/proxy errors `0.158063 m`, `1.815077 deg`,
  `0.010919 m`, and `0.072426 deg`.
- Residual-label admission independently fails at raw maxima
  `[0.440000,0.150003,0.013778]`, normalized to
  `[1.466667,0.375007,0.137775]`. Labels were not applied, residual action
  stayed zero, and no dataset, capture, BC, PPO, or training started. The
  dynamic union remains `42/70`.
- Sealed hashes are admission
  `21728d274ce0bf8752bd10b3fd7d2fad9a819c5d65f1bb37f59791ea60f6b837`,
  gate `d2df0ea14b1608fa8acdc118c2d974a4fe4094b1beda1763e23d4788e111f936`,
  log `cdee0b62ab7e91369b117cc7de8d09b1d36a028ad0c4a6a8a1bb7c5c29776fe5`,
  exit-code evidence
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`,
  and summary
  `1f0228634cf31ee8379539d93c0210124f9b29273e81a1b4e111fa34a9970f0d`.

## Next round after Round 95

- Stay CPU-only. The sealed trace localizes the dominant reversal at phase
  `3.5-4.9 s`: progress is already pinned at `0.10`, position error reaches
  `0.519485 m` at phase `4.199624 s`, and the controller continues commanding
  approximately `-0.34 m/s` while the body needs several seconds to reverse.
  The moving target, not velocity-observer disagreement, dominates the error.
- Add an optional, fail-closed zero-progress recovery mode which permits the
  existing position governor to hold the current immutable target only at its
  full-error boundary. Keep the default `0.10` minimum unchanged, preserve all
  source/plan arrays and controller gains, and retain the existing completion,
  timeout, safety, thermal, and quality gates so a non-converging hold remains
  a rejection.
- Add telemetry and negative/default-compatibility tests before considering any
  runtime route. Do not create a namespace or authorization token, launch
  Isaac, case 43, residual capture, BC, PPO, or training in the CPU contract
  change.

## Round 96: exact phase hold is dynamically rejected

- Added the default-off exact-hold contract at `a19a5ba`, its focused test fix
  at `0d401c9`, and the hash-bound case-42 route at `44b793e`. The authoritative
  CPU suite passed `374/374` before launch. The route reused the v20 portfolio
  manifest and case-42 plan without changing source anchors, source/execution
  clocks, controller gains, safety gates, or quality thresholds.
- The exclusive zero-progress canary initialized successfully for `2.0 s` and
  `400` steps, then exhausted its bounded `144.9 s` scored horizon at phase
  `3.503471/48.297533 s`. One hold segment occupied `25,896/28,980` scored
  steps (`89.3582%`). Position p95/max failed at `1.306713/1.372330 m`; the
  only dynamic failures were completion and the two position gates.
- Balance and the independent subsystems remained bounded: attitude p95/max
  were `0.148136/2.692105 deg`, pitch p95/max were
  `5.243685/6.539545 deg`, riser p95/max were `0.009672/0.010626 m`, thermal
  load max was `0.001422`, and action saturation was zero. The residual-label
  envelope independently failed, labels were never applied, and no dataset,
  capture, BC, PPO, or training started.
- During the hold, the target remained fixed near base XY `(1.157,-0.167) m`
  and the requested velocity stayed negative, but wheel velocity converged to
  approximately `+0.093 m/s`. The actual base moved monotonically away from
  the target. Exact phase hold is therefore structurally rejected and must not
  be retried unchanged.
- Sealed hashes are admission
  `60025a79bb0ac1e6ee14b11e7f7ec8cc3b8753d000cc19c0ca2e7cb75f31a2d0`,
  gate `206be73921a900fb24cc594e23e4909f87e64e05e4234a462558a47a32e97d8b`,
  log `cf9cf84f0575a4893923be073a0614ce2ed71cfd4f9f53b8aa827de1512896d2`,
  exit-code evidence
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`,
  and summary
  `8273fe58fb818ec0fc33fc0f947c3189ada18e6dd462b81607b057b778824716`.

## Round 97: longitudinal cancellation diagnosis and bounded recovery cap

- A CPU readback of all `129` one-hertz hold samples decomposed the frozen
  common action using the exact selected gain. Mean pitch-error contribution
  was `+0.536747`, mean wheel-velocity contribution was only `-0.000874`, and
  the remaining pitch-rate contribution was `-0.481058`. The observed common
  action averaged `+0.054815`. The inferred sampled pitch rate was approximately
  `-0.340454 rad/s` while sampled pitch stayed near `+3.4 deg`, exposing a
  policy-rate longitudinal limit cycle aliased by the one-hertz trace.
- The signed plant contract is internally consistent: positive common effort
  drives both wheels in `+X`, and the identified input matrix maps positive
  common effort to negative pitch acceleration and positive wheel acceleration.
  The case-42 defect is not a simple wheel-sign inversion, and the root-versus-
  wheel observer hypothesis remains rejected.
- The promoted deterministic robustness evidence only qualified longitudinal
  commands at `+/-0.2 m/s`. Case 42 held a `-0.4 m/s` request continuously,
  outside that demonstrated envelope. Commit `2ed23fd` therefore adds a
  default-off symmetric tracking-command cap that may reduce but cannot expand
  the existing `0.4 m/s` limit, exposes exact per-state LQR action
  contributions, and rejects non-finite/non-positive limits. It does not change
  default commands, LQR gains, plans, gates, or residual behavior.
- The intended bounded candidate combines the already rejected exact phase
  hold with a `0.2 m/s` cap. This tests whether the controller can converge at
  its qualified longitudinal envelope; it does not claim that the candidate
  passes dynamically. Focused local tests pass `6/6`; the authoritative `.98`
  CPU suite passes `379/379`. The transfer bundle SHA is
  `bbab26ba8094a2d4b7f994143fea36111fced563e67ddde243c114194bc762a8`.
- The control hierarchy remains fixed: deterministic LQR owns primary balance,
  deterministic gates own hard safety and limits, and any future learned layer
  may only provide bounded supervisory residuals above that stack. No runtime
  authorization, namespace, Isaac process, case 43, residual capture, BC, PPO,
  or training is opened by this CPU commit.

## Next round after Round 97

- Add one fresh fail-closed case-42 route that pins `2ed23fd`, the unchanged v20
  portfolio and plan identities, the `0.2 m/s` command cap, exact phase hold,
  gains, USD, source, thresholds, ownership, and no-learning invariants.
- Require the output and summarizer to prove both the cap and hold contracts,
  preserve the existing dynamic/thermal/controller/runtime/label outcome split,
  and reject missing or conflicting environment overrides before Isaac starts.
- Run focused route tests and the full authoritative CPU suite. Only then may
  one exclusive bounded case-42 canary be considered. Case 43, residual capture,
  BC, PPO, training, and obstacle work remain closed.

## Round 98: the bounded `0.2 m/s` recovery cap is dynamically rejected

- Commit `b7f01e3` bound the unchanged v20 case-42 plan to one fresh exclusive
  route with exact phase hold and a `0.2 m/s` symmetric tracking-command cap.
  The authoritative preflight suite passed `380/380`; source anchors, both
  clocks, gains, USD, thresholds, and no-learning invariants remained pinned.
- The runner closed fail-closed after the full bounded horizon. Initialization
  completed for `2.0 s` and `400` steps. Scored execution reached only
  `3.208039/48.297533 s` in `28,980` steps; one exact hold occupied
  `25,180` steps (`86.8875%`). Position p95/max failed at
  `11.776149/12.494410 m`, while completion also failed.
- The cap contract itself passed: the requested base velocity was bounded to
  `0.2 m/s` and the effective controller reference reached only
  `0.171485 m/s`. Nevertheless, the robot converged to approximately
  `+0.099 m/s` body-forward motion while the route correctly required reverse
  motion. At yaw near `174 deg`, this drove the robot monotonically away from
  the frozen target to more than `12 m` error. The velocity cap hypothesis is
  therefore falsified and must not be retried unchanged.
- Independent subsystem outcomes stayed separate. Pitch max was
  `4.560527 deg`, attitude max `1.662295 deg`, action saturation zero, riser
  thermal admission passed, internal attitude IK had zero failures, and no
  termination occurred. The prospective residual-label envelope passed, but
  admission remained false; residual action stayed exactly zero and no dataset,
  capture, BC, PPO, or training started.
- Sealed hashes are admission
  `99e7ee3faf60b710fb87df6185d6c58b7af04ae2b8d61efec2484a8f80214de6`,
  gate `76c4a506d3e48b21132e1514cf7a7796803f3406682a764b0c4ea5b38ada22bd`,
  log `10ef2b37b6a7f5e8300c216501b3315b4857f8e435789df7170e08f2b80182bd`,
  and summary
  `952d89ed6e8c7aec874fdda2f2c78e606276d4e86cb937793e07398d84cb3c86`.

## Round 99: CPU diagnosis isolates asymmetric physical pitch authority

- Case 42's path sign is correct. At the held target and yaw near `174 deg`,
  negative body velocity is required to move toward increasing world X. The
  path layer continues to request `-0.2 m/s`; neither source retiming nor the
  motion-direction rule explains the opposite steady motion.
- The riser plant carries an observed equilibrium-pitch bias of approximately
  `+1.65 deg`. The current cascade clips the velocity-generated pitch offset to
  `+/-6 deg` before adding that bias. Its physical pitch target therefore has
  asymmetric bounds of approximately `-4.35/+7.65 deg`. This differs from the
  earlier `+/-0.2 m/s` proof, whose simplified chassis had approximately zero
  equilibrium offset, and explains why reducing the velocity command did not
  restore the previously demonstrated authority.
- A default-off CPU candidate applies the existing `+/-6 deg` limit to the
  total physical pitch target, then derives the velocity correction around the
  measured equilibrium bias. With `+1.65 deg` bias, saturated reverse/forward
  targets become exactly `-6/+6 deg`; the corresponding velocity corrections
  are `-7.65/+4.35 deg`. This does not relax the physical pitch limit, LQR gain,
  action limit, plan, or any dynamic/quality threshold.
- Evidence now distinguishes the velocity pitch correction from the total
  physical pitch target and aggregates the latter at policy rate. The option is
  disabled by default and zero-bias behavior is bitwise-compatible in focused
  tests. The local focused controller/evidence/playback suite passes `56/56`.
  The Mac cannot collect the full suite because `gymnasium` is absent. Commit
  `2889727` was pushed and transferred to `.98` with bundle SHA
  `4233550056df451efcea9423fbf06186a408d1893b98c88a299485dc65cf1040`;
  the authoritative `.98` suite passes `382/382` in `26.60 s`.

## Next round after Round 99

- Request review of the physical-target limiting contract before considering
  one bounded case-42 canary. Case 43, residual capture, BC, PPO, training, and
  obstacle work remain closed.
- If approved, add a separate fail-closed route commit that pins the exact
  controller, plan, gain, USD, source, total-target limit, cap, hold, thresholds,
  ownership, and no-learning identities. Do not combine authorization with the
  reviewed CPU candidate commit.

## Round 100: symmetric physical-pitch canary contract is authorized

- Independent CPU review added a nonzero-bias compatibility regression at
  `5c05865`. It proves the default-off path retains the legacy physical target
  range of approximately `-4.35/+7.65 deg` under a `+1.65 deg` equilibrium
  bias, while only the candidate produces a symmetric `-6/+6 deg` physical
  target. The authoritative `.98` suite then passed `383/383`.
- Separate commit `68500c6` adds one unique case-42 authorization token and
  namespace. It pins the unchanged v20 portfolio manifest
  `3d7f9650a4f701f80a11948364a53ecd34641160bffb6bc3ed697d038d559b72`,
  plan `ea2e54273c42efa3980eaa3ea9b161109702047467df131d4ad1d2604f063984`,
  planner `5a66e3deef01fceacc80fee37b199045705d7f02`, the existing `0.2 m/s`
  cap, exact phase hold, initialization, gains, USD, source, runtime hashes,
  ownership, thresholds, and no-learning invariants.
- Runtime evidence must independently prove the total physical pitch target
  reaches but never exceeds `6 deg`, while the velocity-generated correction
  exceeds `6 deg` to compensate the nonzero equilibrium bias. Missing, forged,
  inactive, or mismatched evidence fails admission without changing commands or
  relaxing any physical/quality gate.
- Focused route/evidence tests pass `98/98`; the authoritative `.98` suite
  passes `384/384` in `27.29 s`. Transfer bundle SHA is
  `84a55abf0bcceaee3d575d44577e054e60f95a77b8e81f4626cbe68e5ba1fac6`.
  No Isaac process, namespace, residual capture, BC, PPO, or training was
  started by these CPU commits.

## Next round after Round 100

- Seal the documentation commit, verify local and `.98` HEAD equal their pushed
  upstream, confirm the new namespace is absent, and prove WSL playback,
  Windows Kit/playback ownership, and NVIDIA compute ownership are all empty.
- If and only if preflight is clean, launch the single authorized case-42
  canary. Do not launch case 43 or any parallel GPU work.
- On closure, hash and audit admission, gate, log, exit code, and summary. Treat
  dynamic quality, thermal admission, runtime contract, total-pitch evidence,
  and residual-label envelope as independent outcomes; stop after the first
  result with no automatic retry.

## Round 101: symmetric physical-pitch authority repairs case 42 but does not yet admit it

- The one authorized exclusive canary ran from clean pushed commit `7dfd503`
  in namespace
  `20260720_gate_c_smoothed_case42_v20_zero_progress_hold_cap020_total_pitch_limit_v1_exclusive`.
  It reused the unchanged v20 portfolio and case-42 plan, retained the exact
  phase-hold candidate and `0.2 m/s` command cap, enabled the default-off total
  physical pitch target limit at `+/-6 deg`, and wrote no learned action or
  dataset.
- The runner closed fail-closed with execution phase
  `43.916637/48.297533 s` after `28,980` scored steps. Dynamic quality remains
  false because `completed_reference` failed and position p95 was
  `0.167272 m` against the unchanged `0.150 m` gate. Position max now passes
  at `0.192288 m`; no safety termination occurred.
- This is a structural improvement over the otherwise identical capped
  exact-hold canary: position p95/max fell from `11.776149/12.494410 m` to
  `0.167272/0.192288 m`, and completed phase increased from
  `3.208039/48.297533 s` to `43.916637/48.297533 s`. Case 42 is still an
  honest reject and is not added to the dynamic union.
- The independent contracts passed. Total physical pitch target aggregate max
  was exactly `0.104720 rad` (`6 deg`), while velocity-generated pitch offset
  reached `0.140887 rad` (about `8.07 deg`), proving nonzero-bias compensation
  was active without widening the physical target. Runtime, thermal,
  controller, total-pitch, and residual-label-envelope evidence passed;
  action saturation was zero, thermal load max was `0.001513`, and internal
  attitude IK had zero failures.
- Balance and auxiliary tracking remained bounded: attitude p95/max were
  `0.131599/1.745864 deg`, pitch p95/max were `5.694907/6.784310 deg`, and
  riser p95/max were `0.011161/0.013356 m`. Effective base velocity reference
  max was `0.174813 m/s`, and root-versus-wheel velocity mismatch RMS was
  `0.003971 m/s`.
- Exact zero-progress holds were rare: `35` scored steps in three segments,
  while mean progress scale was `0.303082`. The dominant p95 failure is a
  localized mid-route XY lag over phase approximately `24.17-26.69 s`, not a
  terminal divergence. At the one-hertz peak near phase `24.7838 s`, position
  error was about `0.191925 m`, primarily world-X lag, while yaw and vertical
  tracking remained healthy. The final sampled error recovered to about
  `0.11165 m`; the run exhausted the frozen `3x` horizon with approximately
  `4.381 s` of execution phase still unconsumed.
- Residual-label admission remained false despite its independent envelope
  pass. Residual action stayed zero, and no dataset, residual capture, BC, PPO,
  training, case 43, or obstacle work started. The dynamic union remains
  `42/70`.
- Sealed hashes are admission
  `c0d52639321ba58ecbe9c619a29a5dfe4559cc81827d22799be6c8a622fcbd4b`,
  gate `9b3d50b5f92ab4592fbbb7a91d2b8761beb530a80d10e302f9a5f2af1ba2873a`,
  log `2362f9a41468f81b091a116607aa1e41dac45c69b52373c00d2b6793a465bb2c`,
  exit-code evidence
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`,
  and summary
  `de00d91f8a1cfb252fbd2a83eeec6c6c996158bdf400a3c498dd7da213780579`.

## Next round after Round 101

- Stay CPU-only. Quantify execution-clock demand and controller progress across
  the localized phase `24.17-26.69 s` lag and the remaining completion budget.
- Derive at most one bounded execution-clock/progress candidate. It must keep
  every authoritative source anchor and source timestamp immutable, preserve
  source order and endpoint, retain symmetric total-pitch authority and all
  frozen safety/quality gates, and recover time only from already healthy
  regions rather than globally extending the `3x` horizon.
- Add regression and evidence-contract tests and perform a no-gate-relaxation
  diff audit before requesting a new runtime authorization. Do not create a
  namespace, launch Isaac, case 43, residual capture, BC, PPO, or training
  during this CPU stage.

## Round 102: commanded-base phase-governor candidate

- A rate-slack audit rejects a duration-preserving, state-identical retime.
  Slowing the phase `24.17-26.69 s` window from its saturated `0.4 m/s`
  feed-forward toward `0.2 m/s` needs approximately `2.533 s` of additional
  execution time, while every other interval together contains only
  `0.057343 s` of legal rate slack. Recovering that time elsewhere would
  exceed an unchanged base, yaw, riser, or proxy rate limit.
- The runtime phase governor currently uses error to the nominal plan base
  allocation even after camera-lever compensation shifts the base command by
  up to `0.05 m`. That nominal base state is an internal decomposition target,
  not the physical `cam_link` quality target. The corrected deterministic
  ownership is to use error to the base command actually sent to the tracking
  controller, while still taking the minimum with physical camera-position
  error, balance pitch, and the existing camera-recovery governor.
- A read-only reconstruction of all `145` one-hertz Round-101 trace samples
  reproduces the recorded legacy progress scale with zero numerical error.
  Substituting the already-recorded compensated command target changes base
  error by mean `-0.044370 m` and never more than the frozen `0.05 m` lever
  correction. On the same recorded states, mean progress would increase from
  `0.306371` to `0.518858`, representing `30.8106 s` of sampled phase budget
  versus the observed `4.3809 s` completion deficit. This is counterfactual
  diagnosis only, not dynamic pass evidence.
- Commit `5e6a9b1` adds a default-off
  `--use-commanded-base-progress-error` candidate. It is rejected unless both
  camera-lever compensation and the phase governor are active. Default behavior
  still selects the nominal plan base error exactly; enabling the candidate
  changes only phase-progress error ownership, not the base command, LQR,
  source/plan arrays, either clock, gains, velocity cap, total-pitch limit,
  runtime horizon, safety limits, or quality thresholds.
- Policy-rate evidence separately records nominal, commanded, and selected
  base errors, the selected source, p95/max and selected-versus-nominal deltas.
  Missing or internally inconsistent samples fail controller evidence. The
  established tracking-profile identifier is unchanged; the candidate has a
  separate explicit contract field.
- Focused Mac syntax and candidate tests pass `6/6`. The broader Mac mirror
  reaches `52` passes with only two unrelated failures caused by its missing
  legacy whole-body URDF. A detached `.98` CPU worktree with the authoritative
  hardware-envelope fixture passes the full suite `390/390` in `26.67 s`.
  The temporary worktree was removed afterward. No route, authorization token,
  runtime namespace, Isaac process, dataset, residual capture, BC, PPO, or
  training was created.

## Next round after Round 102

- Review the `5e6a9b1` candidate and its no-gate-relaxation diff. If accepted,
  add one separate hash-bound case-42 authorization route that keeps the v20
  source, plan, both clocks, initialization, `0.2 m/s` cap, exact-hold floor,
  symmetric total-pitch target, `3x` horizon, and all current physical/quality
  gates unchanged.
- Require runtime evidence to prove the selected error equals the compensated
  commanded-base error at every policy step, its difference from nominal base
  error never exceeds the `0.05 m` correction, and residual action remains
  zero. Stop after one result with no automatic retry.
- Do not launch case 43, residual capture, BC, PPO, training, or obstacle work.

## Round 103: hierarchical policy boundary and commanded-base admission route

- The project goal remains unchanged, but the control ownership is now stated
  explicitly. Deterministic inner loops own primary two-wheel balance, actuator
  limits, hard collision enforcement, emergency behavior, riser hard limits,
  and gimbal cable/angular limits. A future learned policy may operate only
  above those loops as a bounded supervisory residual or allocation policy.
- The initial learned-action contract therefore remains the already defined
  bounded residual over deterministic commands, currently
  `[delta_v, delta_wz, delta_h_target]`. Camera-attitude residuals may be
  introduced later using semantic camera attitude, never physical DJI motor
  joint labels. Every final command must still pass the deterministic safety
  supervisor. No learned action is active in Gate C.
- Review hardening commit `20ed7cb` made commanded-base phase-governor
  telemetry fail closed: policy-rate sample count, selected-source identity,
  and the selected-versus-nominal error bound must all agree, and the latter
  may not exceed the frozen `0.05 m` camera-lever correction.
- The new CPU-only admission route is restricted to case 42 and the existing
  v20 source, manifest, plan, initialization, `0.2 m/s` cap, exact-hold
  floor, symmetric total-pitch target, `3x` horizon, gains, USD, and quality
  gates. It records `20ed7cb` as the reviewed controller parent and uses a
  fresh namespace. The playback switch remains default-off.
- Runtime summarization now treats physical dynamics, thermal/safety
  admission, commanded-base controller evidence, residual-label envelope, and
  training admission as independent outcomes. Missing, forged, non-finite, or
  over-`0.05 m` progress telemetry rejects the runtime contract without
  changing commands, clipping labels, or creating a dataset.
- The focused route and summarizer suites pass `42/42`. The Mac full suite is
  not authoritative because six unrelated modules require `gymnasium`, which
  is absent from the local Python environment; full validation remains assigned
  to the established `.98` Isaac Python environment.

## Next round after Round 103

- Commit, push, and transfer the exact CPU contract to `.98`, then run the
  complete authoritative CPU suite and verify clean HEAD equals upstream.
- Only after a fresh WSL, Windows-process, and NVIDIA ownership preflight may
  one exclusive case-42 canary be considered. Stop after that one result with
  no automatic retry.
- Keep case 43, residual capture, BC, PPO, all training, and obstacle work
  closed. A dynamic pass would qualify only this deterministic candidate; it
  would not itself authorize learned-policy training.

## Round 104: commanded-base progress canary is dynamically rejected

- Commit `92c15aa` sealed the case-42-only route after focused tests passed
  `42/42`. It was pushed and transferred to `.98` with bundle SHA-256
  `b561c4825fc80f36725ea84926283a8cfcec427d9c6855295f0a0725d6eb8c49`.
  The authoritative suite passed `394/394` in `27.87 s). Local, GitHub,
  remote HEAD, and remote upstream all matched, the namespace was fresh, and
  WSL, Windows, and NVIDIA ownership checks were empty before launch.
- The one exclusive case-42 run closed fail-closed with no retry. It consumed
  `28,980` scored steps and the full `144.9 s` wall/`146.9 s` simulated
  horizon, reaching phase `44.911829/48.297533 s`. This is approximately
  `0.995 s` farther than Round 101, but still leaves `3.386 s` unconsumed.
- Physical position p95 failed at `0.192780 m` against the unchanged
  `0.15 m` gate; max remained bounded at `0.202003 m`. The only failed
  dynamic checks were `completed_reference` and `position_p95_bounded`.
  Attitude p95/max were `0.131819/1.745864 deg`, pitch p95/max were
  `5.710350/6.793391 deg`, proxy error max was `1.558272 deg`, riser error
  p95/max were `0.010933/0.013184 m`, action saturation was zero, thermal
  admission passed, and no termination occurred.
- The new controller evidence passed independently for all `28,980` policy
  steps. The selected source was the compensated commanded-base target; mean
  selected-versus-nominal error delta was `-0.044223 m`, absolute maximum was
  `0.05 m`, and all source, sample-count, and bound checks passed.
- The prospective residual-label envelope passed, but admission remained
  false. Raw labels were not applied, residual action stayed exactly
  `[0,0,0]`, no dataset was written, and residual capture, BC, PPO, and all
  training remained closed. The demonstrated dynamic union remains `42/70`.
- Trace comparison rejects another phase-only adjustment. During the critical
  reverse segment around elapsed `84-96 s`, effective deterministic velocity
  reference was approximately `-0.162` to `-0.166 m/s`, while measured
  wheel/root velocity settled near `-0.098` to `-0.114 m/s` with zero action
  saturation. Advancing the target faster increased the long position-error
  plateau rather than supplying missing physical tracking authority.
- Sealed hashes are admission
  `27c9cbe39d0af86e33e67c54ed0a2758d91160e437084b37f861f183533b4b95`,
  gate `5aae2fd78b0b960294c6097ed4c80094d4a69a2d2b95424b2fc984563200f9fa`,
  log `617923aa1bae1fa4e0ae54f6cda19928f43e6e59a1e391a6a6c9e8a287d587f7`,
  exit-code evidence
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`,
  and summary
  `4654037493ed85d265416cc0d316c2d8807df6dc055741053bb65f8f64173052`.

## Next round after Round 104

- Stay CPU-only and audit deterministic longitudinal authority inside the
  balance cascade. Use policy-rate pitch, pitch-rate, wheel-velocity, common
  action, total-pitch-limit, and effective-reference evidence to explain the
  steady `0.05-0.06 m/s` reverse tracking deficit despite zero saturation.
- Derive at most one bounded, default-off inner-loop/controller candidate with
  synthetic sign, stability, saturation, and legacy-compatibility tests. Do
  not compensate by moving source anchors, relaxing the `0.15 m` position
  gate, widening residual limits, or handing primary balance to RL.
- Do not add a runtime token or namespace until CPU review and the full
  authoritative suite pass. Keep case 43, residual capture, BC, PPO, training,
  and obstacle work closed.

## Round 105: opposing longitudinal PI-memory candidate

- Reconstructing the sealed case-42 trace with the frozen gain matrix localizes
  the missing reverse authority to stale outer-loop PI memory rather than LQR
  saturation. When the effective reference became approximately
  `-0.16 m/s`, the stored longitudinal integral was still about `+0.294` and
  opposed the requested direction. It crossed zero only several seconds later;
  the same lag then appeared with opposite sign at the reverse-to-forward
  transition. The inner action was not saturated during this interval.
- A bounded, default-off controller candidate resets only the longitudinal PI
  seed when all three conditions hold: the effective reference exceeds a
  `0.05 m/s` deadband, measured velocity is deficient in that commanded
  direction, and the existing integral opposes that direction. It then
  integrates the current error normally. It does not reset during overspeed
  braking, alter yaw memory, change the frozen LQR gains, change source or plan
  data, relax the symmetric total-pitch limit, alter action limits, or bypass
  anti-windup.
- New policy-rate evidence records controller-update and held-command counts,
  reference sign changes, opposing-memory events, actual reset count,
  directional velocity deficit, pitch/pitch-rate/wheel-velocity LQR
  contributions, PI magnitude, total-pitch-limit occupancy, and common-action
  magnitude. Evidence changes do not alter deterministic commands.
- Synthetic coverage proves default-off bitwise compatibility, one reset across
  a gradual zero crossing, deadband behavior, compact controller-state support,
  anti-windup from the reset seed, preservation of legitimate overspeed braking
  memory, finite telemetry, and fail-closed validation. The focused controller,
  evidence, and playback suite passes `65/65`; compilation, stale-identifier
  scan, and `git diff --check` also pass.
- The broader Mac mirror reaches `233` passes. Its `27` failures are all missing
  generated URDF/mesh assets in the lightweight local clone, so authoritative
  full validation remains assigned to the complete `.98` worktree. No runtime
  route, token, namespace, Isaac process, dataset, residual capture, BC, PPO,
  or training was created.

## Next round after Round 105

- Commit and push this CPU-only candidate, transfer the exact commit to `.98`,
  and run the complete authoritative CPU suite with the generated assets
  present. Diff-audit that only the default-off PI-memory candidate, telemetry,
  tests, and this log changed.
- Stop for controller/evidence review after the authoritative suite. Do not add
  a runtime authorization route or run another case-42 canary in the same
  round.
- Preserve deterministic ownership of primary balance and all hard safety
  limits. A future learned policy remains a bounded supervisory residual over
  model-based commands; case 43, residual capture, BC, PPO, training, and
  obstacle work remain closed.

## Round 106: authoritative CPU qualification

- Candidate commit `35f775c39ed2d0c22b52be5dd8f9641354ee0b8f` was
  pushed to `codex/two-wheel-riser-rl` and transferred exactly to `.98` with
  incremental bundle SHA-256
  `b8a8874ea962d72b2174b28e8e883ba070230065738738441da72fd657520f0c`.
  Local, GitHub, `.98` HEAD, and the `.98` upstream tracking ref agree.
- The complete authoritative `.98` CPU suite passes `402/402` in `39.92 s`
  (`42.84 s` measured command wall time), with only the two established
  pytest configuration warnings. The focused Mac suite remains `65/65`.
- The six pre-existing untracked evidence directories and the pre-sync archive
  on `.98` remain untouched. WSL playback-process and NVIDIA compute-owner
  checks are empty after validation. No runtime route, authorization token,
  namespace, Isaac process, dataset, residual capture, BC, PPO, or training was
  created.

## Next round after Round 106

- Stop for review of the default-off longitudinal PI-memory trigger and its
  telemetry contract. The next runtime action, if separately authorized, is
  exactly one fresh, hash-bound deterministic case-42 canary with the same
  source, plan, clocks, gains, physical limits, and quality gates as Round 104.
- Compare completion, position p95/max, reverse/forward transition lag,
  integral-reset count, velocity-deficit duration, total-pitch-limit occupancy,
  action saturation, thermal evidence, and all existing safety metrics. Stop
  after the first result with no automatic retry.
- A deterministic pass would qualify only this controller candidate. It would
  not authorize residual capture or learning; the raw residual envelope must be
  recomputed only after deterministic dynamic qualification, and BC/PPO remain
  separately gated.

## Round 107: hash-bound opposing-PI-reset case-42 route

- A fresh one-case authorization route now binds case 42 to the unchanged v20
  source manifest, smoothed plan, initialization separation, `0.2 m/s`
  deterministic velocity cap, exact-hold floor, symmetric `6 deg` total-pitch
  limit, commanded-base phase governor, frozen gains, riser USD, and all
  existing physical/quality gates. Its namespace is
  `20260720_gate_c_smoothed_case42_v20_zero_progress_hold_cap020_total_pitch_commanded_base_progress_opposing_pi_reset_v1_exclusive`.
- The route pins reviewed controller parent
  `35f775c39ed2d0c22b52be5dd8f9641354ee0b8f`, requires it to exist and be an
  ancestor of the eventual clean pushed runtime commit, and seals runtime file
  identities in admission evidence. Conflicting or missing authorization,
  dirty tracked state, HEAD/upstream mismatch, reused namespace, identity
  mismatch, or occupied WSL/Windows/NVIDIA ownership fails before Isaac.
- Runtime evidence now independently requires the default-off candidate to be
  enabled at the exact `0.05 m/s` deadband and records the longitudinal
  authority schema, policy-rate and controller-update counts, held-command
  count, sign changes, reset count, velocity-deficit metrics, PI magnitude,
  LQR contribution aggregates, total-pitch-limit occupancy, and common-action
  magnitude. At least one reset must be observed for candidate exercise; no
  reset, absent/non-finite telemetry, forged parent, or inconsistent counts
  reject the runtime contract without changing physical dynamic classification.
- Healthy, missing/zero-reset, non-finite, inconsistent-count, and forged-parent
  paths are fail-closed. Focused wrapper, summarizer, controller, telemetry, and
  playback tests pass `108/108`; shell syntax, Python compilation, and
  `git diff --check` pass. The diff contains no source, plan, gain, robot-asset,
  gate-threshold, controller-command, residual-policy, capture, BC, or PPO
  change.

## Next round after Round 107

- Commit, push, transfer, and run the complete authoritative `.98` CPU suite.
  Only if HEAD equals upstream, the namespace is absent, and WSL, Windows, and
  NVIDIA ownership are empty may the new authorization token be used once.
- Run case 42 only and stop after the first result with no retry. Audit dynamic
  completion separately from longitudinal-authority telemetry and the
  residual-label envelope. Keep residual action unapplied and write no dataset.
- If physical dynamics fail, preserve and seal the first reject and return to
  CPU diagnosis. If they pass, stop before capture and recompute the raw
  residual envelope as a separate next stage; BC/PPO remain closed.

## Round 108: PI-memory reset completes case 42 but misses position p95

- Route commit `b0e2e8448a06b9a4f97707fa8cddce0e052bc90f` was pushed
  and transferred exactly to `.98` with bundle SHA-256
  `68678851016886e986fcf3fe4b4ef49b951c44fa83c632fb2e2a77b36b512e40`.
  The complete authoritative CPU suite passed `403/403` in `27.95 s`
  (`30.21 s` command wall time), and the fresh namespace plus WSL, Windows,
  and NVIDIA ownership preflight passed before one exclusive launch.
- The canary completed the full `48.297533 s` execution phase in `28,437`
  scored steps and `142.185 s` wall time. This recovers the prior
  `3.385704 s` phase deficit and uses `543` fewer scored steps than Round 104.
  There was no termination and no action saturation.
- Dynamic admission still fails solely on position p95:
  `0.186930 m` against the unchanged `0.15 m` gate. Position max passes at
  `0.200354 m`; attitude p95/max pass at `0.135134/1.743115 deg`, pitch
  p95/max are `6.513145/7.025337 deg`, proxy servo max is `1.557279 deg`,
  thermal admission passes, and all controller/runtime evidence passes.
- Longitudinal telemetry is complete for all `28,437` policy steps and
  `7,110` controller updates. It records `25` opposing-memory resets,
  `18,874` directional-deficit steps (`66.3713%`), mean/max deficit
  `0.079778/0.363252 m/s`, PI magnitude reaching the unchanged `0.7` bound,
  total-pitch-limit occupancy for `2,624` steps (`9.2274%`), and common-action
  maximum `0.596117` below the unchanged `0.8` limit.
- One-hertz direction segmentation shows the candidate solved the formerly
  delayed positive segments, whose mean position error falls to approximately
  `0.067-0.076 m`. Reverse segments remain systematic at approximately
  `0.150-0.162 m`. The exact phase now completes, so another retime or phase
  governor adjustment is not justified; the remaining gap is deterministic
  longitudinal feedback authority during directional deficit.
- Residual-label envelope passes independently, but training admission remains
  false. Raw labels were not applied, residual action stayed zero, no dataset
  was written, and residual capture, BC, PPO, and training remained closed.
  The demonstrated dynamic union remains `42/70`.
- Sealed hashes are admission
  `d8253da9b2668256bfb41c5516555a3532d5496e28926ccafad3802e80dd68b2`,
  gate `ce27128bf0018b55e109273ce0ceb5a0f0caaf771ff35dc53d218524ec2f404b`,
  log `01e45faf179602babc799ddbefa7c2e7a6d09fdcd84a27bb999362ebf2bf536a`,
  exit-code evidence
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`,
  and summary
  `92cca965711af3e72411d761e46d1e8097602cad6f7bcf7e0b9b2fc13db72e87`.

## Next round after Round 108

- Stay CPU-only. Derive one default-off longitudinal proportional-authority
  candidate from the sealed trace while preserving the symmetric `6 deg`
  total-pitch limit, `0.8` action limit, gains, PI-memory reset, source/plan,
  both clocks, and every physical/quality gate.
- Use counterfactual target/action bounds plus synthetic sign, saturation,
  anti-windup, default-compatibility, and provisional plant-envelope tests to
  choose a single bounded `vx_kp` value. Do not tune from position score alone
  and do not add a runtime token in the controller commit.
- Keep case 43, residual capture, BC, PPO, training, and obstacle work closed.

## Round 109: evidence-bounded longitudinal proportional authority

- The sealed Round-108 telemetry provides a direct action-headroom bound for a
  proportional-gain increase:
  `delta_Kp <= (action_limit - reserve - observed_action_max) /
  (abs(pitch_action_gain) * maximum_velocity_deficit)`. Using unchanged action
  limit `0.8`, explicit reserve `0.03`, observed common-action max `0.596117`,
  frozen pitch gain magnitude `3.954227`, and maximum deficit `0.363252 m/s`
  gives maximum `vx_kp = 0.721056` from the `0.6` baseline.
- Select `vx_kp = 0.72` as the single bounded candidate. Its conservative
  worst-case counterfactual common-action bound is below `0.77`, preserving at
  least the explicit `0.03` reserve before the existing `0.8` clip. The sealed
  one-hertz trace predicts approximately `0.365 deg` more mean reverse physical
  pitch target while retaining the same symmetric `+/-6 deg` total target.
  This is a controller-authority hypothesis, not dynamic pass evidence.
- Added a fail-closed reusable derivation helper and an optional
  `--controller-vx-kp` playback override restricted to `(0, 1]`. The option is
  absent by default, and default controller output/evidence remains unchanged.
  Runtime evidence records the resolved Kp and includes it in controller
  overrides only when explicitly selected.
- Synthetic tests prove signed authority increases in both directions, the
  total physical pitch target remains inside `+/-6 deg`, action remains inside
  `+/-0.8`, the candidate stays close to the baseline for representative
  deficits, and malformed/non-finite/no-headroom derivations fail closed.
  Focused controller/evidence/playback tests pass `71/71`; the broader route,
  summarizer, controller, evidence, and playback set passes `114/114`.
- No source, plan, clock, gain matrix, robot asset, physical/quality threshold,
  authorization token, runtime namespace, Isaac process, dataset, residual
  capture, BC, PPO, or training was added or changed.

## Next round after Round 109

- Commit, push, transfer, and run the complete authoritative `.98` CPU suite.
  Review the derivation, default compatibility, and action-margin evidence
  before creating any separate runtime route.
- If reviewed, run the existing deterministic push/plant-envelope campaign at
  `vx_kp=0.72` before another case-42 rollout. Require nominal and provisional
  variations, both `+/-0.2 m/s` directions, fixed pitch/action limits, and no
  learned action. Reject the candidate if balance, recovery, saturation, or
  direction symmetry regresses.
- Keep case 42 runtime retry, case 43, residual capture, BC, PPO, training, and
  obstacle work closed until that robustness gate passes.

## Round 110: representative riser plant-envelope contract

- A source audit rejected the existing `evaluate_lqr_tracking_push.py` default
  as qualification evidence for the riser controller. Its default
  `TWO_WHEEL_BALANCE_CFG` loads the lightweight
  `recomoProto2_two_wheel_balance.usd`, whose equilibrium pitch is close to
  zero; trajectory playback instead loads the complete
  `TWO_WHEEL_RISER_CFG`. The earlier chassis campaign remains valid only for
  the lightweight balance plant and must not be relabeled as riser evidence.
- Added an explicit, default-off `--robot-form riser` evaluation path. It loads
  the complete riser/gimbal/camera USD, holds a selected riser height and the
  zero semantic-gimbal pose, and computes the physical COM equilibrium pitch
  from all current rigid-body masses and `body_com_pos_w` for every environment.
  The original `balance` form remains the default and retains its existing
  schema.
- The riser path exposes and records the exact deterministic controller options
  needed by the case-42 candidate: root-velocity outer feedback, total physical
  pitch limiting, opposing PI-memory reset and its deadband, and the selected
  `vx_kp`. It does not alter the frozen LQR gain matrix, the `6 deg` pitch
  limit, the `0.8` action limit, or any trajectory gate.
- Riser evidence fails closed on maximum riser and gimbal hold error and records
  per-scenario COM-bias bounds, robot form, USD path, controller settings, and
  independent no-learned-action/no-dataset/no-capture/no-BC/no-PPO/no-training
  markers. The candidate has not run in Isaac and has no dynamic admission.
- The new source-contract tests and related controller/playback regressions pass
  `102/102` locally. Syntax compilation and `git diff --check` also pass.
  Implementation commit `0dc1aad417aecdc0ed1b29110d08e7abf5db0622`
  was pushed and transferred exactly to `.98` with bundle SHA-256
  `814dc113ffa06aef28d1c2b0f9e7545d57a0815a087b524de6aad4ad417a6f89`.
  Remote HEAD and upstream match, and the authoritative `.98` CPU suite passes
  `414/414` in `27.41 s` (`29.36 s` command wall time).

## Next round after Round 110

- Run the complete authoritative `.98` CPU suite after commit transfer. Do not
  issue a runtime token or start Isaac from an unpushed or dirty commit.
- Build a separate guarded plant-envelope route for exactly one height per
  shard at riser positions `0.0`, `0.6`, and `1.2 m` (physical camera heights
  approximately `0.6`, `1.2`, and `1.8 m`). Each shard must cover both
  `-0.2/+0.2 m/s`, nominal plus the provisional plant variations, the pinned
  case-42 deterministic controller options, and the unchanged pitch/action and
  recovery gates.
- Stop on the first balance, tracking-recovery, saturation, direction-symmetry,
  riser-hold, or gimbal-hold reject. Only a clean three-height pass may open one
  fresh case-42 retry route. Residual capture, BC, PPO, training, case 43, and
  obstacle work remain closed.

## Round 111: guarded three-height plant-envelope route

- Added a riser-only bidirectional authority gate. It requires both forward and
  reverse scenarios and rejects achieved-speed magnitude asymmetry above
  `0.05 m/s`. This is additional admission evidence; it does not change the
  controller, command, plant, or trajectory thresholds. The lightweight
  balance form retains its existing behavior.
- Added `run_riser_lqr_plant_envelope.sh` as a single-shard launcher. The only
  accepted shards are `low`, `mid`, and `high`, corresponding to riser joint
  positions `0.0`, `0.6`, and `1.2 m`. Each uses an independent authorization
  and fresh namespace; the launcher never advances to another height.
- Every shard pins the reviewed evaluator lineage and exact evaluator,
  controller, tracking, robot-config, gains, and full-riser USD hashes. It
  requires clean HEAD equal to upstream, rejects conflicting environment
  overrides, and checks WSL processes, Windows command lines, and NVIDIA
  compute ownership before launch and after exit.
- The campaign contract is `56` scenarios per height: both `-0.2/+0.2 m/s`,
  zero yaw command, `-20/+20 N` deterministic pushes, and all `14` provisional
  plant variations. It pins `structural_robust_v1`, `vx_kp=0.72`, the existing
  `6 deg` total-pitch limit, `0.8` action limit, root-velocity outer feedback,
  and opposing PI-memory reset. Success requires `56/56`, complete direction
  evidence, hold gates, and no learned action.
- Admission and final-status evidence record runtime identities, result/log
  hashes, the exact height, controller and scenario checks, and explicit
  no-dataset/no-capture/no-BC/no-PPO/no-training fields. Missing output or any
  failed check closes the shard with no retry or next-height launch.
- Focused route/controller/playback tests pass `105/105`; dedicated negative,
  hash-binding, syntax, and source-contract checks pass `9/9`. No runtime
  namespace, authorization use, Isaac process, dataset, or training was started
  by this CPU-only change.

## Next round after Round 111

- Commit, push, transfer, and pass the full authoritative `.98` CPU suite.
- Verify exclusive WSL, Windows, and NVIDIA ownership, then run only the `low`
  shard under its exact authorization. Seal and audit its admission, result,
  log, exit code, and final status.
- Stop on any low-height reject. Only a complete low-height pass may admit the
  `mid` shard; only low plus mid may admit `high`. Do not start another
  case-42 rollout, residual capture, BC, PPO, training, case 43, or obstacle
  work during this campaign.

## Round 112: low-shard output quarantine and semantic-proxy correction

- Commit `37dfad82382c0a5ed766e7c8de1b6d105f7ff2ef` was pushed and
  transferred to `.98` with bundle SHA-256
  `0fe4580937ddb593ff0cfb1aed57df4b60e291ef1fbf5fda32ca4eb52fc49e9b`.
  The authoritative `.98` CPU suite passed `418/418` in `28.03 s`
  (`30.08 s` command wall time). Clean HEAD/upstream, a fresh low namespace,
  and empty WSL/NVIDIA ownership were verified before exactly one launch.
- The v1 low route failed its evidence contract and did not admit dynamics.
  `OUTPUT_WIN` escaped `$NAMESPACE`, so Windows wrote `result.json` beneath the
  literal path `artifacts/two_wheel_riser$NAMESPACE` while the intended
  namespace correctly reported `result_written=false`. The misrouted output is
  preserved as
  `20260720_riser_lqr_plant_envelope_vxkp072_low_v1_MISROUTED_OUTPUT_QUARANTINED`;
  its result SHA-256 is
  `788bfd3d8983a961bef60108277efae1b1f95f438b764f6f6333657e293b9c80`.
- The quarantined result is diagnostic only, not admitted evidence. All `56`
  scenarios survived and recovered balance/tracking, aggregate selected `vx`
  RMSE was `0.049120 m/s`, peak pitch was `8.653920 deg`, action saturation was
  zero, and forward/reverse achieved-speed asymmetry was `0.001401 m/s`.
  Every scenario failed solely on the `1 deg` gimbal hold gate, with maximum
  proxy error `9.258271 deg`.
- That gimbal result exposed a contract mismatch rather than a wheel-controller
  failure. The DJI joints in this asset are semantic attitude-setpoint proxies,
  not physical motor shafts. Accepted trajectory playback writes their semantic
  state through the deterministic DJI adapter each policy step; the new plant
  gate had only applied a passive position drive. Added an explicit,
  riser-only `--semantic-proxy-state-adapter` option that uses the same ideal
  proxy-state contract. It remains default-off and does not alter wheel
  commands, LQR gains, pushes, plant variations, pitch/action limits, or gates.
- Fixed the Windows output path expansion, added an exact path regression, and
  moved all three shard tokens/namespaces to fresh v2 identities. The v2 runner
  requires and records the semantic proxy adapter. No mid/high shard, case 42,
  residual capture, BC, PPO, or training started.

## Next round after Round 112

- Run focused and full authoritative CPU tests, commit, push, and transfer the
  v2 route. Verify the evaluator hash, output-path expansion, clean
  HEAD/upstream, fresh v2 low namespace, and exclusive WSL/Windows/NVIDIA
  ownership.
- Run exactly one v2 low shard. Treat all physical, direction, hold, identity,
  and no-learning checks as hard. Stop on any reject; only a complete low pass
  may admit the v2 mid shard.

## Round 113: low-height dynamics pass; final-status float repair

- Commit `10d48a6b66ca4333c1d4a85ed2d5705329e86c67` was pushed and
  transferred exactly with bundle SHA-256
  `70d3c79aad4e243053284e26f8d85c6cb85f8737550bf860e3854058bc2f09ca`.
  The authoritative `.98` suite passed `418/418` in `27.24 s`
  (`29.21 s` command wall), and clean HEAD/upstream, fresh v2 low namespace,
  and empty WSL/NVIDIA ownership were verified before one launch.
- The corrected v2 low simulator result is a physical and scenario pass:
  `56/56` scenarios passed, survival/balance-recovery/tracking-recovery were all
  `1.0`, aggregate selected `vx/wz` RMSE was
  `0.050709/0.000701`, peak pitch was `8.426244 deg`, action saturation was
  zero, riser/gimbal hold maxima were `0.00000415 m/0.057881 deg`, and
  forward/reverse achieved-speed asymmetry was `0.017954 m/s`.
- Result SHA-256 is
  `d359aa90ae34cb2a59ffdcc130e04914ba3de2344fcaa3b34529befad85dd794`;
  admission SHA-256 is
  `75614e75a64f62f016ecf220676e34241e20a31a8db1897ea0988aa8af06b793`;
  runtime-log SHA-256 is
  `6e43f238119655680e1da22f317649f726f343ca4817e043dee4dc21577abf92`.
  No learned action, dataset, capture, BC, PPO, or training was present.
- The v2 `final_status.json` remained false only because its embedded validator
  compared the serialized `pitch_reference_limit_deg=6.000000000000001` to
  `6.0` with exact equality. Every physical, identity, hold, scenario,
  direction, and no-learning check passed. This is an evidence-validation
  defect, not a dynamic reject, and no Isaac rerun is justified.
- Extracted the final-status logic into
  `summarize_riser_lqr_plant_envelope.py`. It requires all four source evidence
  files, verifies admission/runtime/shard/height identity, preserves every hard
  gate, uses absolute-tolerance comparisons only for serialized numeric
  constants, seals source hashes plus its own blob hash, and emits schema v2.
  Healthy, roundoff, no-learning-negative, and missing-evidence tests pass.
  The launcher now pins and records this summarizer rather than embedding a
  second validator implementation.

## Next round after Round 113

- Commit, push, transfer, and run the full authoritative CPU suite. Preserve the
  original v1 final status as pre-reseal evidence, then reseal the existing low
  namespace with the committed summarizer. Verify every check and source hash;
  do not rerun low-height physics.
- Only after the resealed low status passes may one fresh v2 mid-height shard
  run. High, case 42, residual capture, BC, PPO, training, and obstacle work
  remain closed.

## Round 114: low resealed pass; mid hold-gate contract correction

- Commit `96875bd64f92c71b68863b1aaedf042e0398c85e` was pushed and
  transferred with bundle SHA-256
  `c19179c8130a91f8ba880f7285fe0b39de4407bdefe0a7a2c46a7286bcad8127`;
  the authoritative `.98` suite passed `421/421` in `27.25 s`
  (`29.51 s` command wall time).
- Preserved the original low final status at SHA-256
  `8bb9203e411686c019761e1bb8dc429e8b92445ca3c8e938535878f5b44665af`
  and resealed the unchanged low admission/result/log/exit artifacts. Schema-v2
  final status passes every check at SHA-256
  `b5d99fcefc660200b6b5883b44065761aee12a35c93e4fbf05d9f22b0c452026`.
  No low physics rerun occurred.
- One exclusive mid-height v2 shard then ran at riser position `0.6 m` and
  stopped fail-closed. All `56` scenarios survived and passed balance recovery,
  tracking recovery, `vx/wz` tracking, pitch/roll, action saturation, gimbal
  hold, and direction symmetry. Aggregate selected `vx/wz` RMSE was
  `0.038222/0.001502`, peak pitch was `8.442031 deg`, action saturation was
  zero, and speed asymmetry was `0.033005 m/s`.
- The only failed check was riser hold: maximum error `0.011223 m` against the
  newly introduced `0.010 m` maximum. Result/final-status SHA-256 values are
  `8a7f7a9599a976e73656ecd9491f6998cbefb1b838ff4bd291b201e4565abd64`
  and `cb463058bbb8c86c7aacc07bd553bbf6d9d833759cf7eccc70771817ed75ce1d`.
  High did not start.
- CPU audit found the `0.010 m max` threshold inconsistent with the established
  riser contract. Static/dynamic riser smokes and Gate C use `0.030 m`; prior
  accepted dynamic evidence reports `0.0105-0.0138 m` maxima. Corrected the
  plant-envelope route to an explicit, hash-bound `0.030 m maximum`, which is
  stricter than the established `0.030 m p95` gate. The summarizer now proves
  this exact threshold plus the unchanged `1 deg` gimbal threshold. No actuator,
  controller, plan, plant, push, pitch/action limit, or learned component was
  changed.
- Mid/high route identities move to fresh v3 namespaces. The v2 mid reject is
  preserved and will not be relabeled. Residual capture, BC, PPO, training,
  case 42, high, and obstacle work remain closed.

## Next round after Round 114

- Pass focused and full authoritative CPU tests, commit, push, and transfer the
  gate-contract correction. Verify fresh mid-v3 namespace and exclusive
  WSL/Windows/NVIDIA ownership.
- Run exactly one mid-v3 shard. Stop on any reject. Only the existing sealed low
  pass plus a clean mid-v3 pass may admit one high-v3 shard.

## Round 115: three-height plant envelope passed; case-42 route prepared

- The corrected mid-v3 shard passed all `56/56` scenarios. Aggregate selected
  `vx/wz` RMSE was `0.038222/0.001502`, peak pitch was `8.442031 deg`, action
  saturation was zero, direction asymmetry was `0.033005 m/s`, and riser/gimbal
  hold maxima were `0.011223 m/0.084689 deg`. Result/final-status SHA-256 values
  are `d5f2abb66663e1afaf68b403acde59d70daea2ea8531dfd3ab7de9a3301566cc`
  and `c6c06b74d3752ca8b24d1ad01251d8b5a77a47fd9980eacb3bc822396cbc46ee`.
- The exclusive high-v3 shard, representing the approximately `1.8 m` physical
  camera-height condition, also passed `56/56`. Aggregate selected `vx/wz` RMSE
  was `0.030486/0.002347`, peak pitch was `9.383905 deg`, action saturation was
  zero, direction asymmetry was `0.046568 m/s`, and riser/gimbal hold maxima
  were `0.011437 m/0.084396 deg`. Result/final-status SHA-256 values are
  `e7874d6e9f223040435dc0cbe573168e668b67090641aa769ce101cf0f8377b0`
  and `c377bae12c23069633e337b6a6db22308bc6bf153ba9eb2a18d719eb870594ef`.
- Together with the sealed low-v2 pass, the representative full-riser plant
  envelope is now complete at low, mid, and high height. This admits one fresh
  case-42 deterministic trajectory canary at the already selected
  `vx_kp=0.72`; it does not admit residual capture, BC, PPO, or training.
- Added a fresh fail-closed case-42 authorization and namespace. It preserves
  the exact v20 source/portfolio/plan identities, initialization preroll,
  zero-progress hold, `0.2 m/s` recovery cap, `6 deg` total-pitch limit,
  commanded-base progress error, and opposing PI-memory reset. Admission,
  runtime gate, and summary must all report exactly `controller_vx_kp=0.72` and
  reviewed parent `1e7ebbde4dcb241fde63275e5434dfa2fc4d1cb8`.

## Next round after Round 115

- Pass focused and full authoritative CPU tests, commit, push, and transfer the
  route. Verify a clean pushed HEAD, fresh namespace, exact plan/controller
  identities, and exclusive WSL/Windows/NVIDIA ownership.
- Run exactly case 42. Stop and seal either outcome. Do not start case 43,
  residual capture, BC, PPO, training, or obstacle work.

## Round 116: explicit-vx-Kp case-42 reject and retiming diagnosis

- Commit `84f1af82bb4dc26f016e99bf20143f44c6e94540` was pushed and
  transferred with bundle SHA-256
  `f027219f4e82e4c65b71802f02d2895354c00d1db03d63265d3596eec33c10d4`.
  The authoritative `.98` CPU suite passed `422/422` in `27.14 s`
  (`29.06 s` command wall time). Clean HEAD/upstream, fresh namespace, and
  exclusive WSL/Windows/NVIDIA ownership were verified before one launch.
- The exact `vx_kp=0.72` case-42 canary completed all `48.297533 s` of plan
  phase in `28653` policy steps and stopped fail-closed. Thermal admission,
  attitude, pitch, action saturation, exact gain evidence, and the residual
  label envelope passed. No residual action was applied and no dataset,
  capture, BC, PPO, or training was created.
- The only physical failed check was position p95: `0.182557 m` against the
  unchanged `0.15 m` gate; maximum position error was `0.189559 m`. Compared
  with the `vx_kp=0.60` run (`0.186930/0.200354 m`), the gain increase improved
  p95 by only `4.4 mm` and max by `10.8 mm`. Action saturation remained zero.
- Admission/gate/log/summary SHA-256 values are
  `76c3ac61d3ad8e61aa50e6fbd7c0186e24376448536b108d4ff89e28b2e98aa0`,
  `3d6377db0eb62c6984f439cb2852a85c352810c3c8ea828db70a4edef4aab53e`,
  `731808822b9fe6d38222fda3fb397f2c6b977c5f258e08910547b7e6587506d5`,
  and `9a81da746201be06c4f82abfcf2b0c0fe72a39ce024ab7785218aaa44f6d6172`.
- Trace diagnosis shows persistent longitudinal XY lag, not Z, attitude, or
  cross-track failure. In the high-error intervals the velocity reference is
  already capped at `-0.2 m/s`, while achieved root speed is typically
  `-0.12..-0.14 m/s`. More balance-loop Kp is therefore not the next lever.
  The next plan must use closed-loop plant authority and acceleration/reversal
  constraints in its execution retiming while preserving exact-source anchors.
- The run also exposed an independent evidence bug: a configured zero-capable
  phase governor was treated as invalid unless that particular execution
  actually produced a zero-scale sample. Corrected validation to accept either
  consistent hold activation or consistent non-activation. This changes no
  command, gate threshold, source, plan, or physical result; the sealed case-42
  dynamic reject remains a reject.

## Next round after Round 116

- Pass focused and authoritative CPU tests and commit the evidence-only repair.
- CPU-only, derive and validate one case-42 plan variant using plant-qualified
  closed-loop velocity plus acceleration/reversal retiming. Preserve source
  anchors and both clocks, and pass duration/path/transition/kinematic gates.
- Do not launch Isaac again until the derived plan and a fresh authorization
  have been independently audited. Keep case 43, residual capture, BC, PPO,
  training, and obstacle work closed.

## Round 117: ratio-2 retime fails closed on initialization preservation

- The corrected evidence summarizer resealed the unchanged case-42 runtime to
  a separate file at SHA-256
  `61afbac7ac8c3f2cac8ff51dd8452c07da2add4bab16692c0304c19629c12ffc`.
  Physical dynamic quality remains false, while runtime, controller, thermal,
  zero-capable governor, and exact `vx_kp=0.72` evidence now pass. The original
  summary remains preserved.
- A CPU-only uniform-retime derivation used the maximum admitted
  execution/source ratio `2.0`. It produced `64.907202 s` execution duration,
  kept all exact-source geometry and timestamps, never sped an interval up,
  recomputed feedforward, and passed all timing, path, transition, and kinematic
  gates. Prospective portfolio median remained bounded at `1.499904`.
- The candidate correctly remained invalid because the generic retime save path
  replaced the separate v20 initialization arrays with empty arrays. The failed
  namespace is preserved as
  `20260720_smoothed_plan_case42_v21_plant_capacity_uniform_ratio2_cpu`;
  plan/manifest/summary SHA-256 values are
  `bb218af7ae584535de8533846ba0d9c1474186b65ca60673aa3904d037a11f0d`,
  `4f9e2955bda0a8bf815df09c13752f3d435882d50c4ead88665eb9fa10b02c9a`,
  and `67b80c196cbd60ba3e3a50f8e467378f3be056dfa01afec9b8abfeb43557b849`.
- Added an explicit derivation step that copies the unscored initialization
  clock/state verbatim after execution retiming, plus a regression proving the
  execution clock remains changed while initialization remains exact. This is
  an artifact-preservation repair, not a controller, source, plan-geometry,
  governor, threshold, or learning change.

## Next round after Round 117

- Pass focused and authoritative CPU tests, commit, push, and transfer the
  initialization-preservation repair.
- Derive a fresh v22 ratio-2 candidate and require every source, initialization,
  execution-clock, feedforward, timing, path, transition, and kinematic check to
  pass. Stop before any runtime authorization or Isaac launch.

## Round 118: v22 preserves arrays; audit requires admitted-parent semantics

- Commit `d80e59bd6bcbf3187b600a32680a0ed80eb390a9` was pushed and
  transferred with bundle SHA-256
  `63acdbe8f2c9de58627f59b89b772c8c3d1e8d8cd5ddbb94e83ac586f7bd6242`.
  The authoritative `.98` suite passed `423/423` in `28.36 s`
  (`30.45 s` command wall time).
- Fresh v22 successfully preserved both initialization arrays, and all source,
  execution-clock, feedforward, path, transition, and kinematic checks passed.
  It still failed closed because the generic static audit always requires empty
  initialization, even when the hash-bound admitted parent intentionally owns
  a separate unscored pre-roll. The v22 plan/manifest/summary SHA-256 values are
  `dc82e3f8dd04e20b28f51bf649bf78bfef5f55dcb303b26c80b62bd425826028`,
  `981d20aaa0354640df3287d96b5db58ea67b4039590055399245792bf0875cbf`,
  and `6c7de9a50671a8945797b28c025ad63efa817cc51b645f22e9f373ecb5053372`.
- The output also revealed that the generic save path did not carry the parent
  `initialization_preroll` metadata even though the arrays were restored. The
  repair now preserves that metadata exactly and replaces only the retime
  audit's empty-initialization assertion with parent-preservation checks. The
  global static audit remains unchanged for newly exported plans.

## Next round after Round 118

- Pass focused and authoritative CPU tests, commit, push, and transfer the
  parent-semantics audit repair.
- Derive a fresh v23 ratio-2 candidate. Require all checks true and stop at the
  CPU artifact boundary; do not issue authorization or launch Isaac.

## Round 119: initialization goal changed from 70 to 40 teachers

- The user changed the immediate learning objective: a bounded initial policy
  may use already dynamically accepted trajectories and no longer needs to wait
  for 70 qualified cases. The 70-case milestone is now later coverage evidence;
  79/79 remains the final evaluation goal.
- The demonstrated physical dynamic union is 42 cases:
  `[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,30,31,32,33,34,36,37,41,52,53,66,67,68,70,74,77]`.
  Historical Gate-C runs selected these plans but intentionally wrote no
  residual datasets, so they cannot be merged directly into BC input.
- CPU evidence inspection found 42 physical passes but only 39 passes under the
  provisional `[0.30,0.40,0.10]` residual-label scale. Cases 10, 28, and 70
  overflow only the old label normalization. They remain valid physical teacher
  candidates; clipping or misreporting 39 as 40 is forbidden.
- Added a fail-closed candidate auditor that binds each selection to the exact
  source manifest, v16 plan hash, admission, gate, summary, dynamic/thermal/
  runtime result, zero applied residual, and no-dataset/no-training evidence.
  Its output remains `valid_for_training=false` and requires a fresh homogeneous
  capture of at least 40 cases.
- Added the initialization split minimum `30 train / 5 validation / 5 holdout`.
  BC remains closed until fresh capture, raw-envelope recomputation, zero
  clipping, command reconstruction, and case-disjoint split gates pass. PPO
  remains closed after BC initialization.
- The authoritative `.98` CPU audit selected all `42/42` requested physical
  teacher candidates from the v16 portfolio with no missing case. The selection
  manifest is
  `20260720_initial_teacher42_selection_v1/selection.json`, SHA-256
  `e0f1d2b44061aabfe64ad2ffa3d23f57bf9b3e51015b2e3fa0703ba24316bb06`.
  It reports raw residual maxima `[0.319627,0.216497,0.017296]`, 39 passes under
  the legacy scale, and legacy-scale rejects `[10,28,70]`. Selection passes,
  while capture, BC, PPO, and training admission remain false.
- The authoritative `.98` regression suite passed all `427/427` collected
  tests after the per-case legacy-schema compatibility and modern-evidence
  preference fixes. No GPU/Isaac application was launched.

## Next round after Round 119

- Run the CPU candidate auditor on `.98` and seal the 42-case selection manifest.
- Implement scale-independent raw-command capture and prove it on a small
  accepted canary before scheduling the 40-plus-case homogeneous recapture.
- Do not alter the deterministic LQR/safety supervisor, launch PPO, or train
  from historical Gate-C traces.

## Round 120: scale-independent raw capture canary passes

- Added `cinebotrl_two_wheel_riser_executed_raw_teacher_v1`. Raw artifacts hold
  executed observations, baseline wheel actions, teacher commands, both clocks,
  and unnormalized residual commands. They explicitly freeze no action scale,
  apply no residual, and remain `valid_for_training=false`.
- Added a dedicated auditor and guarded case-2 route. The route pins the exact
  source, v16 portfolio, case plan, selection, gains, robot USD, code identities,
  clean pushed HEAD, and exclusive WSL/Windows GPU ownership. Normalized dataset
  capture, policy rollout, scale freeze, BC, PPO, and training are all closed.
- Commits `a35b5eb`, `6107129`, `2fd232f`, `ca24fb3`, and `a3ca6f2` were pushed.
  The authoritative `.98` suite reached `434/434` passes before the canary;
  subsequent route-focused tests passed `31/31`.
- A v1 route attempt failed before Isaac because the immutable portfolio's
  original 70-plan admission threshold was incorrectly passed as 40. This did
  not change the downstream teacher threshold. A v2 attempt then exposed a
  Bash/Windows path escaping defect and was preserved fail-closed. Neither
  failed attempt produced eligible training data.
- The corrected v2 physical run at commit `ca24fb3` completed case 2 with
  `9,199` policy steps over `18.241928 s` execution time (`9.439314 s` source).
  Position p95/max were `0.139830/0.153514 m`, attitude p95/max were
  `0.137239/0.175777 deg`, and peak pitch was `6.632458 deg`. Dynamic, thermal,
  and controller-evidence gates all passed with no termination.
- The raw NPZ SHA-256 is
  `ffcdd55e946225cf5e3c976013da94dadbf400c6179af4e95a266a10feaae8a8`.
  Raw residual maxima were `[0.245937,0.145314,0.012289]`, applied residual was
  exactly zero, and reconstruction error was `2.98e-8`.
- WSL Python lacked NumPy and the first post-run auditor looked for
  `completed_reference` at the wrong JSON level. Commit `a3ca6f2` repaired only
  this CPU evidence route; Isaac was not rerun. The posthoc audit SHA-256 is
  `43dea4e7e1bbecde9aec3d80b1cac8adf203b51e8394bf75fe5f4297969ee757`,
  and final-status-v2 SHA-256 is
  `125cd09b7dd7eda762c1fce20a2aba96f9ce3345595145c2c93c08619e57eced`.
- The canary passes raw-capture admission but does not freeze scales or admit
  training. BC and PPO remain disabled.

## Next round after Round 120

- Implement CPU-tested corpus admission and raw-to-normalized relabel tooling.
  Previous-action observation channels must be rebuilt from the previous
  normalized teacher action; zero placeholders may not enter BC.
- Produce a deterministic case-disjoint `30 train / 5 validation / 5 holdout`
  split only after at least 40 fresh physical raw captures pass.
- Prepare, hash-bind, and review a fail-fast sequential capture route. Do not
  launch the long batch, BC, PPO, or obstacle work until that route and corpus
  admission tests pass.

## Round 121: corpus admission and relabel tooling complete

- Commit `2a1b5c54c09bb3629fa66edfe2a0be0f1035a808` adds the CPU-only
  raw-corpus admission and dataset-construction boundary. The authoritative
  `.98` suite passed `440/440` tests in `31.36 s`.
- Corpus admission requires at least 40 exact case/gate/raw triples, exact
  selection and plan hashes, completed physical reference, dynamic/thermal/
  controller-evidence passes, no termination, zero applied residual, no
  normalized runtime dataset, matching row counts, and command reconstruction
  error at most `2e-6`.
- Frozen scales use corpus-wide raw absolute maxima, a `1.10` margin, and
  per-channel `0.05` quantization with minimum `[0.30,0.40,0.10]`. No clipping
  or equality at the normalized action bound is admitted.
- Raw-to-normalized conversion now replaces the raw artifact's zero
  previous-action placeholders: row zero remains zero and every later row uses
  the preceding normalized teacher action. This prevents a train/rollout
  observation-contract mismatch.
- The builder deterministically selects exactly 40 admitted cases and emits
  `30 train / 5 validation / 5 holdout`; any additional captured cases remain
  coverage-only. It rechecks source hashes, trajectory separation, normalized
  bounds, prior-action recurrence, and teacher-command reconstruction.
- A passed dataset is marked `valid_for_bc_initialization=true`, but
  `bc_authorized=false`, `ppo_authorized=false`, and `training_started=false`.
  No long capture, BC, PPO, or learned rollout was launched in this round.

## Next round after Round 121

- Implement a fresh, hash-bound, sequential capture scheduler over the sealed
  42-case selection. It must support resume without overwrite, preserve every
  case log and exit code, stop on the first physical reject, and generate one
  corpus admission only after at least 40 cases pass.
- Estimate the total runtime from the selected execution durations before
  issuing the one-use batch authorization. Do not let a label-scale result stop
  deterministic physics.
- After capture, run corpus admission and construct the `30/5/5` dataset. Stop
  again for review before any bounded BC process starts.

## Round 122: previous-action exposure bias diagnosed; masked BC is a near-pass

- The fresh 40-case corpus, case-disjoint `30 train / 5 validation / 5 holdout`
  split, bounded BC initialization, and rendered case-4 rollout were completed
  under the existing no-PPO boundary. Holdout remained unopened. The original
  policy completed case 4 but missed the unchanged position gate at
  `0.181195/0.221956 m` p95/max.
- Commit `6f5940c2fb19bfcfad1472a6bfb4bbf17db2cf8d` added a CPU-only
  teacher-forcing diagnosis. On immutable case-4 teacher states, the original
  policy's MSE ratio to a zero predictor was `0.030837`; recursively replacing
  only observation indices `23:26` with the policy's own previous output raised
  the ratio to `0.549984`. Teacher-state action correlations remained
  `0.965/0.989/0.993`. The bounded classification is therefore
  `autoregressive_previous_action_exposure_bias`, not insufficient trajectory
  coverage or a failed deterministic controller.
- The diagnosis artifact is
  `20260721_initial_teacher40_bc_case4_cpu_diagnosis_v1`; its report SHA-256 is
  `9c81451eeb4549da1504f4ae6baa141aa67aa415ababb481c9b37e202917766c`.
  Lever-arm saturation also occurs in successful teacher playback, so it is an
  operating condition and possible error amplifier, not the demonstrated root
  cause.
- Commit `7932a9efc35b99e5b87c3a7e8eb653647fce471b` added the
  `state_shared_lookahead_fusion_previous_action_masked_v1` architecture. It
  masks normalized previous-action inputs inside the model, records the mask in
  the TorchScript/evidence contract, and passed `52/52` focused CPU tests.
- The masked BC run completed at best epoch `32` after `42` epochs and passed
  aggregate offline validation. Its final-status SHA-256 is
  `ded00f25dde299207dc0e3af0b611418e09d5368d4fc9e7cab53b57df9a36bba`,
  report SHA-256 is
  `3f0efb4a2707b343a775dd5dd8b0ad49d6506474da627d8449ca81556cbbcd3e`,
  and TorchScript SHA-256 is
  `34fa67192f8c66b879eb7d11a83c96ffd2320932e6807f2224cdfa2f74a4c0e4`.
- Commit `a1a5e9394db1f372669eef4f2f4a7b939021d3cb` added the guarded,
  single-case learned-playback route. The exclusive case-4 canary completed all
  `21.514453 s` of execution in `6,316` policy steps with dynamic, thermal, and
  controller-evidence gates true. Position p95/max improved to
  `0.137441/0.167322 m`, a `24.1%` p95 improvement over the original BC policy;
  attitude p95/max were `0.184099/0.214760 deg`, and pitch max was
  `6.160219 deg`.
- The masked policy now passes the absolute `0.15 m` dynamic p95 gate, but it is
  not admitted. It misses the unchanged teacher-plus-5-percent p95 budget
  (`0.135173 m`) by `0.002268 m` and also misses the corresponding max-error
  budget by about `0.008012 m`. Gate-summary SHA-256 is
  `ff483e5ee8b975419fc75efbdf7c22e013a5c5dccc56df17e538d9113d35abdc`;
  final-status SHA-256 is
  `1a4fbcd16fa3490d9b187b4af90298c8c04f7d674060189ae8116cd500257cdb`.
  No holdout metric, residual capture, PPO process, or obstacle curriculum was
  opened.

## Next round after Round 122

- Keep the masked checkpoint as a diagnostic near-pass, not an admitted policy.
  Do not relax either the absolute or teacher-relative gates.
- Implement bounded sequence-aware BC that exposes training to policy-generated
  previous actions using a deterministic scheduled-sampling contract. Preserve
  case-disjoint validation, teacher actions, scales, controller commands, and
  all safety gates.
- First compare the candidate offline against the original and masked policies,
  including recursive previous-action diagnostics. Only an offline improvement
  may receive one exclusive case-4 canary. Holdout, PPO, obstacle work, and
  broader learned rollout remain closed until that canary passes unchanged
  absolute and teacher-relative gates.

## Round 123: action-history alternatives fail offline; masked policy remains best

- Commit `8e2b58224571d216a895bf26fab8a1ef886067b1` added deterministic
  sequence-window scheduled sampling, recursive validation model selection, and
  fail-closed case isolation. Commit `a3548ca929050e51843d6a0e668df04b80590ef3`
  added a validation-only comparator against the original and masked policies.
  The `.98` focused suite passed `59/59` before training.
- The scheduled candidate used 32-row windows, a five-epoch warmup, a 25-epoch
  ramp to fully policy-generated previous actions, and `0.75` recursive
  validation weight. Training passed its teacher and bounded-recursive
  validation gates at best epoch `50` and stopped after `60` epochs, but the
  full-recursive case-4 comparator rejected it. Aggregate MSE was `0.004731`,
  versus `0.008875` original and `0.003492` masked. No learned playback began.
  Final-status SHA-256 is
  `abab8f8726dc0ac545ae3f45f147e3e10e590ef507682ebe5197d0d80b5f447e`;
  comparison SHA-256 is
  `1173da5c72ca3fe0d2f75059a0571af6333b6f1d8bf36ed93f4229f8d1571009`.
- Commit `5401fa3f31ea7416595b87bf4541b5f685a7007a` added a fixed normalized
  previous-action gain, and commits
  `25aa5c9640de78db9a44e3027b405af49ed80dbc` and
  `c22d4725afa30969a18ecc97faceed5762c2ebcc` added its guarded route.
  The scalar `0.10` candidate passed row-wise validation at best epoch `76`, but
  full-recursive case-4 aggregate MSE was `0.003978`, still worse than masking.
  It improved longitudinal and riser MSE over masking (`0.004909` vs `0.006160`,
  `1.44e-5` vs `1.97e-5`) but degraded yaw (`0.007010` vs `0.004297`).
  Final-status SHA-256 is
  `628380b9f9ae8f7a837ccea30906bfb5f9f3518d01124ea6dd4eb92e40f685bf`;
  comparison SHA-256 is
  `f2dfbfb631890a477ab31a705759ee4f27bce6dbd3183771d9b89e7c98ee9949`.
- The channel evidence justified one final non-sweep candidate. Commit
  `ae30ea6ad2fcc476f5c2d420382e9625568fc5d9` added per-channel gains, and
  commit `cd236182abf2e087dacaf43ac5280f879f55230a` added the guarded
  `[0.10,0.00,0.10]` route. The `.98` suite passed `67/67` before training.
  This candidate stopped at best epoch `29` after `39` epochs. Its recursive
  case-4 aggregate MSE was `0.003836`: yaw and riser slightly beat masking, but
  longitudinal MSE regressed to `0.007404` and teacher-state aggregate MSE also
  became worse than masking. Final-status SHA-256 is
  `4e1190078383c94c987540c70790c27ef953af3995360fe131e8874c18455aa6`;
  comparison SHA-256 is
  `8699a85753e5a912fb2c72598c50159b297f6980bc914ccf332351560e224db3`.
- All three alternatives were rejected before Isaac. The masked policy remains
  the best demonstrated learned playback: absolute position p95 passes at
  `0.137441 m`, but its unchanged teacher-relative gate still misses by
  `0.002268 m`. Holdout, PPO, obstacle work, and broader rollout remain closed.
  GPU/process ownership was clean after the final offline comparison.

## Next round after Round 123

- Stop tuning previous-action architecture or gains. The evidence now shows that
  no tested action-history dependence beats full masking recursively.
- Perform a CPU-first physical-state covariate-shift diagnosis using the masked
  case-4 teacher and learned-rollout traces. Localize which non-action
  observations diverge before the tracking-error peak and distinguish base,
  riser, camera, lookahead, and balance-state contributions.
- If the traces prove a state-distribution gap, prepare one bounded
  validation-only DAgger-style teacher relabel capture on policy-visited states.
  Keep the deterministic LQR/safety supervisor unchanged, preserve the original
  40-case corpus, and do not open holdout, PPO, obstacles, or broader rollout.

## Round 124: full-rate trace narrows the gap but does not prove one precursor

- Commits `9fd4cf3` and `a24d8bb` first compared the existing 1 Hz teacher and
  masked-policy traces. They found base state to be the only coarse non-output
  precursor candidate, but only 32/33 rows were available and the exact 65D
  policy observation, lookahead, and policy-rate action were absent. The v2
  coarse report SHA-256 is
  `177d431dc8278513fe484540a79440df53e26ffb79eff37af32b17c5a8f7e0af`.
- Commit `31d10059ba20e8d1ba41290f327645e6885caabd` added a default-off,
  non-trainable `policy_trace_v1` contract and one-use case-4 route. It records
  the exact pre-action 65D observation, applied residual, final high-level
  command, balance-wheel command, both clocks, and post-step physical outcome.
  It hard-codes `valid_for_training=false`, contains no teacher label or
  residual dataset, and leaves DAgger, BC, and PPO unauthorized. The complete
  `.98` CPU suite passed `487/487` before runtime.
- The exclusive trace route produced 6,172 deterministic-teacher rows and
  6,316 masked-policy rows. Teacher position p95/max were
  `0.128736/0.151724 m`; masked-policy p95/max reproduced at
  `0.137441/0.167322 m`. Both runs completed the immutable execution phase and
  passed dynamic, thermal, and controller-evidence gates. Trace SHA-256 values
  are `1999de74dc1206c4a3aed04b375bbafe5a81755f9424df5d1dee333406cdc9bb`
  for teacher and
  `99c5969305e20cdc73ad2750d9609ed62c8094ed5b626d5ba2ebe6aa374f4ef7`
  for learned.
- The route's first final status failed closed because successful learned
  playback left a shell sentinel at `99`. Commit
  `f190c10d7a23475328379328f9880172685117dc` clears the sentinel before launch.
  No Isaac rerun was performed. A posthoc audit resealed the existing JSON,
  NPZ, and logs at final-status-v2 SHA-256
  `d2dcde485d88c1483259f8b1bd434828784826a29e91c914a65bbca3ce031f8d`.
- Commit `557cddc15b9606fa28f61d2826c20c5714e6436f` added the reproducible
  policy-normalized diagnosis. Its `.98` focused suite passed `53/53`; report
  SHA-256 is
  `9e64301f49ce1a3b7ff7d41c2228c4be6cecb0514fb1fdb271fbf322515af945`.
  The learned excess position error is `0.035086 m` p95 and `0.038592 m` max.
  The strongest non-output association is feed-forward divergence at only
  `r=0.127`; the strongest positive signal is the outcome-coupled `+0.25 s`
  camera-error lookahead at `r=0.679`. Pitch-rate and camera-attitude shifts are
  larger in magnitude but weakly or negatively associated with excess error.
- Phase-aligned learned-vs-teacher normalized action-error p95 is
  `[0.068622,0.162376,0.011600]` for `[vx,wz,riser]`, but every magnitude
  correlation with excess error is negative. The bounded classification is
  therefore `no_single_non_output_policy_input_precursor_proven`, not a proven
  base-only or action-history failure. No holdout, DAgger capture, BC, PPO,
  obstacle work, or broader learned rollout was opened. GPU ownership is empty.

## Next round after Round 124

- Do not tune another BC architecture or infer causality from magnitude-only
  shifts. Preserve the masked checkpoint as the best diagnostic near-pass.
- Design one separately authorized, validation-only case-4 shadow-teacher
  measurement. During the learned rollout, record the deterministic
  pre-overwrite `[vx,wz,riser]` teacher command already computed on the same
  policy-visited state, but do not apply it, create a training dataset, or alter
  the learned command.
- Compare the on-policy shadow-teacher labels with applied residuals and the
  original phase-aligned labels. Only a clear, bounded on-policy label gap may
  justify a later DAgger dataset proposal. Holdout, BC, PPO, obstacles, and
  broader rollout remain closed.

## Round 125: case-4 shadow teacher proves an on-policy label gap

- Commit `a5b511287f439dbaa72982afae9f24fd85394a4b` adds a separate
  `shadow_teacher_trace_v1` contract. During learned playback it records the
  deterministic teacher command already computed on the policy-visited state
  before policy overwrite, plus the actually applied policy command and
  post-step physical outcome. Runtime scales are explicitly
  `[0.35,0.40,0.10]`; raw/normalized teacher labels and both high-level command
  paths must reconstruct independently.
- The contract is measurement-only: shadow commands are never applied, the
  artifact is not a residual dataset, and labels are explicitly unadmitted for
  training. The route is validation case 4 only, hash-bound, single-use, and
  guarded across WSL, Windows, and NVIDIA ownership. The complete `.98` CPU
  suite passed `496/496` in `45.18 s` before runtime.
- The exclusive learned rollout reproduced the prior case-4 baseline exactly:
  all five pinned position, attitude, and pitch metrics match within `1e-9`,
  all 6,316 policy steps were recorded, dynamic/thermal/controller gates pass,
  and applied commands reconstruct from policy outputs without any shadow
  teacher contribution. Learned-gate SHA-256 is
  `fe5754bf20e20b608e205f7179e1c7a92f079617e8349949abab52b8470c4b46`;
  shadow-trace SHA-256 is
  `ccaf6b5f75df9f185e022f192af6771c6c9db12ed6a3de9feed470021a6c0c7a`.
- The on-policy diagnosis is positive. Policy-to-shadow normalized action-error
  p95 is `[0.166973,0.161318,0.011368]` for `[vx,wz,riser]`, compared with
  `[0.068622,0.162376,0.011600]` against the original phase-aligned teacher.
  On-policy shadow-label shift from the original teacher has p95
  `[0.165576,0.097817,0.012356]`; `vx` and `wz` cross the fixed material-shift
  thresholds while riser does not. Aggregate policy-to-shadow RMSE is
  `0.070146`, versus `0.043487` against phase labels, a `1.613039x` gap.
- The bounded classification is
  `on_policy_teacher_gap_supports_bounded_dagger_proposal`. Diagnosis SHA-256
  is `6580b0a71cdc3d518318ea19d03a34da61e1cb12c9f52e5a5a960e6522d9ab79`;
  final-status SHA-256 is
  `28c92a7c321cf165908e9d66bce80ebe8d77cca748358526894e4dff6c372f2a`.
  GPU/process ownership is empty after closure.
- This does not make case 4 trainable. Case 4 belongs to validation, so its
  shadow labels remain diagnostic evidence and cannot enter BC without
  violating the frozen split. No dataset, DAgger training corpus, BC, PPO,
  holdout metric, obstacle work, or broader learned rollout was opened.

## Next round after Round 125

- Build a CPU-only similarity ranking over the existing training split to find
  one or a very small number of training cases that exercise case-4-like
  teacher actions and physical-state regimes. Do not use validation case 4 or
  any holdout case as training data.
- Review a separately hash-bound proposal for shadow-teacher measurements on
  those selected training cases. Each must remain one-case, unapplied,
  non-trainable evidence until its runtime and label-gap audit passes.
- Only after training-split evidence confirms the same on-policy gap may a new,
  explicitly versioned DAgger dataset admission be proposed. BC retraining,
  PPO, holdout evaluation, obstacles, and broad rollout remain closed.

## Round 126: training-split DAgger shadow proposal selects case 21

- Commit `4219768f76bae720d1973b6088af1987a57f8a67` adds a deterministic,
  CPU-only similarity ranker and proposal builder. Focused `.98` tests pass
  `64/64`; no Isaac or GPU process was launched.
- Ranking compares normalized teacher-action quantiles, masked-policy-effective
  observation means and standard deviations, and execution duration. The
  frozen weights are `0.40/0.25/0.25/0.10`; previous-action channels remain
  masked and do not affect ranking. Ranking SHA-256 is
  `6019780884c6c4d9279742d918d99f32e426c737b6938735a46e46c100cf12db`.
- The nearest training cases to validation case 4 are `[21,30,31]`. Case 21 is
  rank 1 with score `0.369645`, 5,486 samples, and `27.425 s` duration. Cases
  30 and 31 score `0.396646/0.398475` but are both about `57.5 s`; the bounded
  first proposal therefore selects case 21 only.
- The proposal proves case 21 belongs to training and that its existing
  deterministic teacher gate passes dynamic, thermal, and controller-evidence
  checks. Its plan SHA-256 is
  `81c0da4be22d5b800978d1d46ca9705912f72007f7c615b31715c672dd86a1d4`;
  teacher-gate SHA-256 is
  `35a72e768d12d162d7522d15500434268d020e90d58657fe27cef2c19d7068de`.
- Validation cases `[4,8,16,22,32]` and holdout cases `[3,5,13,19,24]` are
  explicitly excluded. Proposal SHA-256 is
  `5bc65581cda29cca7668a16d40d77df3cda717b43f87bc2744d4676dd0d3ef7d`.
  It issues no authorization token, starts no runtime, creates no dataset, and
  leaves DAgger, BC, PPO, holdout, obstacles, and broad rollout closed.

## Next round after Round 126

- Review and, only if accepted, add one fresh hash-bound case-21
  shadow-teacher route using the unchanged masked policy, plan, action scales,
  controller, and safety/quality gates. It must remain a measurement, not a
  DAgger dataset.
- Stop after case 21. If physical dynamics pass and the on-policy label-gap
  audit reproduces the material `vx/wz` shift, prepare a separate dataset
  admission proposal. If either condition fails, preserve the first reject and
  do not open cases 30/31.
- Do not use case-4 validation labels for training and do not open holdout,
  BC retraining, PPO, obstacle work, or broader learned rollout.

## Round 127: case-21 shadow measurement passes physics but rejects DAgger admission

- Commit `76260c518fda3aeedbbd767130027754b95e61d4` adds the fresh,
  hash-bound case-21 route. It pins the Round-126 proposal and ranking, the
  case-21 plan and deterministic teacher gate, the unchanged masked policy,
  action scales, playback, diagnosis, and LQR gains. The authoritative `.98`
  CPU suite passed `503/503` in `45.98 s`; the exclusive preflight passed
  before the one-use token was transferred and consumed.
- The learned policy completed all 5,486 steps. Dynamic quality, thermal
  admission, controller evidence, trace reconstruction, and every no-learning
  check pass. Position p95/max are `0.101943/0.107746 m`, attitude p95/max are
  `0.191831/0.267455 deg`, and pitch max is `6.360049 deg`. Relative to the
  deterministic teacher, position p95/max improve by `0.000763/0.000475 m`;
  attitude p95 improves by `0.000971 deg`, while attitude max and pitch max
  increase by only `0.052254/0.059543 deg` and remain within unchanged gates.
- The 5,486-row shadow trace is measurement-only and reconstructs the applied
  commands exactly. Shadow commands were not applied, labels were not admitted,
  and no residual dataset, raw-teacher capture, policy trace, BC, DAgger, or PPO
  run was created. Final-status SHA-256 is
  `285531d78f31755e3dec816b46266ce6261d4b3811b2c4c0a6572e6bd7bcdef7`;
  trace SHA-256 is
  `79baf7d5000ced7da2dd2b5295f5e732dbe6dde2b5be1fba66d9f3f804d7028d`.
- Unlike validation case 4, case 21 does not reproduce a material on-policy
  label shift. Material `[vx,wz,riser]` flags are `[false,false,false]`.
  Policy-to-shadow p95 is `[0.013513,0.013489,0.001259]`; shadow-to-original
  phase-label p95 is `[0.008312,0.010334,0.001941]`. Aggregate policy-shadow
  RMSE is `0.005507`, below policy-phase RMSE `0.006448`, for a `0.854003x`
  ratio. Diagnosis SHA-256 is
  `b5fd73518c3df537efc68770af702f6cc118c6adb750a2a6a141e7f7eb1228e6`.
- The fail-closed classification is
  `on_policy_teacher_gap_does_not_yet_support_dagger_proposal`. Therefore case
  21 is not admitted as a DAgger seed, cases 30/31 remain unopened, and no
  training stage advances. The one-use token is absent and WSL, Windows, and
  NVIDIA ownership are empty after closure.

## Next round after Round 127

- Do not broaden the case-4 finding into a training-data claim. The current
  offline similarity score did not select a training case that reproduces the
  case-4 on-policy state-distribution shift.
- Perform a CPU-only feature audit comparing the actual case-4 shadow-shift
  intervals against all existing training trajectories. Rank candidates using
  the state/action regions where the shadow-vs-phase label difference occurs,
  rather than whole-trajectory aggregate means, standard deviations, and
  duration. This is a new proposal only and must not authorize runtime.
- Keep cases 30/31, holdout, dataset creation, BC retraining, PPO, obstacles,
  and broad learned rollout closed until that localized audit identifies and
  justifies one new bounded training-split measurement.

## Round 128: localized coverage audit finds no justified training-case canary

- Commit `84efc1a` adds a reproducible CPU-only localized ranker with synthetic
  positive and fail-closed coverage tests. The focused `.98` suite passes
  `50/50`; no Isaac/GPU process, token, runtime namespace, dataset, or training
  process was created.
- The audit first isolates case-4 rows where absolute shadow-minus-original
  phase labels exceed `[0.05,0.05,0.02]` for `[vx,wz,riser]`. This identifies
  4,682 of 6,316 rows (`74.1292%`), with per-channel counts
  `[4128,1860,29]`. It then compares 256 deterministic samples from this
  region with at most 2,048 samples per training case using a directed nearest
  distance over masked-policy-normalized state (`0.75` weight) and normalized
  phase-teacher action (`0.25` weight). Previous-action channels remain
  excluded.
- Case 4's own offline teacher trajectory is the calibration reference, with
  score `0.256960`. The nearest training cases are 18, 30, 31, 41, and 14.
  Their top three scores are `0.936081/0.941425/0.946083`, corresponding to
  reference ratios `3.642897/3.663695/3.681822`. Case 21 ranks ninth at
  `4.862837x`, consistent with its negative live shadow-label result.
- The fixed admission gate requires a candidate score no more than `1.5x` the
  case-4 calibration. No training case passes, so
  `coverage_admission_passed=false`, `proposed_runtime_cases=[]`, and the
  classification is `no_training_case_covers_case4_shadow_shift_region`.
  Report SHA-256 is
  `1d4cac1cc1d7eb1f52f56b3f5188df836ccb62527a72e2a8345d30ba75d650bf`.
- This supersedes the whole-trajectory ranking as a runtime-admission method.
  It does not invalidate case 21's clean physical pass, but it proves that
  opening cases 18/30/31 solely because they are nearest would still be blind
  exploration. No case is authorized and all learning gates remain closed.

## Next round after Round 128

- Build a CPU-only coverage-expansion audit over exact-source trajectories not
  present in the current 30-case training split. Use immutable source/plan
  identities and case-4 hotspot target-command features to determine whether a
  geometrically distinct accepted trajectory can cover the missing region.
- If no unused accepted source passes the same calibrated coverage principle,
  prepare a separate proposal for either a newly designed training trajectory
  or a transparent split reset. Do not silently move case 4 into training or
  reuse its validation labels.
- Keep GPU runtime, DAgger capture, BC retraining, PPO, holdout, obstacles, and
  broad rollout closed until a coverage-expansion proposal is independently
  reviewable and selects exactly one bounded next measurement.

## Round 129: unused exact-source plans do not close the missing coverage

- Commit `6939ba0` adds a CPU-only plan-command coverage audit. Focused `.98`
  tests pass `49/49`. It verifies all 79 v16 plan files, the sealed
  manifest/summary, 70 timing/transition/kinematic passes, 40 current dataset
  cases, 30 existing training cases, and 30 unused admitted cases. No Isaac,
  token, runtime namespace, dataset, or training process was created.
- The unused admitted pool is
  `[20,42,43,44,46,47,48,49,50,51,54,55,56,57,58,59,60,61,62,63,64,65,69,72,73,75,76,77,78,79]`.
  Current train, validation, and holdout cases are excluded. The nine v16
  kinematic rejects `[1,27,29,35,38,39,40,45,71]` are also excluded.
- The comparison uses 42 nominal plan features: current
  `[vx,wz,riser-rate]` plus `0.25/0.50/1.00 s` lookahead base, camera,
  attitude, riser, and feed-forward deltas. Features are normalized over all
  70 admitted plans. Target times are the same 256 deterministic samples from
  case 4's 4,682-row shadow-shift region; no simulated-state or dynamic-pass
  claim is inferred from plan similarity.
- Existing training case 30 is the strongest command-region baseline with
  score `0.831104`; cases 31 and 18 follow at `0.839955/0.904083`. Best unused
  case 78 scores `1.021636`, followed by case 20 at `1.022422` and case 64 at
  `1.092030`. Case 78 is therefore worse than the existing baseline, with
  unused-to-existing ratio `1.229252` against the fixed `<=0.80` material
  improvement gate.
- The fail-closed result is
  `no_unused_admitted_plan_materially_improves_command_coverage`, with
  `proposed_shadow_measurement_cases=[]`. Report SHA-256 is
  `9af02fb895d23a3fea0bf76bf41db219c65a172bd987b90f31c7deebd1ed37d4`.
  Case 78 and every other unused plan remain runtime-disabled; case-4 labels,
  holdout, DAgger, BC, and PPO remain closed.

## Next round after Round 129

- Prepare a CPU-only architecture proposal comparing two honest ways to add
  the missing state coverage: a transparent split reset that permanently
  retires case 4 as validation before admitting its on-policy shadow labels,
  versus controlled perturbation of an existing training trajectory with
  separately generated on-policy teacher labels.
- Preserve holdout `[3,5,13,19,24]` unchanged and unopened. A split reset must
  nominate a fresh, dynamically qualified validation replacement before case
  4 can enter training; it must never continue reporting case-4 validation
  metrics after that change.
- Do not authorize case 78, a perturbation canary, dataset creation, BC, PPO,
  obstacles, or broad rollout until the proposal defines provenance, leakage
  controls, dynamic gates, and one bounded first measurement.

## Round 130: coverage-recovery architecture selects controlled perturbation

- Commit `f0243bf3c31da4077e5cd5ce824633dcbecef373` adds a fail-closed,
  CPU-only architecture proposal generator and focused negative tests. The
  authoritative `.98` coverage/DAgger suite passes `22/22`; no Isaac, GPU,
  runtime namespace, authorization token, dataset, or training process was
  created.
- The proposal binds the frozen `30/5/5` split, case-4 and case-21 shadow
  measurements, the localized state/action audit, the all-79 nominal-plan
  coverage audit, and the shared teacher-dataset identity. Every input contract
  check passes. Holdout `[3,5,13,19,24]` remains unchanged and unopened.
- Controlled perturbation is ranked first. Case 30 is the sole proposed first
  bounded measurement because it is the nearest existing training case in the
  nominal command/lookahead audit. The missing state deviation would be added
  by one deterministic horizontal wrench pulse, disabled by default, without
  altering plan geometry/timing, teacher commands, or learned commands. Pulse
  values remain intentionally unset pending a separate CPU review.
- A transparent split reset is retained only as a fallback. It would
  permanently retire case 4 from validation, add it to training, and nominate
  case 78 as the replacement validation candidate. This split is not applied:
  case 78 currently has kinematic plan admission only and would first require
  deterministic dynamic qualification, a new immutable split manifest, and a
  source-tagged/weighted dataset schema.
- The proposal explicitly keeps case-4 shadow labels unadmitted and records
  `case4_split_changed=false`, `case78_validation_admitted=false`,
  `runtime_authorized=false`, `dataset_created=false`,
  `dagger_authorized=false`, `bc_authorized=false`, and
  `ppo_authorized=false`. Proposal SHA-256 is
  `91e50fdf8ec246b86767b5dedc069e20f70cc22e5ff57e91c051a3d1dd197146`.

## Next round after Round 130

- Implement only the CPU-side, disabled-by-default deterministic wrench-pulse
  contract in riser reference playback. It must be hash-bound, reject missing
  or conflicting profile fields, emit compact perturbation telemetry, and
  prove that the disabled/zero profile is command-identical to current
  playback.
- Define one bounded case-30 measurement proposal after the runner change. The
  existing dynamic/safety gates remain unchanged; the measurement must remain
  trace-only and must independently report physical passage, visited-state
  coverage relative to the case-4 hotspot, and shadow-teacher label gap.
- Do not issue a runtime token or launch the canary while implementing the CPU
  contract. Do not create a DAgger dataset, change the split, open holdout,
  retrain BC, start PPO, or begin obstacle work.

## Round 131: deterministic case-30 perturbation plumbing is CPU-complete

- Commit `2f993c9aee021c2d641f5e401598529459919aa0` adds the
  disabled-by-default deterministic wrench-pulse contract to riser reference
  playback. The complete authoritative `.98` CPU suite passes `528/528` in
  `46.39 s`.
- The profile schema is exact and hash-bound. It currently admits case 30 only,
  a single body-longitudinal force pulse of at most `40 N`, at most `50` policy
  steps (`0.25 s` at `200 Hz`), and application height at most `1.0 m`.
  Missing/extra fields, case mismatch, zero/oversized force, invalid duration,
  invalid height, and non-finite clocks fail closed.
- Pulse onset is indexed by immutable execution-phase time; pulse duration is
  counted in policy steps so a phase-governor hold cannot extend the force.
  Telemetry independently records trigger step/phase, exact active-step count,
  release, and the frozen assertions that the perturbation was not applied to
  planner commands or policy actions. Only two compact pulse fields are added
  to the existing 1 Hz trace.
- The runtime mode requires one learned-policy shadow-teacher case and forbids
  normalized dataset, raw-teacher, policy-trace, and zero-action modes. The
  independent perturbation contract joins final admission without changing the
  existing dynamic, thermal, controller-evidence, or residual-label outcomes.
  Disabled mode emits zero force and remains command-identical.
- Diff audit confirms that this commit changes only the playback seam, the new
  pure perturbation module, and its tests. No planner, source/plan geometry,
  controller, gate-threshold, dataset builder, BC, or PPO file changed.
  Playback/module/test SHA-256 values are respectively
  `cbe96aaa9ee775454f915a223f2ed15b23dfa51dbfbdc9965d62a368542c611b`,
  `37ac835f888b1d4cdfacc3e40a1854bb1d9801cd251dbdb03d80e4bbd5e01897`,
  and `45e8a3b768a3048fba62353a594cf4afdd31ed3679925b30e378a1ee1d6cc494`.
- No pulse profile, runtime namespace, authorization token, Isaac process,
  residual dataset, BC, or PPO run was created. WSL and NVIDIA compute-process
  checks were empty after the CPU suite.

## Next round after Round 131

- Build a CPU-only case-30 pulse-design proposal. Bind the case-4 hotspot,
  case-30 plan/policy/teacher identities, and choose one phase-localized pulse
  profile from the existing validated LQR push envelope. Do not infer a force
  from trajectory distance alone; document the state-coverage hypothesis and
  stop conditions.
- The proposal may write a non-authorizing profile candidate, but it must not
  create a runtime namespace or token. A later, separately reviewed canary must
  still prove unchanged physical gates, exact pulse telemetry, improved
  case-4-hotspot state coverage, and independent shadow-label materiality.
- Keep case 4 in validation, case 78 unadmitted, holdout unopened, and DAgger
  dataset creation, BC, PPO, obstacles, and broad rollout closed.

## Round 132: case-30 pulse profile is localized and remains non-authorizing

- Commit `354ffdd` adds a reproducible CPU-only case-30 pulse designer. The
  complete authoritative `.98` suite passes `532/532` in `45.28 s`; focused
  design/coverage/perturbation tests pass `25/25`.
- The designer binds the frozen teacher dataset, masked policy, case-4 shadow
  trace/diagnosis, localized and unused-plan audits, Round-130 architecture
  proposal, all-70 plan summary, exact case-30 plan, and three passed LQR
  disturbance-envelope results. Every identity and semantic input check passes.
- All 70 admitted plans define the feature normalization. A local-16 directed
  search maps the case-4 material-shift command/lookahead region to case 30 at
  plan index `474`, phase `15.6665929376 s` of `29.2224881939 s`. This leaves
  `2 s` startup and `5 s` terminal-recovery margins. The selected context has
  feed-forward `vx=0.192157 m/s`, `wz=0.250420 rad/s`, riser rate
  `0.006626 m/s`, and riser position `0.781915 m`.
- The candidate profile requests one body-forward `+20 N` pulse for exactly
  `20` policy steps (`0.1 s`) at `0.5 m` height. The sign is an explicit
  hypothesis: case 4's shadow-minus-original normalized `vx` label has signed
  mean `-0.058188`, so a forward disturbance may induce the same negative
  correction. No claim is made that this will reproduce the material `wz`
  shift without measurement.
- The scalar force/duration/height prior is backed by three LQR gates covering
  riser positions `[0.0,0.6,1.2] m`, `168` provisional-plant scenarios,
  `100%` success, peak pitch `9.383906 deg`, and zero action saturation. Those
  gates used global-X force while the proposed pulse is body-X, so
  `frame_transfer_dynamically_validated=false`; this is the canary hypothesis,
  not prior safety proof.
- Profile SHA-256 is
  `6ab2fe1212f88e3baecac4f64156bb9820069203136758a22a2cb632813a5907`;
  proposal SHA-256 is
  `4490a35794988de2cf56fd18a06d1a77c7df8c99267e66c98d9c801b6223e65b`.
  Independent Mac readback matches both. No runtime namespace, token, Isaac
  process, dataset, DAgger, BC, PPO, split change, or holdout access occurred.

## Next round after Round 132

- Build a CPU-only single-case admission wrapper that pins the Round-132
  proposal/profile, case-30 plan, masked policy, controller gains, playback
  code, action scales, and unchanged dynamic/thermal/controller/label gates.
  It must reject every conflicting environment override before Isaac starts,
  require fresh exclusive WSL/Windows/NVIDIA ownership, and write a bounded
  failure summary.
- Do not issue or consume a runtime token as part of the wrapper change. After
  the wrapper and full CPU suite are clean, perform a separate go/no-go review
  for exactly one case-30 measurement. If later authorized, stop after case 30
  regardless of physical or coverage outcome.
- Keep the case-4 validation split, case-78 status, holdout, dataset creation,
  DAgger, BC, PPO, obstacle work, and broad rollout unchanged and closed.

## Round 133: case-30 CPU admission contract is sealed; runtime stays closed

- Commits `8d27c95`, `97f85ba`, and `86c3cfb` add the preflight-only
  wrapper, validator, pinned gate values, and canonical case-30 contract;
  `ff349c8` separates WSL wrapper execution tests from platform-neutral source
  checks. The full authoritative `.98` suite passes `537/537` with one
  intentional Windows-only skip in `53.32 s`.
- The contract pins the Round-132 proposal/profile, exact case-30 plan, masked
  policy and its final/report evidence, frozen teacher dataset, LQR gains,
  robot build audit/URDF/USD, playback, perturbation runtime, shadow diagnosis,
  wrapper, and validator. All SHA-256 and committed Git-blob identities pass.
- Controller arguments are frozen at `wz_kp=1.05`, duration scale `3.0`, camera
  lever-arm compensation gain `1.0`, correction cap `0.05 m`, and CUDA policy
  execution. Residual scales remain `[0.35,0.40,0.10]`; all existing dynamic,
  thermal, controller-evidence, and residual-label thresholds remain unchanged.
- Live `.98` preflight at commit
  `ff349c85c80586314d572de1f71f0c45815eb6f3` proves clean pushed HEAD,
  canonical tracked contract/blob, reviewed-parent ancestry, fresh namespace,
  exact profile/action/gate values, all artifact identities, and exclusive
  WSL/Windows/NVIDIA ownership. Contract SHA-256 is
  `2e567b5a00adab72a488714a9a6fef6ab66a36931521466ccb4fc46a705d0993`;
  contract Git blob is `3279a41f16ef7493d6b3093bab1ad42bdef3e431`.
- The wrapper has no Isaac playback command and no valid authorization token.
  An explicit `.98 --execute` attempt exits `7` with machine-readable reason
  `runtime_authorization_not_issued`; it creates no namespace and records
  `runtime_started=false`, `dataset_created=false`, and all learning gates
  false.
- Decision: **GO** to prepare a separate, reviewed one-use runtime authorization
  change for exactly case 30; **NO-GO** to launch from the current commit. This
  distinction preserves CPU-contract readiness without converting it into GPU
  or training authorization.

## Next round after Round 133

- Only after explicit review, add a separate one-use runtime authorization
  contract and bounded execution path pinned to the current CPU contract. The
  route must consume the token before Isaac starts, enforce a `600 s` timeout,
  preserve partial logs on failure, clear the wrench, and always write a final
  machine-readable status with physical, perturbation, and label outcomes kept
  independent.
- Run exactly one case-30 learned-policy shadow-teacher measurement. Stop after
  case 30 regardless of outcome. A dynamic failure ends the path; a dynamic
  pass still creates no training data and only permits CPU comparison of visited
  state coverage and shadow-label materiality against the case-4 hotspot.
- Do not change case 4/78 splits, open holdout, create a DAgger dataset, retrain
  BC, start PPO, begin obstacle work, or run a broader campaign.

## Round 134: case-30 perturbation is dynamically clean but does not support DAgger

- Commits `125955b`, `8b66a4d`, `7445d3d`, and `222a834` added the
  one-use runtime admission, exact committed-blob checks, independent final
  summarizer, and negative authorization tests. The first v1 playback completed
  physically, but a wrapper quoting bug routed Windows outputs to the literal
  sibling `artifacts/two_wheel_riser$NAMESPACE`. Its playback exit was zero,
  but diagnosis was not run and final admission failed. Both locations are
  preserved as quarantined path-contract evidence; nothing was moved or
  relabeled.
- Commits `2589c93`, `21bb639`, `d887d76`, and `cb52535` repaired only the
  Windows namespace construction, advanced to a fresh v2 namespace, and
  resealed both CPU and runtime contracts. The regression test forbids an
  escaped literal namespace. Diff audit confirms no planner, plan, policy,
  controller, perturbation profile, physics, threshold, or dataset code changed.
- The complete authoritative `.98` suite passes `546` tests with two
  intentional platform skips in `62.21 s`. WSL preflight at clean pushed commit
  `cb525353b02c8476f61704e70e4b2da25ebd02aa` passes every CPU/runtime identity,
  lineage, fresh-namespace, training-closure, and ownership check. The token was
  mode `0600` and was removed before Isaac started.
- The canonical v2 canary passes. It completes source/execution clocks
  `18.144412/29.2224881939 s` in `11,494` policy steps. Position p95/max is
  `0.141947/0.170459 m`; attitude p95/max is `0.154563/0.225145 deg`; pitch max
  is `7.030357 deg`; riser p95/max is `0.011899/0.014258 m`; all action, riser,
  and proxy saturation ratios are zero. Dynamic, thermal, controller-evidence,
  and residual-label-envelope gates all pass with no termination.
- Perturbation telemetry proves exactly one body-forward `20 N` pulse for
  `20` policy steps, triggered at phase `15.6668266485 s`, then released. The
  planner and policy commands were not modified by the pulse contract and no
  dataset was created. Raw residual maxima are
  `[0.240709,0.163077,0.017046]`; normalized maxima are
  `[0.802364,0.407692,0.170461]`, inside the frozen envelope.
- The independent `11,494`-row shadow diagnosis finds no material p95 shift in
  `vx`, `wz`, or riser. Shadow-minus-original p95 is
  `[0.014944,0.016629,0.001968]`; policy-to-shadow RMSE is `0.006761` versus
  policy-to-phase RMSE `0.008191`, ratio `0.825441`. Therefore
  `dagger_dataset_proposal_supported=false`. This is a successful measurement,
  not evidence to admit labels or retrain.
- Canonical evidence SHA-256 values are admission
  `b01478172476c6e0b598a2102e721ba745e6ecb2293fe5f0f37d416f74e15382`, gate
  `5aa269f310399328654c2d37cb235e2b6196cc7edf46785dac8b3c57b3d272cf`, trace
  `24818981a5d6e43c7e32f77dcb54539f5330f962b85c9c0f3f5147607a91ea53`,
  diagnosis `6eb3b948297a048d53d11b8ec10b7b5a5772436ad568058bdaa8762bdc59d96e`,
  and final status
  `fc08b890ec0dfee0a1f0d05505afd806f68c54ad564afc99fe5e73533e3ebfb6`.
  WSL and NVIDIA ownership are empty after closure. Dataset creation, split
  change, holdout access, DAgger, BC, PPO, obstacles, and broad rollout remain
  closed.

## Next round after Round 134

- Perform one CPU-only visited-state coverage audit. Compare the canonical v2
  case-30 trace with the frozen case-4 material-shift hotspot under the same
  masked-policy observation normalization and directed state/action-distance
  contract used in Round 129. Report perturbed versus nominal case-30 scores,
  exact improvement ratio, and the existing `1.50x` reference calibration.
- Keep physical success and label materiality independent. If state coverage
  did not materially improve, reject this `+20 N` profile as a coverage source;
  do not create a dataset or infer that a stronger pulse is automatically safe.
  A later CPU proposal must choose between one separately validated bounded
  profile change and the transparent split-reset fallback.
- Do not launch Isaac, open case 78/holdout, change the split, write a DAgger
  dataset, retrain BC, start PPO, begin obstacle work, or run a broader campaign
  during the coverage audit.

## Round 135: the measured perturbation fails hotspot coverage; split fallback is proposed

- Commits `b91b4a8` and `891a4c6` add and publish a CPU-only coverage audit
  using the same masked-policy normalization and directed state/action-distance
  formula as Round 129. Focused `.98` coverage/ranking tests pass `6/6`.
- The audit compares `2,048` canonical v2 policy-visited case-30 rows against
  nominal teacher case 30 and the frozen `4,682`-row case-4 material-shift
  hotspot. The nominal and perturbed scores are `0.941425` and `0.938781`, so
  the perturbed/nominal ratio is `0.997191`: only `0.280868%` improvement versus
  the frozen `10%` material threshold. The perturbed/reference ratio is
  `3.653404`, above the existing `1.50` calibration limit.
- Therefore both `state_coverage_materially_improved` and
  `reference_calibrated_coverage_passed` are false. The comparison explicitly
  uses phase-aligned original case-30 teacher actions, not shadow labels, and
  states that a nominal-teacher comparison cannot prove causal attribution to
  the pulse. Coverage-report SHA-256 is
  `86fab43a35b8c4ca5676be851aedb730923801faa20eaa4cc870c5c02819e9f1`.
- Combined with Round 134's no-material-label result, this rejects the measured
  `+20 N` pulse as a DAgger coverage source. It does not prove every possible
  perturbation is useless, but it provides no basis to escalate force blindly;
  no stronger pulse is authorized.
- Commit `370d3b8` adds a non-applying transparent split-reset proposal; focused
  `.98` proposal/coverage tests pass `4/4`. All proposal input checks pass. It
  keeps the current split frozen while proposing, only after all future gates,
  case 4 moving permanently from validation to training and case 78 replacing
  it in validation. Holdout `[3,5,13,19,24]` is unchanged.
- Case 78 is exact-source/transition/kinematic clean with `6,870` source and
  execution states, source/execution clocks `135.487646/192.299567 s`, path
  length `62.453112 m`, and kinematic position p95/max
  `0.106876/0.239139 m`. It has no dynamic qualification and remains
  `valid_for_training=false`; the split is not applied. Proposal SHA-256 is
  `975bfa46cede07daa70ae36f81c354ea707898a38addf2fe54ec73a75aaf8072`.
- No Isaac launch, runtime token, split mutation, case-4 label admission,
  case-78 validation admission, holdout access, dataset, DAgger, BC, PPO, or
  obstacle work occurred in this round.

## Next round after Round 135

- Build only the CPU-side case-78 deterministic dynamic-qualification contract.
  Pin the exact plan, controller/gains/USD/playback identities, unchanged
  dynamic/thermal/controller gates, no-policy/no-capture mode, and one-case
  stop. Benchmark and justify a wall timeout for the `192.299567 s` execution
  plan instead of reusing the short case-30 timeout blindly.
- The CPU contract must issue no runtime token and must not create a namespace.
  A later separately reviewed exclusive canary may dynamically qualify case 78;
  only a full pass can permit a new immutable split manifest. A failure leaves
  the current split unchanged and requires a new fallback decision.
- Keep case 4 in validation, case 78 unused, holdout closed, and dataset
  creation, DAgger, BC, PPO, obstacles, and broad rollout disabled until that
  sequence is satisfied.

## Round 136: case-78 deterministic qualification is CPU-admitted; runtime stays closed

- Commits `d105ffd` and `a21fedb` add the case-78 CPU validator, non-executing
  preflight wrapper, negative tests, and canonical hash-sealed contract. The
  complete authoritative `.98` suite passes `554` tests with two intentional
  platform skips in `60.76 s`.
- The contract pins the split-reset and failed-coverage proposals, exact case-78
  plan/manifest/summary, LQR gains, robot build audit/URDF/USD, deterministic
  playback, plan loader, tracking, riser-control and recovery-evidence modules,
  timing handoff, wrapper, and validator. All 17 identities pass SHA-256 and,
  where applicable, committed Git-blob checks.
- The future canary is frozen to one case 78 in deterministic-teacher mode with
  no residual policy, zero-policy baseline, capture, or dataset. Controller
  values remain `wz_kp=1.05`, duration scale `3.0`, camera lever-arm gain `1.0`,
  and correction cap `0.05 m`; all existing dynamic, thermal, controller, and
  saturation thresholds remain unchanged.
- Timeout is derived rather than copied from a short run. Deterministic case 1
  observed `102.425/77.833030=1.315958x` wall/execution time; case 52 observed
  `354.76/292.740729=1.211857x`. Case 78's worst allowed simulated horizon is
  `192.299567*3=576.898702 s`; scaling by the larger observed ratio gives
  `759.174473 s`. The proposed `900 s` wall timeout retains `140.825527 s` for
  startup and shutdown.
- Live `.98` preflight at clean pushed commit
  `a21fedb4d4838f80b2efceafbe8dc477247a3f8b` passes every semantic, identity,
  lineage, fresh-namespace, training-closure, and WSL/Windows/NVIDIA ownership
  check. Contract SHA-256 is
  `c3d77a1110a46c5b2a1e60dd1a33d3fa19ca22cb6c3b992d5ebcfee80b9950e5`;
  Git blob is `95fe36ca25d9b4e1041e15e54391f7e270ae6e26`.
- Explicit `.98 --execute` exits `7` with reason
  `runtime_authorization_not_issued`, records runtime/GPU/dynamic/split/dataset
  flags false, and leaves the namespace absent before and after. No token,
  Isaac process, split change, case-4 label admission, case-78 validation
  admission, holdout access, dataset, DAgger, BC, PPO, or obstacle work occurred.

## Next round after Round 136

- Perform a separate go/no-go review of the sealed CPU contract. A GO may add a
  one-use runtime authorization contract and deterministic runner pinned to the
  `900 s` bound and current blobs. The runner must consume its token before
  Isaac, preserve partial logs, write a machine-readable final status on every
  exit, and stop after case 78.
- A physical pass must independently prove complete reference execution,
  unchanged dynamic/thermal/controller gates, no termination, and no dataset.
  It admits case 78 only for the proposed validation role; it does not itself
  apply the split or authorize case-4 labels. A physical failure keeps the
  current split unchanged and ends this fallback path for review.
- Do not create a runtime contract and token in the same unreviewed change. Keep
  case 4 in validation, case 78 unused, holdout closed, and all learning and
  obstacle work disabled meanwhile.

## Round 137: case-78 qualification times out fail-closed; split fallback remains blocked

- Commits `d3b541d`, `8f81b8f`, and `3002580` add and seal the separate
  one-use runtime wrapper, final summarizer, authorization validator, negative
  tests, and immutable runtime contract. The runtime contract pins the reviewed
  CPU checkpoint `dacb00b`, implementation commit `d3b541d`, exact case 78,
  current `unused` split, namespace
  `20260721_case78_dynamic_qualification_v1_exclusive`, four committed runtime
  identities, one-case limit, and `900 s` timeout. Split mutation, holdout
  access, dataset creation, DAgger, BC, and PPO remain false.
- The authoritative `.98` suite passes `559` tests with three intentional
  platform skips in `58.35 s`. At clean pushed commit
  `3002580988ca72432b15864296a03c35a69daf4e`, preflight passes every CPU and
  runtime identity, lineage, canonical-path, fresh-namespace, learning-closure,
  and ownership check. The mode-`0600` token is consumed before Isaac starts.
- The sole deterministic case-78 canary reaches the exact `900 s` wrapper
  limit and exits `124`; observed wall duration is `900.020283 s`. The process
  does not write `case_0078.json`, so source/execution completion, physical
  quality, thermal admission, and controller evidence are unproven and fail
  closed. This is a runtime-horizon rejection, not a physical trajectory pass
  or a case-78 validation admission.
- GPU release passes after timeout. The final status records
  `physical_quality_passed=false`, `dynamic_qualification_passed=false`,
  `case78_validation_admitted=false`, `split_changed=false`,
  `dataset_created=false`, and `valid_for_training=false`. Case 4 remains in
  validation, case 78 remains unused, and holdout `[3,5,13,19,24]` is unopened.
- Canonical evidence SHA-256 values are admission
  `8a2f3fd15141520d257e462e6a9f1143362148a139e0a9b6e38326ba6027558f`,
  playback log
  `8a86a8486fcbe4d371017bbb38c58cb512946dc8dba833858c27b6e950bf207f`,
  and final status
  `94fe6c39ae9550802979ddd68fcbf45ef3d1964600db8afa01a69ddd804defb8`.
  NVIDIA compute ownership is empty after closure.
- The timeout derivation in Round 136 is disproven for this case. Ratios from
  completed case 1/52 executions do not bound a run whose progress governor can
  consume the full `3x` step horizon, and the current runner keeps its 1 Hz
  trace in memory until normal completion. Consequently the timeout preserves
  no last phase/step/controller state, which is an evidence-contract gap rather
  than a reason to enlarge the timeout blindly.

## Next round after Round 137

- CPU-only, add a lightweight atomic runtime heartbeat/checkpoint at a bounded
  cadence. It must expose completed policy steps, elapsed and phase clocks,
  current progress scale, tracking/safety state, and the fixed maximum-step
  horizon without changing commands, simulation cadence, the existing 1 Hz
  final trace, source/plan geometry, or gates. A timeout finalizer must seal the
  last checkpoint and independently state that no gate result or dataset was
  produced.
- Replace the completed-trajectory wall/execution timeout estimate with a
  conservative policy-step-throughput contract derived from measured bounded
  runs plus explicit startup/shutdown margin. Review the new evidence and bound
  separately before considering one fresh-namespace retry; do not reuse the
  consumed token or current namespace.
- Do not apply the split fallback, admit case 78, open holdout, create labels or
  datasets, retrain BC, start PPO, begin obstacle work, or launch another Isaac
  run during this CPU-only repair.

## Round 138: timeout semantics are repaired CPU-only and progress can be preserved

- Commit `ed207f4` adds an optional atomic runtime heartbeat to the deterministic
  playback without changing controller commands, physics cadence, phase update,
  source plan, existing 1 Hz in-memory trace, gates, or capture permissions. At
  a default `2,000`-policy-step cadence it overwrites one lightweight JSON with
  completed/max steps, emitted host epoch, virtual elapsed/phase clocks,
  progress scale, current tracking state, running safety maxima, saturation,
  and termination state. Every snapshot is explicitly non-training and creates
  no dataset.
- The same commit adds a canonical-hash wall-bound audit and four focused tests.
  The complete authoritative `.98` suite passes `563` tests with three
  intentional platform skips in `59.12 s`. The local focused tests pass `4/4`;
  the Mac full suite cannot collect because its Python environment lacks
  `gymnasium`, so `.98` is the authoritative result.
- The audit proves that case 30's field labeled `wall_duration_s=57.47 s` is
  exactly `11,494 / 200 Hz`, i.e. virtual policy-step duration. It is not host
  elapsed time. The sealed admission-to-final filesystem envelope is `440 s`,
  producing a conservative measured throughput of `26.122727` policy steps/s.
  This confirms the Round-136 timeout model used the wrong clock.
- At case 78's fixed maximum `115,381` policy steps, that conservative rate
  estimates `4,416.881851 s` for the loop. Adding `900 s` for startup, shutdown,
  and diagnosis and rounding upward to five-minute quanta proposes a `5,400 s`
  wall limit. This is deliberately a bound proposal, not permission to run for
  90 minutes or evidence that case 78 will need the full horizon.
- All canonical input hashes pass, including the case-30 admission/gate/final
  and case-78 timeout final status. Wall-bound report SHA-256 is
  `dc99f0a8e48dc859147e93a38b4723a05fb8ea7ee8a2df8e7db054d4faab9663`.
  It records runtime retry, GPU launch, split change, dataset creation, BC, PPO,
  and training validity false. No Isaac process or authorization token was
  created in this round.

## Next round after Round 138

- Build a fresh CPU-only case-78 v2 qualification contract. Pin the new
  heartbeat-enabled playback/helper, canonical wall-bound report, unchanged
  plan/controller/gains/USD identities and gates, `2,000`-step cadence,
  `5,400 s` ceiling, and a fresh namespace. Its preflight must issue no token,
  create no runtime namespace, and reject execution.
- The future timeout finalizer must hash and seal the last heartbeat, report its
  age and host-observed step throughput, and keep timeout, physical quality,
  thermal/controller admission, data absence, and GPU release independent. A
  timeout remains a rejection even when the heartbeat is healthy.
- Do not retry case 78, apply the split fallback, admit labels, open holdout,
  capture data, retrain BC, start PPO, or begin obstacle work until the v2 CPU
  contract receives a separate go/no-go review and one-use runtime layer.

## Round 139: heartbeat-enabled case-78 v2 qualification is CPU-admitted

- Commits `d1b4c8f` and `8251aed` add the v2 CPU validator, non-executing
  preflight wrapper, negative/semantic tests, and canonical contract. The
  contract pins the unchanged case-78 plan, fallback proposal, LQR gains,
  robot build/URDF/USD, plan loader, tracking/control/recovery code, timing
  handoff and dynamic thresholds, plus the heartbeat-enabled playback/helper,
  prior timeout final status, and corrected wall-bound audit.
- The future deterministic command line remains unchanged apart from adding an
  observational heartbeat. The contract fixes heartbeat cadence at `2,000`
  policy steps, maximum steps at `115,381`, and wall ceiling at `5,400 s`.
  The heartbeat is atomic, cannot alter commands, and cannot create a dataset.
- The complete authoritative `.98` suite passes `568` tests with three
  intentional platform skips in `58.58 s`. Live preflight at clean pushed
  commit `8251aedc9d2fa2b29ef29cef07f1c8c23aa5a261` passes all 19 identity,
  semantic, lineage, canonical-path, fresh-namespace, training-closure, and
  WSL/Windows/NVIDIA ownership checks.
- Contract SHA-256 is
  `f4244f83a9b31ca0f1c403fe4e2467ce4379eaf82e6b404cc16a523ec7c1a5c7`;
  Git blob is `d9e93ead11810d7c2528b9ade69cbcf3b341afdc`. Explicit
  `--execute` exits `7` with `runtime_authorization_not_issued`, and namespace
  `20260721_case78_dynamic_qualification_v2_heartbeat_exclusive` remains absent.
- No runtime token, Isaac launch, split change, case-78 validation admission,
  case-4 label admission, holdout access, dataset, DAgger, BC, PPO, or obstacle
  work occurred in this round.

## Next round after Round 139

- Perform a separate go/no-go review of the v2 CPU contract and its unusually
  long `5,400 s` bound. A GO may add a new one-use runtime authorization layer
  and timeout finalizer pinned to the current blobs; it must seal the last
  heartbeat and preserve separate runtime, physical, thermal/controller,
  no-data, GPU-release, and split outcomes on every exit.
- If authorized, launch exactly one deterministic case-78 canary in the fresh
  v2 namespace and stop afterward. A full physical pass may support a later
  immutable split-manifest proposal; timeout or any physical/evidence failure
  leaves case 4 in validation and case 78 unused.
- Do not create the runtime contract and token in one unreviewed change. Do not
  launch case 78, open holdout, create data, retrain BC, start PPO, or begin
  obstacle work during the review.

## Round 140: case 78 completes dynamically but narrowly fails position p95

- Commits `18d1cc6`, `8b6492a`, and `cf4a6d8` add, pin, and seal the separate
  heartbeat-enabled one-use runtime layer. Its synthetic success, timeout,
  stale-heartbeat, wrapper, and token tests pass `7/7`. The complete
  authoritative `.98` suite passes `574` tests with four intentional platform
  skips in `58.47 s`; runtime preflight passes all CPU/runtime identities at
  clean pushed commit `cf4a6d820de7ca9606a7625ac848605bed46edbf`.
- The mode-`0600` token is consumed before Isaac starts. Exactly one
  deterministic case-78 canary runs in namespace
  `20260721_case78_dynamic_qualification_v2_heartbeat_exclusive`; no residual
  policy, capture output, dataset, split mutation, holdout access, BC, or PPO is
  enabled.
- The run completes rather than timing out: source/execution clocks are
  `135.487646/192.299567 s`, completed phase is exact, policy steps are
  `85,760`, and measured host wall time is `2,904.385536 s` at
  `29.549360` steps/s. The last atomic heartbeat is only `2.123 s` old at
  finalization. Runtime, heartbeat, wall-bound, GPU-release, thermal,
  controller-evidence, no-termination, and no-data checks all pass.
- Dynamic quality fails only `position_p95_bounded`: position p95/max is
  `0.162650/0.229624 m` against `0.15/0.25 m`. The p95 miss is `12.65 mm`;
  pitch max is `8.179473 deg`, attitude p95/max is
  `0.146361/0.284986 deg`, riser max is `0.013981 m`, proxy max is
  `0.225347 deg`, proxy rate is `44.229582 deg/s`, and all saturation ratios
  are zero. Every other physical check passes.
- Residual-label envelope independently fails because raw vx residual reaches
  `0.36 m/s`, or `1.20x` the frozen `0.30 m/s` label scale. That label was not
  applied to commands and no label dataset exists. Physical failure and label
  envelope failure remain independent.
- Canonical SHA-256 values are admission
  `8f840b1214ecaf3904b77ee8a7c4178e050b1cabca9a545a4beb0574b4bc6c3e`,
  heartbeat
  `a470d7a65abaacc922cc83d0069cb788d54c34ff3011efed78784a0af7d9d31f`,
  gate `46ab1f27d2ed16271853e068e21497d66f6cacfb8599f98dde0c72df6d31c97a`,
  final status
  `7de1e760f7431a03a7d783107ad0ebf638819d45022aa2c8b94495bd06390a3e`,
  and playback log
  `e88b6285dd0f4fdf8167df6c5c914717d30e4132e10a7da0bf89f25239774feb`.
  GPU ownership is empty after closure. Case 4 remains validation and case 78
  remains unused.

## Round 141: existing camera-recovery governor is the bounded CPU candidate

- Commit `ccb7875` adds a canonical-hash CPU trace audit with two focused tests;
  the complete `.98` suite passes `576` tests with four intentional skips in
  `59.30 s`.
- The sealed 430-sample 1 Hz trace confirms only position p95 failed. Camera
  lever-arm correction was saturated for `96.9767%` of trace samples, while
  `5.1163%` of samples exceeded `0.15 m`. The existing recovery governor is
  therefore structurally relevant: it slows phase only when correction is
  saturated and camera error approaches the unchanged gate.
- Offline application of the existing default governor
  (`start=0.13 m`, `full=0.155 m`, minimum scale `0.20`) changes trace mean
  progress from `0.447660` to `0.439303`. The conservative trace projection is
  a `1.019023x` step multiplier, or `87,392` steps versus `85,760`, still below
  the fixed `115,381` horizon.
- The audit changes no source plan, inner LQR, correction cap, dynamic threshold,
  or training boundary. It explicitly states that the trace estimate is not
  physical proof and issues no runtime/GPU authorization. Report SHA-256 is
  `3c07fbca71b521be24179bb866f45cee38992b6bb18c3948f9719326ec1ba7cf`.

## Next round after Round 141

- Build a CPU-only case-78 recovery contract that changes exactly one runtime
  argument: enable the existing camera-error recovery governor with its frozen
  `0.13/0.155 m` range and `0.20` minimum scale. Pin the v2 failure evidence,
  recovery audit, unchanged plan/controller/gains/USD/gates, heartbeat, and one
  fresh namespace. Issue no token and reject execution.
- A later separately reviewed canary must keep the same `5,400 s` wall bound,
  stop after case 78, and require every original physical threshold. A pass can
  support a later split-manifest proposal but cannot apply the split itself.
- Do not relax the p95 gate, widen the residual label scale, increase the camera
  correction cap, alter source geometry, open holdout, capture data, retrain BC,
  start PPO, or begin obstacle work during the CPU contract round.

## Round 142: bounded camera-recovery candidate is CPU-admitted

- Commits `aada1f2` and `8d3bec9` add and seal the CPU-only recovery validator,
  non-executing preflight, negative/semantic tests, and canonical contract. The
  checked-in contract changes exactly the existing recovery-governor argument
  family and pins the prior v2 gate/final evidence plus the CPU trace audit.
- The candidate enables camera-error recovery with the existing implementation
  defaults `start=0.13 m`, `full=0.155 m`, and minimum phase scale `0.20`.
  Case-78 source plan, LQR, `wz_kp=1.05`, duration scale `3.0`, camera
  lever-arm gain/cap `1.0/0.05 m`, all physical thresholds, heartbeat cadence
  `2,000`, and wall limit `5,400 s` remain unchanged.
- The complete authoritative `.98` suite passes `580` tests with four
  intentional platform skips in `60.37 s`. Live preflight at clean pushed
  commit `8d3bec9ca02f1079d9d925c39759b92d0a2a6738` passes all 12 identity,
  semantic, lineage, fresh-namespace, learning-closure, and ownership checks.
- Contract SHA-256 is
  `efa707d6bda6cc1428060ff8dac599593b85deb4c3b5d0e1225fa2d8284088ad`;
  Git blob is `8ca63149fdd0fedc63112f8494e9fe59873f5559`. Explicit
  `--execute` exits `7` with `runtime_authorization_not_issued`, and namespace
  `20260722_case78_camera_recovery_v1_exclusive` remains absent.
- No runtime token, second Isaac launch, split change, label admission, holdout
  access, dataset, BC, PPO, or obstacle work occurred in this round.

## Next round after Round 142

- Perform a separate go/no-go review of the recovery CPU contract. A GO may add
  a one-use runtime wrapper/authorization/finalizer pinned to the current blobs
  and must pass the exact recovery arguments while preserving heartbeat and all
  v2 no-data/timeout/GPU-release evidence contracts.
- If authorized, run exactly one fresh-namespace case-78 recovery canary and
  stop. It must beat the original `0.15 m` p95 gate without regressing any
  original physical threshold. A physical pass still requires a later immutable
  split-manifest change; a failure keeps case 4 in validation and case 78 unused.
- Do not relax thresholds or label scales, change plan/LQR/correction cap, open
  holdout, create data, retrain BC, start PPO, or begin obstacle work.

## Round 143: camera-error recovery completes but remains dynamically rejected

- Commits `23ea37c`, `e578055`, and `56e424c` add, pin, and seal the separate
  one-use recovery runtime. The authoritative `.98` CPU suite passes `584`
  tests with five intentional skips. Its mode-`0600` authorization is consumed
  before Isaac starts, and exactly one case-78 run executes in namespace
  `20260722_case78_camera_recovery_v1_exclusive` with no policy, capture,
  dataset, split mutation, holdout access, BC, or PPO.
- The run completes the exact `135.487646/192.299567 s` source/execution clocks
  in `88,922` policy steps and `2,966.443678 s` host wall time. Runtime,
  heartbeat, GPU release, safety, thermal, controller evidence, no-termination,
  and no-data checks all pass.
- Dynamic quality still fails only `position_p95_bounded`. Position p95 worsens
  from the baseline `0.162650 m` to `0.173540 m`, while max improves slightly
  from `0.229624 m` to `0.226585 m`. Recovery activates for `9.7917%` of the
  run and adds `3,162` policy steps. The residual label envelope independently
  remains false, but no residual is applied and no labels are written.
- Canonical recovery evidence SHA-256 values are gate
  `ced834a3f0787ca11e33bc23c9134e041bd6d4ee4159249c05c0f7ef6e32eb50`,
  admission
  `e7c9fdec97017a3542e4464fb18f533115b01158856f1c3445daf4e3bd55563b`,
  heartbeat
  `6f6ecef8a36ba64099d0438bf441a3bef91b82036de4974eef72f429c4e88aa7`,
  final status
  `bd92e4f5c8c8bf277096fb86270a65058c21572840f0eef4ceb0ad54e20d2fd5`,
  and log
  `ee32bb959d69854b5baf2684680eef32b69e40722cb0435c502399e0c648bbcc`.
  Case 4 remains validation and case 78 remains unused.

## Round 144: phase audit rejects dwell recovery and selects one direct correction

- Commits `571982b`, `d31f85d`, `b30b2a9`, and `5751df8` add and harden two
  canonical-hash CPU audits. The recovery outcome audit separates evidence
  validity from candidate admission and passes all evidence checks.
- Phase alignment at `0.25 s` shows that recovery improves phase-matched p95 by
  `4.983 mm`, from `0.141361 m` to `0.136378 m`, but time spent dwelling at
  high error worsens the official time-weighted p95 by `10.890 mm`. The
  recovery governor is therefore rejected as a dynamic candidate even though
  it is not a broad per-phase regression. Report SHA-256 is
  `96c0a66c937addb28a9ed6dc7f9a222cdf4af59a108cfa1adf36c81556767dee`.
- Both sealed traces reconstruct the current camera lever-arm correction to
  numerical precision. The `0.05 m` cap is saturated in nearly every sampled
  state, while base tracking and all non-position physical gates pass. An
  ideal unit-transfer replay selects one bounded direct change: keep recovery
  disabled and increase only the correction cap from `0.05 m` to `0.10 m`.
- The idealized `0.10 m` replay projects baseline p95/max to
  `0.123593/0.199190 m` and recovery-evidence p95/max to
  `0.132212/0.231080 m`, retaining at least `15 mm` projection margin under
  both unchanged gates. The calculation explicitly does not model closed-loop
  dynamics and is not physical proof. Candidate report SHA-256 is
  `ed910cb9651e264613d96dd04e30375e75ce1604f97e15778e325a6adddccfe3`.

## Round 145: case-78 direct-cap candidate is CPU-admitted; runtime stays closed

- Commits `1981180`, `127978f`, and `2c1a5d9` add the non-executing wrapper,
  hash/lineage validator, negative tests, and sealed CPU contract. The only
  controller delta from the heartbeat baseline is
  `maximum_camera_lever_arm_correction_m: 0.05 -> 0.10`; source plan, LQR,
  `wz_kp=1.05`, duration scale `3.0`, USD, heartbeat, physical thresholds,
  and all training boundaries remain unchanged.
- The complete authoritative `.98` suite passes `600` tests with five
  intentional skips in `55.56 s`. Live preflight at clean pushed commit
  `2c1a5d9e0c1a240435255300d5e2fea92b36ca23` passes every canonical-path,
  exact-identity, Git-blob, reviewed-parent/implementation lineage,
  fresh-namespace, no-learning, and WSL/Windows/NVIDIA ownership check.
- Contract SHA-256 is
  `1c864557d6302613d111eccbafd914673d88316c2bcbc659e1afb028c7fec176`;
  Git blob is `bad4fb8f9ce7ffdc493d4ad9de86016e7c5c642b`. Explicit
  `--execute` exits `7` with `runtime_authorization_not_issued`, and namespace
  `20260722_case78_camera_cap_v1_exclusive` remains absent.
- No runtime token, Isaac launch, dynamic proof, split change, case-78
  admission, label admission, holdout access, dataset, BC, PPO, or obstacle
  work occurred in this round.

## Next round after Round 145

- Perform a separate go/no-go review of the sealed CPU contract. A GO may add
  a one-use runtime/finalizer layer in a separate commit, pinned to the current
  blobs and exact `0.10 m` cap. It must keep recovery disabled, preserve the
  `5,400 s` wall bound and `2,000`-step heartbeat, consume its token before
  Isaac, and stop after exactly one case 78.
- Only a complete physical pass under every unchanged dynamic, thermal,
  controller-evidence, no-data, heartbeat, and GPU-release check can support a
  later immutable split decision. Failure leaves case 4 in validation and case
  78 unused.
- Do not create the runtime contract and token in the same unreviewed change.
  Do not open holdout, create labels/data, retrain BC, start PPO, or begin
  obstacle work during the runtime-contract review.

## Round 146: the bounded `0.10 m` camera-cap candidate passes dynamically

- Commits `8c00406`, `2034c40`, and `17b5243` add, lineage-pin, and seal the
  separate one-use runtime wrapper, authorization validator, and final
  summarizer. Commit `cfe8325` makes only the POSIX token-mode/symlink tests
  skip on Windows while preserving those checks in the WSL runtime validator.
  The complete authoritative `.98` suite passes `605` tests with seven
  intentional platform skips in `58.23 s`.
- Tokenless runtime preflight at clean pushed commit
  `cfe8325656ce4679273cfb95a45202f0a96106a2` passes every CPU/runtime
  identity, committed-blob, lineage, canonical-path, fresh-namespace,
  no-learning, and WSL/Windows/NVIDIA ownership check. Runtime contract
  SHA-256 is
  `2ec51603a2574ee486be9b3f9a2e1e5912ea4162106c3aab146b80a2ee3ae252`.
- The mode-`0600` token is consumed before Isaac starts. Exactly one case-78
  deterministic canary runs in namespace
  `20260722_case78_camera_cap_v1_exclusive`, with camera-error recovery
  disabled and the only controller change set to
  `maximum_camera_lever_arm_correction_m=0.10`. No policy, capture, dataset,
  split mutation, holdout access, BC, or PPO path is enabled.
- The canary completes the exact `135.487646/192.299567 s` source/execution
  clocks in `83,050` policy steps and `2,802.755613 s` host wall time at
  `29.658205` observed policy steps/s. The heartbeat is only `2.519 s` old at
  finalization. Playback exits zero, does not time out, and post-run WSL,
  Windows, and NVIDIA ownership is empty.
- All unchanged dynamic, thermal, controller-evidence, no-termination,
  heartbeat, wall-bound, GPU-release, and no-data checks pass. Position
  p95/max improves to `0.116601/0.184238 m`, versus the unchanged
  `0.15/0.25 m` gates. Pitch max is `7.568641 deg`; attitude p95/max is
  `0.147844/0.298857 deg`; riser max is `0.013690 m`; proxy max is
  `0.283882 deg`; proxy rate max is `54.511654 deg/s`; and all saturation
  ratios are zero.
- The applied camera correction reaches the exact `0.10 m` cap and remains
  saturated for `90.1613%` of samples; raw correction reaches `0.241389 m`.
  This is dynamic proof for the bounded model-based cap, not evidence that the
  full raw lever-arm request should be applied.
- The residual-label envelope independently remains false. No residual is
  applied and no labels are written, so dynamic qualification does not admit a
  dataset or training. Case 78 also remains `unused` until a separate immutable
  split change is reviewed and committed.
- Canonical SHA-256 values are final status
  `e413b0df0b09c3c04ac49130ca2d38c7e495364504c04f3e89d72521e8e5a4f6`,
  gate `304fa9e1202d4099f976e6933e9ffc21a2833e7cc380ab9f95d7473bf2126c73`,
  admission
  `2f799c7895c342aebb1e93b8617796c06259115d6be8b804bff84afea27523f0`,
  heartbeat
  `4bfd81db5187d0890d7a4a32b83e330ef8bd68a243a92c96ca9a0d0138990393`,
  and playback log
  `96cc2bfca165de989cd52aefc8dd556ededb48339fa3ae141637c0e33420f9b4`.

## Next round after Round 146

- Build a CPU-only immutable split-admission proposal pinned to the passed
  case-78 final/gate evidence and the existing transparent split-reset
  proposal. The change may move case 4 from validation to training and case 78
  from unused to validation, while keeping holdout `[3,5,13,19,24]` unchanged.
  Do not alter teacher arrays, source plans, or physical gates.
- Keep residual-label admission separate. Before creating any dataset, audit
  the passed run's raw residual distribution and define a new evidence-backed
  envelope or action contract without clipping. Dynamic qualification alone
  must not be used to relabel the current overflow as valid.
- Do not launch another Isaac run, capture labels, retrain BC, start PPO, or
  begin obstacle work during the split and label-envelope CPU reviews.

## Round 147: case 4/78 role swap is admitted without rewriting historical data

- Commit `0cd792a` adds a canonical-hash CPU builder and focused positive and
  negative tests for the post-qualification role swap. It binds the pending
  fallback proposal, immutable 403,569-row teacher summary, and passed case-78
  final/gate evidence. The complete authoritative `.98` suite passes `607`
  tests with seven intentional platform skips in `73.08 s`.
- The resulting manifest is
  `20260722_case4_case78_split_admission_v1_cpu/manifest.json`, SHA-256
  `eac2c8c5389b0a8e3590d5b6355eaa80b50019091d5eb906408a6599c19cb623`.
  Every input contract check passes.
- Split version `initial_teacher_case4_train_case78_validation_v2` is admitted
  for the next dataset build. Training contains the original 30 cases plus
  case 4; validation remains five cases `[8,16,22,32,78]`; holdout remains
  exactly `[3,5,13,19,24]`. These are the only role changes.
- The existing teacher artifact remains an immutable historical 30/5/5 split:
  no rows, arrays, actions, hashes, or source files are rewritten. Case 78 has
  no labels, so the role manifest does not claim the historical dataset already
  implements the new split.
- Split roles are admitted, but label capture, dataset creation, BC, PPO, and
  training validity remain false. The next dataset builder must consume this
  manifest explicitly and may not infer admission from case IDs alone.

## Next round after Round 147

- Perform a CPU-only raw-residual audit using the passed case-78 gate and the
  existing 40-case corpus envelope. Separate physical command authority from
  learned-label normalization, quantify per-channel maxima/quantiles and
  overflow duration, and propose one non-clipping action contract.
- The proposal must preserve the validated LQR/model-based inner controller and
  safety supervisor. A learned residual remains bounded above those loops; it
  must not inherit primary balance, hard-limit, or emergency authority.
- Do not capture case-78 labels, rebuild the dataset, run BC/PPO, open holdout,
  or launch Isaac until the new label contract is independently admitted.

## Round 148: the teacher-40 residual scale is retained without opening holdout

- Commits `c7815ed` and `4227228` add and repair a hash-pinned CPU audit of
  the passed case-78 gate, immutable teacher-40 summary, raw-corpus audit, and
  admitted split manifest. Focused tests pass `4/4` on both macOS and `.98`;
  the complete authoritative `.98` suite passes `611` tests with seven
  intentional platform skips in `61.45 s`.
- The audit reads only the 35 train/validation raw cases. It verifies every raw
  NPZ against the source-row SHA-256, covers `360,021` rows and `1,799.93 s`,
  and skips holdout `[3,5,13,19,24]` before any holdout file is opened.
- Development-label absolute residual p50/p90/p95/p99/p99.9 values are:
  `[0.055850,0.022630,0.009903]`,
  `[0.133988,0.083762,0.011782]`,
  `[0.166104,0.106735,0.012339]`,
  `[0.230667,0.143646,0.013442]`, and
  `[0.273634,0.170230,0.014399]`. Absolute maxima are
  `[0.294894,0.292132,0.017596]`.
- The existing teacher-40 normalization `[0.35,0.40,0.10]` has zero observed
  train/validation overflow samples and zero overflow duration. Combining the
  whole-corpus aggregate maximum with case 78's full-rate maximum gives
  `[0.302604,0.292132,0.017596]`, or normalized utilization
  `[0.864583,0.730330,0.175956]`; therefore no scale change or clipping is
  required for the current teacher-40 action contract.
- Case 78 still has only aggregate maxima, not a policy-rate raw-label series.
  Its old `[0.30,0.40,0.10]` runtime envelope overflow is preserved, and its
  overflow count, duration, and quantiles remain explicitly unknown. The audit
  does not infer or fabricate them.
- Canonical output is
  `20260722_case78_residual_action_contract_v1_cpu/audit.json`, SHA-256
  `fd2b97d5e4a6cada368f4fb776086ebcab10403df5b350d7fa16a694163b535c`.
  It retains the teacher-40 scale while keeping case-78 label capture, dataset
  creation, BC, PPO, and training false. No Isaac or GPU work occurred.

## Next round after Round 148

- Build a CPU-only, one-case shadow-label measurement contract for case 78. It
  must retain the passed `0.10 m` camera correction, deterministic LQR/model-
  based commands, all physical gates, and the `[0.35,0.40,0.10]` candidate
  normalization, while recording policy-rate raw labels and elapsed/source
  clocks without applying residuals or creating a dataset.
- The post-run audit must verify exact command reconstruction, signed minima
  and maxima, p50/p90/p95/p99/p99.9, and per-channel overflow count/duration.
  A failed physical gate or any candidate-scale overflow keeps case 78 outside
  the next dataset.
- The immutable historical teacher-40 dataset remains usable as the current
  initialization corpus and must not be rewritten merely to perform this
  measurement. Do not open holdout, train BC, start PPO, or begin obstacle work
  while preparing the shadow-only contract.

## Round 149: deterministic case-78 shadow-label measurement is CPU-ready

- Commit `9521653` extends the non-trainable shadow trace so it can observe
  deterministic-controller states without a residual policy. It does not
  modify the deterministic command branch. Zero-policy mode remains forbidden
  because that mode reconstructs commands from feedforward instead of retaining
  the dynamically passed model-based controller. Historical learned-policy
  shadow traces remain loadable through explicit legacy-source inference.
- Commits `0fff1f2`, `df1c45d`, `5aecda0`, `506d688`, `3ae04cd`, and
  `741babc` add, seal, schema-align, and test the CPU validator and preflight.
  The canonical contract is
  `scripts/two_wheel_balance/case78_shadow_label_cpu_contract_v1.json`,
  SHA-256
  `3cbe1b816ee78f216232f67f5b0fa387ad52cb11219e60fae692326cc09b4aba`.
- The contract pins case 78, the exact source/execution clocks and plan hash,
  passed `0.10 m` camera-cap gate/final evidence, LQR gains, robot USD,
  playback/serializer/tracking/heartbeat blobs, split admission, and Round-148
  residual audit. Controller arguments remain deterministic with no residual
  policy, no zero-policy shortcut, and candidate scales `[0.35,0.40,0.10]`.
- The trace contract requires policy-rate elapsed/phase clocks, raw and
  normalized labels, exact teacher-command reconstruction fields, zero applied
  residual actions, deterministic-controller visited-state metadata, and no
  dataset or training validity.
- Final live `.98` preflight at clean pushed commit
  `741babc6e58cc2f4e11b4d4dad1795859fde2e88` passes all 17 semantic,
  identity, lineage, namespace, and ownership checks. Preflight output SHA-256
  is `625a18f5a8cf3f7e2af2191d7e7231c10b9cd8991c2f0fc3c171684931c88a6f`.
  The complete authoritative `.98` suite passes `623` tests with seven
  intentional platform skips in `58.75 s`.
- Runtime authorization, GPU launch, shadow measurement, label capture,
  dataset creation, BC, PPO, holdout access, and training remain false. No
  namespace, token, Isaac process, or new trace was created in this round.

## Next round after Round 149

- Review the sealed CPU contract independently. If approved, create a separate
  one-use runtime/finalizer layer pinned to the current contract and blobs. The
  token must be mode `0600`, consumed before Isaac, and authorize exactly one
  case-78 deterministic shadow trace in the fresh namespace with a `5,400 s`
  wall bound and `2,000`-policy-step heartbeat.
- Finalization must independently require unchanged dynamic/thermal/controller
  gates, zero applied residual actions, exact command reconstruction, both
  clocks, full-rate signed extrema and p50/p90/p95/p99/p99.9, and per-channel
  overflow count/duration against `[0.35,0.40,0.10]`. It must always verify GPU
  release and preserve failed evidence.
- A successful shadow measurement may admit case-78 labels for a future split
  rebuild; it does not itself authorize dataset creation or training. In
  parallel, the immutable teacher-40 corpus remains the valid initialization
  corpus for a separately reviewed BC authorization. PPO and obstacle work
  remain closed.

## Round 150: case-78 deterministic shadow labels pass at policy rate

- Commits `54d1712`, `20cdfdf`, `a078bf6`, `9168e51`, and `54aa067` add,
  lineage-pin, authorize, repair, and reseal the one-use runtime/finalizer
  layer. Tokenless live preflight passes all 20 contract checks. The complete
  pre-run `.98` suite passes `628` tests with nine intentional platform skips
  in `59.84 s`.
- A mode-`0600` token containing only
  `AUTHORIZED_CASE78_DETERMINISTIC_SHADOW_LABEL_V1` is hash-verified and
  consumed before Isaac. Exactly one deterministic case-78 run executes in
  `20260722_case78_shadow_label_measurement_v1_exclusive`; no residual policy,
  zero-policy shortcut, raw-teacher capture, normalized dataset, BC, or PPO is
  enabled.
- The run exits zero after `2,843.285444 s`, completing exact source/execution
  clocks `135.487646/192.299567 s` in `83,050` policy steps. Position p95/max
  is `0.116601/0.184238 m`; pitch max is `7.568641 deg`; attitude p95/max is
  `0.147844/0.298857 deg`; riser max is `0.013690 m`; proxy max/rate are
  `0.283882 deg` and `54.511654 deg/s`; no termination occurs.
- The shadow trace contains exactly `83,050` policy-rate rows over `415.245 s`.
  Raw residual signed minima/maxima are
  `[-0.302604,-0.142180,-0.000532]` and
  `[0.109112,0.185892,0.013768]`. Absolute p50/p90/p95/p99/p99.9 values are
  `[0.053185,0.033577,0.010074]`,
  `[0.094985,0.081567,0.011788]`,
  `[0.110629,0.103015,0.012245]`,
  `[0.215335,0.137043,0.012785]`, and
  `[0.291322,0.164892,0.013412]`.
- Absolute maxima `[0.302604,0.185892,0.013768]` normalize to
  `[0.864583,0.464730,0.137677]` under `[0.35,0.40,0.10]`. Overflow sample
  counts and durations are exactly zero for all channels. Applied residual
  actions are exactly zero; raw-normalization, teacher-command reconstruction,
  and deterministic-command match maximum errors are `2.38e-8`, `2.98e-8`,
  and `0.0` respectively.
- Every admission, heartbeat, gate, physical, trace, and filesystem check
  passes. Final status SHA-256 is
  `63004e41d1185a8589c8715e620a5c976db44bfb6a130786214109a1ab2d5bd7`;
  gate is
  `ad0dc3ee618819ec808ac4d0318bda711dc2cba38dd041119a1f78584e97e459`;
  trace is
  `dc04cbef0aab9960018579292b9ff9ee25e8bd427cd4be641b4a9d96e04525e3`;
  admission is
  `7acab3500762f55546d33335df6a60f172d8989b80f88aa6cd37d189aa7cc7b0`.
- Post-run WSL, Windows `kit`, and NVIDIA ownership are independently empty;
  the token is absent and exactly one trace exists. The trace remains
  non-trainable evidence: label admission, dataset creation, BC, PPO, and
  holdout access are still false.

## Next round after Round 150

- Perform a CPU-only label-admission audit pinned to the Round-150 final, gate,
  trace, admission, and Round-148 scale audit. It must independently verify the
  deterministic source, exact clocks, zero applied actions, command
  reconstruction, zero scale overflow, semantic camera contract, and no
  training side effects.
- If admitted, convert the trace immutably into the existing scale-independent
  raw-teacher schema and rebuild a new 31/5/5 dataset from the 40 accepted raw
  teachers plus case 78, consuming the sealed split manifest. Preserve the old
  403,569-row dataset and keep holdout cases unopened during policy selection.
- Dataset/schema/split admission and BC authorization remain separate CPU
  decisions. Do not start BC, PPO, or obstacle work merely because the shadow
  trace passed.

## Round 151: case 78 enters a resealed 41-case initialization dataset

- Commit `3011946` adds three CPU-only boundaries: a hash-pinned case-78 label
  admission, immutable shadow-to-raw conversion, and an explicit teacher-40 plus
  case-78 dataset builder. The tools pin the Round-150 final/gate/trace/runtime
  admission, Round-148 scale audit, sealed split manifest, and original
  teacher-40 dataset identities. They do not launch Isaac or authorize BC/PPO.
- Canonical case-78 label admission is
  `20260722_case78_shadow_label_admission_v1_cpu/label_admission.json`, SHA-256
  `cd752e402c912d7a83767544c8059d2068979d10868dff2408fd93836b71033d`.
  All 15 checks pass: exact source/execution clocks, deterministic visited
  states, semantic physical-camera contract, zero applied and previous-action
  placeholders, exact command reconstruction, and zero overflow under
  `[0.35,0.40,0.10]`.
- The immutable raw-teacher conversion contains `83,050` policy-rate rows in
  `case_0078_executed_raw_teacher_v1.npz`, SHA-256
  `1dc0dec589ed365683968ca4b8a2b1611e16347dd18a6c183bc8d463dcca6924`.
  Its summary SHA-256 is
  `c1456b360de4af15bbf65d7f59ef3b0e3f3c453c3be3857cf41f38e2e56a81df`.
  The source shadow trace is not modified and raw labels were never applied to
  physical commands.
- The first dataset output in
  `20260722_initial_teacher41_case78_31_5_5_v2_cpu` is preserved as a
  provisional build because its metadata inherited the historical corpus path
  and random seed. It is superseded and must not be used for BC. The resealed
  canonical namespace is
  `20260722_initial_teacher41_case78_31_5_5_v2_resealed_cpu`.
- Canonical dataset `initial_teacher41_case78_31_5_5_v2_resealed.npz` has
  `41` cases and `486,619` rows with exact split `31/5/5`: case 4 moves to
  train, case 78 enters validation, and holdout remains
  `[3,5,13,19,24]`. Dataset SHA-256 is
  `03e3f2b8b4a6b7626a9b43f1fb2a88cbbfdfceb4b6373a51abdb21590bf53497`;
  summary SHA-256 is
  `2b7b177f481fdc632aca2134d9eea69cec66814581a5e39d9c6a099e3d8bcbfb`.
- Production-loader audit SHA-256
  `7a764d9cc41e9d43dd808251e2b8466e7ef0940bd356cbebeee38ffcd88e34cb`
  passes all 12 schema, row, case, split, provenance, and learning-closed checks.
  No holdout policy metrics are computed. The original teacher-40 dataset and
  summary remain byte-identical at SHA-256
  `53f3b679e227446c6008ba8bcd9191ae877b946dd86644388c43f89723bb9d44`
  and `815463ffa133addbaec4f09a453fd9dae8e63eb690b37f56fd0a5c1877879542`.
- Focused local and `.98` regression sets pass `44/44`. The complete
  authoritative `.98` suite passes `632` tests with nine intentional skips in
  `59.43 s`. BC, PPO, learned rollout, obstacle work, and holdout evaluation
  remain closed.

## Next round after Round 151

- Build a separate CPU-only BC admission pinned to commit `3011946`, the
  canonical dataset/summary/loader-audit hashes, the label admission, and the
  original teacher-40 identities. The authorization must select only by the
  five validation cases and keep holdout unopened.
- Before issuing any one-use training token, audit the previous teacher-40 BC
  experiments and choose one bounded architecture/previous-action contract.
  Do not silently reuse a failed historical policy merely because the new
  dataset loader passes.
- If separately authorized, train BC only above the frozen LQR/model-based
  controller. PPO, learned Isaac rollout, and expansion beyond the admitted 41
  cases require later independent gates.

## Round 152: masked previous-action BC is selected but not yet authorized

- Commit `56f28b9` adds a CPU-only architecture/admission contract that pins
  the canonical 41-case dataset, summary, production-loader audit, case-78
  label admission, and all relevant teacher-40 BC comparison evidence. It
  verifies clean Git ancestry from dataset implementation commit `3011946`.
- Historical evidence rejects blind architecture retuning. Original BC passed
  only its offline gate. Masked previous-action BC was the best dynamic
  near-pass: case-4 position p95 `0.137441 m` passed the absolute `0.15 m`
  gate and strongly beat zero residual, but remained worse than the teacher's
  `0.128736 m`. Scheduled sampling and both attenuated previous-action variants
  failed their comparison gates.
- Canonical CPU contract is
  `20260722_initial_teacher41_masked_bc_contract_v1_cpu/contract.json`, SHA-256
  `41bef3b5b39eb216ecc69cead67b4668424ab79a28a8b8da9df54b58e653dd84`.
  It selects architecture
  `state_shared_lookahead_fusion_previous_action_masked_v1`, 80 epochs maximum,
  patience 10, batch size 4096, seed `20260722`, and validation-only model
  selection over `[8,16,22,32,78]`.
- The contract orders any later dynamic canary as case 8 first and case 78
  second, with no automatic case-78 or broad rollout. It issues no runtime
  token and keeps BC training, learned rollout, PPO, holdout access, and
  runtime validity false.

## Next round after Round 152

- Add a one-use, fail-closed masked-BC training wrapper pinned to the Round-152
  contract, dataset, production trainer, clean HEAD/upstream identity, and an
  exclusive GPU ownership check. The token must be consumed before CUDA work.
- Finalization must independently verify validation-only offline improvement,
  exact architecture/mask/scales/dataset identities, deterministic seed,
  checkpoint and TorchScript hashes, no holdout metrics, no learned rollout,
  and no PPO. Training success alone must not authorize deployment.
- Only after a separately sealed offline pass may a bounded case-8 Isaac canary
  be proposed. Do not launch case 78 automatically.

## Round 153: teacher-41 masked BC passes offline without opening runtime

- Commit `68a9443` adds the one-use teacher-41 masked-BC wrapper. It pins the
  dataset, summary, production-loader audit, Round-152 CPU contract, trainer,
  policy module, clean pushed HEAD, and reviewed dataset lineage. Preflight
  passed with WSL, Windows, and NVIDIA ownership empty.
- A mode-`0600` one-use token was hash-checked and consumed before CUDA work.
  Offline BC completed in
  `20260722_initial_teacher41_masked_bc_v1` at training commit `68a9443`, with
  architecture `state_shared_lookahead_fusion_previous_action_masked_v1`,
  seed `20260722`, best epoch 74, and all 80 bounded epochs completed.
- Validation-only case-balanced MSE improved over zero residual in all three
  channels: candidate `[0.008421,0.000916,0.00000407]` versus zero
  `[0.060097,0.011948,0.010376]`. Prediction absolute maxima
  `[0.845344,0.546735,0.135830]` remain inside the normalized action envelope.
- Admission/report/final SHA-256 values are respectively
  `dc7a0a0d441917f80e7f5a4bcb806d059dab6d365459e3712e4c5d5193638d99`,
  `b7915caddea9467847430a247924eae2e856ad486da06135e1b8f543c42b891a`,
  and `d232ffac1f67d1a4510e2ad7e6670f82742b259433ed0ce6b15727c4e39db3d9`.
  Checkpoint SHA-256 is
  `dcd7d811b1c882be7fe8c9f5e9361da823591c1e61149f29302ac0cc57fbb52f`;
  TorchScript SHA-256 is
  `0d796c600c6dca7dce176da555f4cd1f769163f41093d2b6313f4e6264888db7`.
- Finalization passes all 17 identity, architecture, split, offline-gate,
  checkpoint, no-rollout/no-PPO, and GPU-release checks. The token is absent,
  no training/playback process remains, and holdout metrics were not computed.
  The policy is only `case8_canary_proposal_ready`; learned rollout remains
  unauthorized and case 78 must not launch automatically.
- The complete authoritative `.98` suite now passes `636` tests with nine
  intentional skips in `59.56 s`.

## Next round after Round 153

- Build a CPU-only case-8 learned-policy canary contract pinned to the
  Round-153 final/report/TorchScript, exact case-8 smoothed plan, frozen LQR and
  controller identities, and deterministic teacher reference evidence.
- The canary must be exclusive, one case only, preserve the physical camera and
  semantic DFR attitude contract, apply the same residual scales, write no
  dataset, and keep holdout/PPO closed. Compare learned tracking against both
  the deterministic teacher and zero residual under unchanged hard gates.
- Do not authorize case 78 or broad rollout unless case 8 passes both absolute
  dynamic gates and the separately declared teacher-regression budget.

## Round 154: case-8 learned-policy runtime contract is admitted CPU-only

- Commit `b0fa962` adds a committed case-8 CPU contract and fail-closed
  validator. The contract pins the exact v9 dynamic-retime plan, both case-8
  deterministic teacher records, the teacher-41 masked TorchScript policy,
  frozen 28 kg LQR/controller/robot identities, and the semantic DFR-to-physical
  camera contract. It contains no runtime token and authorizes no GPU launch,
  dataset, case 78, holdout, BC, or PPO work.
- The selected plan is
  `20260718_smoothed_plan_all79_v9_case8_dynamic_retime_cpu/case_0008_smoothed_riser_plan_v1.npz`,
  SHA-256
  `f07ff020128dee70ea9c8c2d806dc75c8e0ef3964dccb4e0aabfd1b0048f3655`.
  It preserves `663/663` source/execution states with distinct source/execution
  durations `12.940941/18.1173174 s` and passed all timing, transition, and
  kinematic checks.
- The deterministic teacher reference remains
  `20260718_gate_c_smoothed_case8_dynamic_retime_v1_exclusive/gates/case_0008.json`,
  SHA-256
  `19506045f9b6ec04cee58efa1b5d2d5600824ce166b1534db05a4895596cf1e0`.
  Its position p95/max errors are `0.131254/0.143331 m` under the unchanged
  camera lever-arm and structural controller settings.
- Real-artifact CPU admission is
  `20260722_initial_teacher41_masked_bc_case8_canary_contract_v1_cpu/admission.json`,
  SHA-256
  `59b270beb7ef6e9b6a919ef85ec54599465a0ff10a2b5842c1f268326e7ff057`.
  All 32 semantic, identity, Git, namespace, and learning-closed checks pass.
  The canonical contract SHA-256 is
  `18c48f566fe4ed04977f601cebccb0bbea062359695d541aef9eaf65353025f6`.
- Commit `02bcdfc` adds the one-use zero-then-learned runtime wrapper and a
  robust finalizer. The authorization is consumed before Isaac, both runs use
  the same case/plan/controller/scales, and no capture option is available.
  The complete authoritative IsaacLab Python suite passes `647` tests with
  nine intentional skips in `62.74 s`.

## Round 155: teacher-41 masked BC passes the case-8 dynamic canary

- The exclusive namespace is
  `20260722_initial_teacher41_masked_bc_case8_canary_v1_exclusive`, run at
  commit `02bcdfc`. The one-use token is absent after the run; WSL playback,
  Windows Kit, and NVIDIA compute ownership are empty, and the tracked worktree
  remains clean.
- The zero-residual baseline does not complete the reference: it reaches phase
  `6.851671/18.117317 s` in `10,872` steps with position p95/max
  `1.438923/1.502709 m`. It terminates only at the bounded horizon, with no
  physical termination and no residual action or dataset.
- The learned policy completes the full reference in `6,611` steps. Position
  p95/max are `0.133671/0.144092 m`; attitude p95/max are
  `0.155519/0.231078 deg`; pitch p95/max are `5.969875/6.144382 deg`; riser
  servo p95/max are `0.010924/0.011907 m`; proxy servo p95/max are
  `0.120648/0.198412 deg`. Saturation is zero and there is no termination.
- Normalized residual absolute maxima are `[0.658091,0.357680,0.125535]`,
  inside the unit envelope. Learned position p95 is only `1.84%` above the
  deterministic teacher and `90.71%` below the zero baseline, so it passes the
  fixed 5% teacher-regression and minimum-zero-improvement budgets. Every
  per-metric hard and comparison check passes.
- Runtime admission, zero, learned, comparison summary, and final SHA-256
  values are respectively
  `9d6297741250e56c488c7560abf9b3b5c7f36ce48bb823dda529a7a4c76eb314`,
  `03f4b59df1f5a38021f984ed6d1fa3af72d31f037727b01d96e6d77a3e0dd682`,
  `f1e1160f7f503ea3e377cf502c58c9d054d8584bd92055d291b70ea6e4e5c971`,
  `bcf1d892216902590c259fbd325d35b9cb618152ea200bdee16ed31d91948c30`,
  and
  `681bdaa0a2e20d4182d5de944f9b2d079b7b9e87a565070c0d976829a2c69446`.
- This is a successful one-case validation canary, not broad policy admission.
  No dataset was created, case 78 and broad rollout remain unauthorized,
  holdout remains unopened, and BC/PPO remain closed.

## Next round after Round 155

- Build a separate CPU-only case-78 learned-policy canary contract. Pin the
  admitted case-78 plan/teacher evidence, Round-155 case-8 final and summary,
  the same TorchScript policy, controller, camera contract, and unchanged hard
  and comparison thresholds.
- Run case 78 only after a fresh one-use authorization and exclusive ownership
  check. Compare deterministic teacher, zero residual, and learned residual;
  do not reuse the prior shadow trace as learned dynamic evidence.
- If case 78 passes, evaluate the remaining validation cases 16, 22, and 32 as
  a bounded tranche before any train-split expansion, corrective data capture,
  holdout access, or PPO decision.

## Round 156: case-78 learned-first dynamic canary is admitted and active

- Commit `50975ea` adds a CPU-only case-78 contract builder. Its canonical
  output is
  `20260722_initial_teacher41_masked_bc_case78_canary_contract_v1_cpu/contract.json`,
  SHA-256
  `4eb446d8a205d79796c53aecc30ad4f20b6b00a90ed7f346c3b9f864bed7c334`.
  All ten case-8 prerequisite, case-78 plan/teacher/label, policy, clock,
  camera, and learning-closed checks pass.
- Case 78 uses the admitted v16 plan SHA-256
  `28c69e20778e738d1ac4a0ae299160ed5764089094c2a0f9a018c49790860569`,
  with `6,870/6,870` source/execution states and source/execution durations
  `135.487646/192.29956737098348 s`. The deterministic teacher is the sealed
  shadow-label gate SHA-256
  `ad0dc3ee618819ec808ac4d0318bda711dc2cba38dd041119a1f78584e97e459`.
- The case-78 teacher uses a `0.10 m` camera lever-arm correction cap. This is
  intentional and differs from case 8's `0.05 m`; the CPU contract rejects
  silently substituting the case-8 value. Teacher position p95/max are
  `0.116601/0.184238 m`, and the same run supplied the admitted case-78 labels
  in the teacher-41 validation split.
- Commit `0cd6548` adds the generic validation-canary finalizer and the
  one-use case-78 wrapper. The wrapper is learned-first: a fresh zero baseline
  is launched only if the learned rollout passes physical gates. Both runs are
  still required for final admission, but an unsafe learned policy fails early
  without spending the full zero horizon. Combined timeouts are bounded at
  `10,800 s`.
- Runtime preflight passes at commit `0cd6548`; the complete authoritative
  IsaacLab Python suite passes `656` tests with nine intentional skips in
  `59.55 s`. The one-use token was consumed before Isaac. The exclusive
  learned rollout is active under wrapper PID `338393` in namespace
  `20260722_initial_teacher41_masked_bc_case78_canary_v1_exclusive`.
- At the latest recorded heartbeat (`4,000` policy steps), phase is
  `7.954837/192.299567 s`; current position error is `0.066356 m`, peak
  position error is `0.140857 m`, peak attitude error is `0.204020 deg`, and
  peak pitch is `7.278748 deg`. Saturation, termination, and dataset creation
  remain zero/false. This is live progress only, not a completed canary claim.

## Round 157: remaining validation tranche is prepared CPU-only

- Commit `a29028a` adds
  `build_initial_teacher41_validation_tranche_contract.py` and focused
  fail-closed tests. It prepares cases `[16,22,32]` only after the sealed case
  78 learned/zero comparison passes; it does not issue an authorization token,
  create a runtime namespace, open holdout, or authorize capture, BC, or PPO.
- The pinned case 16/22/32 plan SHA-256 values are respectively
  `742d1f705d3559916c3e1d7d35caffd5ea9e7200b6e321d1f9f70c8e5a7dad16`,
  `8f1638cd771cfac32ca251906e2c095bd7091edb2561974f12ae09b0a65d4a79`,
  and
  `71b1986633613fdb13585ac4c12870addc553ad12e895b05cc424a83cf4e037f`.
  Each preserves source/execution clocks and passes all plan timing,
  transition, and kinematic checks.
- Their deterministic teacher gate SHA-256 values are respectively
  `8915d532633c52a7727fd24514141aa122d87cf593bc123d9a2776f2552a000a`,
  `115623a6f1239b9e4fc78a7a60087a176b340f275f817f123c90f593e943892a`,
  and
  `d2a7477254d6a80426370217d8f08db8fe2bdf65e5f4b892a33247f90cf1ce75`.
  The contract also pins each raw-capture case audit, proving exact capture
  admission, both clocks, reconstruction accuracy, zero applied residual, and
  closed training state.
- These three cases use the normal `0.05 m` camera lever-arm correction cap;
  case 78 intentionally retains `0.10 m`. The builder rejects applying the
  case-78 cap to the remaining tranche.
- Four new focused tests and the ten related case-78/finalizer tests pass
  (`14 passed`). At the latest live readback, case 78 had reached `32,000`
  policy steps and phase `66.224982/192.299567 s`; position error was
  `0.044472 m`, peak position error remained `0.209184 m`, peak pitch remained
  `8.140261 deg`, and saturation/termination remained zero/false.
- The live `.98` worktree deliberately remains at runtime commit `0cd6548`;
  pushed CPU-only commit `a29028a` must not be pulled there until the case-78
  wrapper and any conditional zero baseline have closed.

## Round 158: teacher-41 planner-imitation BC fails case 78 position p95

- The learned-first case-78 canary closed at runtime commit `0cd6548` without
  launching its conditional zero baseline. The learned result, final status,
  and admission SHA-256 values are respectively
  `570546d39b4267d20c6c203a1eb1a3a04a7544cb2887270c9d99a6b952ad9c41`,
  `c14467a93622a887b861b7a8628ec1ff13b9cdf0d1a2380336b8ee9963172f1b`,
  and
  `a28229753508c6666e7cb1223b151f57424a3ad4b3480bf65b2a0a163dd1ea6d`.
- The policy completed the full `192.299567 s` execution reference in `85,923`
  steps with no termination or saturation. Position max `0.231194 m` passes
  the unchanged `0.25 m` bound, but position p95 `0.165018 m` fails the
  unchanged `0.15 m` bound. Every other physical, attitude, balance, riser,
  proxy, thermal, initialization, and evidence check passes.
- The fail-closed wrapper recorded learned exit `0`, skipped zero and comparison
  with status `125`, and finalized with exit `6`. No zero output, comparison
  summary, dataset, capture, BC, PPO, or holdout artifact was created. GPU and
  playback ownership are empty and the one-use authorization token is absent.
- Commit `de5174f` adds a hash-bound CPU failure audit. Its canonical report is
  `20260722_initial_teacher41_case78_failure_audit_v1_cpu/report.json`,
  SHA-256
  `97c90a0dc56450e4dc71654ac588eeffd09bd5d0db92bc3a4fbae265709241fd`.
  It identifies eight high-error trace intervals and confirms that the only
  failed dynamic check is `position_p95_bounded`.
- Compared over all 431 phase-aligned trace samples, inferred normalized action
  MAE is `0.029019` for `vx` and `0.025637` for `wz`. In the 27 high-error
  samples it rises to `0.042789/0.062527`; sign mismatch reaches
  `7.41%/11.11%`. Learned progress mean falls to `0.447620` versus teacher
  `0.463101`, with 46 progress-hold steps versus zero, increasing time spent in
  transient camera-error regions.

## Round 159: restore model-based-plus-residual policy layering CPU-only

- The failure audit proves the deployed BC contract was planner imitation:
  labels were `deterministic_planner_command - phase_feedforward`, and learned
  mode applied `phase_feedforward + BC_prediction`. This substitutes a learned
  reconstruction for the complete model-based planner command; it does not
  satisfy the required `u_final = u_model + bounded_delta_u_policy` contract.
- The existing TorchScript checkpoint is therefore classified as
  `planner_imitation_bc_initialization_only`, not a final residual policy.
  Threshold relaxation, the cases 16/22/32 tranche, corrective capture, BC
  retraining, PPO, and holdout access remain unauthorized.
- Commit `97c90e5` adds the explicit
  `model_based_planner_plus_bounded_policy_residual_v1` command composer. A
  zero normalized action reproduces an admissible complete model command
  exactly; normalized overflow fails closed; the safety supervisor still
  clamps final velocity, yaw-rate, and riser bounds. Initial residual scales
  are tightly limited to `[0.05 m/s,0.05 rad/s,0.02 m]`.
- The same commit adds a zero-initialized residual action head and an encoder
  transfer helper. The accepted BC encoder can be reused, but all residual-head
  weights and biases reset exactly to zero, including TorchScript execution.
  Seventy-two focused dataset, policy, audit, and teacher-41 tests pass locally.
  No runner mode, checkpoint, runtime authorization, namespace, or GPU process
  is introduced by this CPU-only change.

## Round 160: seal the exact-zero model-based residual initialization

- Commit `d46b70c` integrates the explicit model-based command base into
  `smoke_riser_reference_playback.py`. In that mode the deterministic planner
  remains authoritative and the policy may supply only a bounded normalized
  residual with fixed scales `[0.05,0.05,0.02]`. Dataset, raw-command,
  policy-trace, and shadow-trace capture remain rejected by the runtime mode.
- Commit `050f3da` adds the CPU-only checkpoint builder and negative contract
  tests. Commit `2d7ca8c` repairs native Windows Python access to the
  WSL-managed Git worktree identity without accepting a caller-provided HEAD.
  The `.98` authoritative suite passed `685` tests with nine intentional skips
  at `050f3da`; the final focused builder/dataset/policy/runtime suite passed
  `43` tests at `2d7ca8c`.
- The canonical CPU output is
  `20260722_model_based_zero_residual_policy_v1_cpu`. Its checkpoint,
  TorchScript, and report SHA-256 values are respectively
  `60377ad7b8b6618b614f9bd272a596574717ca436fcceaa1da827739d0f9e6d2`,
  `b1494f7af219d44cf966d7ba7781370afc1e8fe9575dd4e414d6ec0b7ea1ab19`,
  and `55e3ab5cd1ad2c8ee3aac12b4f834b2db90c5a9704c2fe815dd477f95049ef7e`.
  It binds source commit `2d7ca8cc4676a5ff680049700562475940fec3b7`,
  planner-imitation checkpoint SHA-256
  `dcd7d811b1c882be7fe8c9f5e9361da823591c1e61149f29302ac0cc57fbb52f`,
  and case-78 failure-audit SHA-256
  `97c90a0dc56450e4dc71654ac588eeffd09bd5d0db92bc3a4fbae265709241fd`.
- Independent resealing confirms the transferred encoder and exactly zero
  eager/TorchScript action head. Runtime authorization, training authorization,
  training-started, PPO, holdout, and valid-for-training flags are all false.
  No Isaac process, GPU namespace, residual dataset, or learned action was
  created during this round.

## Next round after Round 160

- Build a fresh CPU-only, hash-bound case-8 canary contract for the
  `model_based_planner_plus_bounded_policy_residual_v1` runtime mode. Pin the
  exact-zero TorchScript artifact, case-8 plan and deterministic teacher,
  controller/USD identities, unchanged physical gates, and one-use ownership
  requirements; do not issue a runtime token in the contract-building commit.
- After authoritative tests and an explicit one-use authorization, run case 8
  in an exclusive namespace. The zero-residual rollout must reproduce the
  deterministic planner within a declared deterministic tolerance and create
  no dataset. Only that proof can admit a bounded case-78 repetition.
- Keep cases 16/22/32, residual target capture, BC retraining, PPO, holdout
  access, and wider trajectory expansion closed until the zero-residual
  preservation sequence passes.

## Round 161: admit the case-8 zero-residual preservation canary CPU-only

- Commit `28709a2` adds the canonical CPU-only case-8 preservation contract and
  validator. The sealed contract SHA-256 is
  `47d4de888b0eeba0dbebad066f7ef106f3042fe9d63bf174f551dfddfbc1185b`;
  the `.98` admission SHA-256 is
  `001a27c14305d3c04f502a203da7f8474c451a842f11b3f0848d1b8da2f4a0de`.
  Every pinned plan, teacher, robot, LQR, planner, runtime, checkpoint, and
  committed-blob identity passes at runtime commit `28709a2`.
- The two admitted command paths are model-based planner plus explicit zero
  policy action and the same model-based planner plus the exact-zero
  TorchScript checkpoint. Both use fixed scales `[0.05,0.05,0.02]`; the old
  phase-feedforward planner-imitation path and `[0.35,0.40,0.10]` scales are
  explicitly rejected.
- Commit `e1f02ea` adds a fail-closed finalizer. It requires both rollouts to
  complete case 8, pass unchanged physical gates, report zero residual action,
  create no dataset, and remain within declared metric-delta tolerances. It
  does not itself authorize case 78, capture, training, PPO, or holdout use.
- Commit `140cc78` adds the complete exclusive wrapper with the runtime
  authorization hash intentionally empty. `--execute` therefore exits before
  environment inspection, namespace creation, or Isaac. The full authoritative
  `.98` suite passes `701` tests with nine intentional skips in `64.57 s`.
- Read-only preflight passes at `140cc78`; HEAD equals upstream, tracked files
  are clean, every pinned hash matches, GPU ownership is empty, and the fresh
  namespace is
  `20260722_model_based_zero_residual_case8_canary_v1_exclusive`. No runtime
  token, namespace, Isaac process, dataset, BC, PPO, or holdout artifact exists.

## Next round after Round 161

- Review and issue exactly one runtime authorization in a separate commit,
  create one mode-0600 token, and rerun the same preflight immediately before
  execution. Consume the token before the first Isaac process.
- Execute explicit zero first. Only if its result exists and GPU ownership is
  released may the exact-zero TorchScript rollout start. Finalize from the two
  sealed JSON outputs and stop on any physical, completion, zero-action,
  metric-preservation, identity, or dataset-absence failure.
- Do not start case 78, residual capture, BC, PPO, or holdout evaluation from a
  case-8 failure. A case-8 pass is only the prerequisite for a separately
  bounded case-78 preservation contract.

## Round 162: exact-zero residual preserves case-8 model-based behavior

- Commit `3c45dbd` issues one authorization hash in a separate change; commit
  `cf0d4d0` corrects preflight evidence to distinguish the issued hash from an
  unconsumed token. Focused authorization/preflight tests pass, and the final
  preflight at `cf0d4d0` confirms clean pushed HEAD, all pinned identities, a
  fresh namespace, and exclusive GPU ownership.
- The one mode-0600 token was consumed before Isaac. Explicit zero ran first;
  the exact-zero TorchScript rollout started only after the first result was
  written and ownership released. Both exit codes are zero, all processes are
  closed, the token is absent, and GPU compute ownership is empty.
- The canonical namespace is
  `20260722_model_based_zero_residual_case8_canary_v1_exclusive`. Explicit-zero,
  zero-checkpoint, runtime-admission, and final-status SHA-256 values are
  respectively
  `5e25d4ddc8184f7e9ee86253118f29aedc7da7a611846fae3d91397a2ac26fa3`,
  `96ab713c51fd00d93bbb5bf5936d9f996c2e473ea91cec5ed2881db1124fd4bc`,
  `83f244210eddbd7e49bcd02cca37567fa1d60ce8ea863eda9942d8fe7c5cab4d`,
  and `24922a7a08e9262b6159732aac5dbee6689ffe13a46b0bb95d29182437c66d9d`.
- Both paths complete execution time `18.1173174 s` in `6,605` policy steps.
  Position p95/max are `0.131254/0.143331 m`; attitude p95/max are
  `0.148800/0.223093 deg`; pitch max is `6.147057 deg`; riser max error is
  `0.0116785 m`; proxy max error is `0.229141 deg`. Every metric delta between
  the two executions is exactly zero.
- All physical, completion, policy identity, zero-action, and dataset-absence
  checks pass. Residual action remains `[0,0,0]`, saturation and termination
  remain absent, and no `.npz` is created. This proves the new policy layering
  starts as an exact no-op on case 8; it does not authorize capture, BC, PPO,
  holdout access, or case 78 by itself.

## Next round after Round 162

- Build a separate CPU-only case-78 preservation contract that pins the sealed
  case-8 pass, the admitted case-78 plan and deterministic teacher, the same
  exact-zero policy, and case 78's intentional `0.10 m` camera lever-arm cap.
  Keep `[0.05,0.05,0.02]` residual scales and all physical thresholds fixed.
- Do not reuse the planner-imitation case-78 wrapper or its failed learned
  result as runtime authorization. The new sequence must compare model-based
  explicit zero with the exact-zero checkpoint and require the same no-dataset
  preservation contract.
- Only a separately authorized case-78 preservation pass may open the design of
  residual-target capture. BC, PPO, holdout use, and cases 16/22/32 remain
  closed meanwhile.

## Round 163: exact-zero residual preserves case-78 model-based behavior

- Commits `1f1ea03`, `6e90944`, and `636fd7c` add the CPU contract, fail-closed
  finalizer, and guarded exclusive wrapper. Commit `067bb57` supplies one
  authorization hash in a separate change. The canonical CPU contract SHA-256
  is `cc958c9e6cb182509d4cf787891ca9586aefa08b7928f30f358d0ab5f0a4014c`;
  it pins plan SHA-256
  `28c69e20778e738d1ac4a0ae299160ed5764089094c2a0f9a018c49790860569`,
  deterministic-teacher SHA-256
  `ad0dc3ee618819ec808ac4d0318bda711dc2cba38dd041119a1f78584e97e459`,
  exact-zero policy SHA-256
  `b1494f7af219d44cf966d7ba7781370afc1e8fe9575dd4e414d6ec0b7ea1ab19`,
  and the intentional case-78 camera lever-arm cap of `0.10 m`.
- The canonical runtime namespace is
  `20260722_model_based_zero_residual_case78_canary_v1_exclusive`. Explicit-zero,
  zero-checkpoint, runtime-admission, and final-status SHA-256 values are
  respectively
  `12da451484212726903b87bd79d5d28d0dc98957c1622eaeb92f97528f672584`,
  `085006f92251db52f3ca5f752cc273b7005ba3f114a9c93a0aa9913de2945cdf`,
  `cbff2384c6daa9a5b34f6562e7f8fb9b671635c458b7700b29055e6ad0b4eff8`,
  and `b0b025d1d40da50a1a9273fb2ff5ba32966b4e1970c539ffbea24d753bc7bea3`.
- Both paths complete the full `192.299567 s` execution reference in `83,050`
  policy steps. Position p95/max are `0.116601/0.184238 m`; attitude p95/max
  are `0.147844/0.298857 deg`; pitch max is `7.568641 deg`; riser max error is
  `0.0136903 m`. Every audited metric is exactly identical between explicit
  zero and the exact-zero TorchScript checkpoint, and action saturation remains
  zero.
- Dynamic quality, thermal admission, completion, policy identity, and
  zero-action preservation pass. The prospective raw residual-label maximum is
  `[0.302604,0.185892,0.013768]`, which is normalized as
  `[1.008680,0.464730,0.137677]` against the historical measurement scales and
  therefore remains independently inadmissible as a label. It is neither
  clipped nor applied to commands and does not invalidate the deterministic
  dynamic pass.
- Both rollout exit codes and finalization exit code are zero. The one-use token
  is absent, GPU/playback ownership is empty, and no `.npz`, dataset, capture,
  BC, PPO, training, or holdout artifact was created. This closes the zero-no-op
  architecture-preservation sequence; it does not by itself authorize learning.

## Next round after Round 163

- Define a CPU-only corrective residual-target contract around the complete
  model-based planner. Labels must represent a demonstrably better admissible
  high-level command minus the exact model-based command, never planner
  reconstruction minus phase feed-forward and never physical DJI gimbal joints.
- Keep fixed residual bounds `[0.05 m/s,0.05 rad/s,0.02 m]` for the deployed
  policy until raw candidate labels are audited. Separate dynamic-quality,
  label-envelope, split-integrity, and training-admission outcomes; reject
  clipping and reject any capture from a dynamically failed rollout.
- Start with CPU schema, provenance, leakage, and synthetic label tests. Do not
  authorize Isaac capture, BC, PPO, or the unopened holdout cases until that
  contract proves how a nonzero target improves on the model-based baseline.

## Round 164: define a paired corrective-teacher experiment CPU-only

- Commit `88f6992` adds the pure
  `model_based_planner_plus_camera_error_corrective_teacher_v1` candidate and
  the fail-closed
  `same_seed_paired_dynamic_improvement_before_label_capture_v1` admission
  contract. No playback/runtime flag or command path is added by this commit.
- The teacher candidate uses only causal physical-camera position error in the
  base frame. Longitudinal, lateral-to-yaw, and vertical gains are
  `[0.20,0.30,0.30]`; deadbands are `[0.01,0.01,0.005] m`; physical residual
  limits are `[0.045 m/s,0.045 rad/s,0.018 m]`, strictly inside the deployed
  `[0.05,0.05,0.02]` policy envelope. Slew limits are
  `[0.10 m/s^2,0.10 rad/s^2,0.04 m/s]`.
- The first proposed paired experiment is training case 30 under the same plan,
  seed, physics, clocks, and previously measured deterministic perturbation.
  The complete model-based planner plus exact zero runs first; the corrective
  candidate may run only if that baseline passes. Neither rollout may capture
  labels or create a dataset.
- Corrective-target admission requires both rollouts to pass unchanged dynamic
  gates, a nonzero action below `0.95` normalized magnitude, position-p95
  improvement of at least `0.003 m` and `2%`, and bounded position-max,
  attitude, pitch, riser, and saturation regressions. Passing the pair remains
  diagnostic and does not itself authorize capture or training.
- The focused local suite passes `43/43`. The complete authoritative `.98`
  suite passes `731` tests with nine intentional skips in `58.58 s`. The
  canonical CPU proposal is
  `20260722_model_based_corrective_teacher_case30_proposal_v1_cpu/proposal.json`,
  SHA-256
  `51770e682ecaaa0a47495301a51e4304aa7dbc855db601c78cf8cb1b806b9a24`.
- The proposal binds case-8 preservation SHA-256
  `24922a7a08e9262b6159732aac5dbee6689ffe13a46b0bb95d29182437c66d9d`,
  case-78 preservation SHA-256
  `b0b025d1d40da50a1a9273fb2ff5ba32966b4e1970c539ffbea24d753bc7bea3`,
  case-30 perturbation SHA-256
  `fc08b890ec0dfee0a1f0d05505afd806f68c54ad564afc99fe5e73533e3ebfb6`,
  and split-summary SHA-256
  `2b7b177f481fdc632aca2134d9eea69cec66814581a5e39d9c6a099e3d8bcbfb6`.
  Case 30 is in train; cases 8/78 remain validation; holdout
  `[3,5,13,19,24]` is unchanged. Runtime, namespace, token, label capture,
  dataset, BC, PPO, training, and valid-for-training flags remain false.

## Next round after Round 164

- Add a separately reviewed, disabled-by-default runtime seam for this exact
  corrective candidate and paired telemetry. The baseline and candidate must
  expose the same model-planner command, corrective delta, final command,
  amplitude/slew-limit counts, plan/seed/physics identities, and independent
  dynamic outcomes.
- Keep the runtime authorization hash empty in that implementation commit and
  run the authoritative CPU suite plus negative override tests. Do not create a
  namespace or issue a token until a separate go/no-go review.
- If a later paired canary fails to improve case 30, reject this teacher rather
  than relaxing gates or treating its commands as labels. Only an admitted
  paired improvement can open a fresh, training-split corrective-label capture
  proposal; BC and PPO remain closed meanwhile.

## Round 165: add the disabled corrective-teacher runtime seam CPU-only

- Commit `cc3db43` adds an explicit `--corrective-teacher-profile` seam to
  reference playback. It is disabled by default, restricted to reviewed case
  30, requires `model_based_planner` plus zero-policy mode, rejects a concurrent
  TorchScript policy, and remains exclusive with all dataset, raw-teacher,
  policy-trace, and shadow-trace outputs.
- The same deterministic wrench profile may now be used for the paired
  model-based zero baseline and corrective candidate. The model planner is
  computed first; only then is the bounded corrective action computed from the
  existing causal physical-camera observation and applied through the existing
  model-based residual safety clamp. The flag-absent command path is unchanged.
- Compact policy-rate telemetry records sample count, unbounded/applied/
  normalized maxima, and per-channel amplitude/slew-limit counts. It always
  reports `labels_captured=false`, `dataset_created=false`, and
  `training_started=false`. The gate JSON separately identifies the profile,
  corrective contract, trajectory command source, and capture authorization.
- The committed case-30 profile SHA-256 is
  `26df020549f7db60f3e17d78bb3797e1f32630f615361a1e12d359977e7cf45b`.
  Playback and corrective-teacher module SHA-256 values are respectively
  `a3c389376fa53274fc223bb4bbd846d8230c2c286f8bd9b243d191048359002c`
  and
  `758179ceb0d9e4031ee9e7186169f9d72482decdabab5ad241ccb621d5e3420b`.
- Focused pre-app, profile, action, telemetry, perturbation, playback, and gate
  tests pass `89/89`. The complete authoritative `.98` suite passes `737`
  tests with nine intentional skips in `58.49 s`. No wrapper, authorization
  hash/token, runtime namespace, Isaac/GPU process, label capture, dataset, BC,
  PPO, or training was created.

## Next round after Round 165

- Build a CPU-only paired case-30 preflight contract and wrapper. Pin commit
  `cc3db43`, proposal/profile, exact case-30 plan and perturbation, gains,
  robot/USD, playback/module blobs, all unchanged physical gates, one seed, and
  a fresh baseline/candidate namespace. Leave the authorization hash empty.
- Add negative tests for conflicting cases, profile, plan, perturbation, seed,
  scales, controller arguments, capture paths, dirty/diverged Git state, and
  occupied GPU ownership. The wrapper must stop before Python/Isaac when any
  check fails.
- After a clean authoritative CPU suite, perform a separate go/no-go review.
  Do not issue a token, run Isaac, capture labels, start BC/PPO, or open
  validation/holdout trajectories as part of the preflight implementation.

## Round 166: seal the no-token paired case-30 preflight

- Commit `352a0e1` adds the canonical paired preflight contract, validator,
  wrapper, and focused negative tests. The wrapper contains no playback command
  and rejects `--execute` unconditionally before Python/Isaac with
  `runtime_authorization_not_issued`.
- The contract binds reviewed controller parent `cc3db43`, training case 30,
  fixed configuration/reset seeds `20260716/20260746`, the exact v16 plan,
  deterministic `20 N`/`20`-step perturbation, LQR gains, robot build audit,
  URDF/USD, corrective profile/runtime, perturbation runtime, playback,
  wrapper, and validator. All 13 required identities and tracked Git blobs are
  mandatory.
- It freezes model-based zero-policy mode, residual scales
  `[0.05,0.05,0.02]`, controller and camera-lever-arm arguments, unchanged
  physical thresholds, rollout order `[baseline,candidate]`, same-plan/seed/
  physics/perturbation parity, and the `3 mm` plus `2%` minimum position-p95
  improvement gate. Label capture and dataset creation are false for both
  rollouts.
- Real `.98` preflight passes every check at clean pushed commit
  `352a0e10fab000b01d03eec9084acb08166eed1b`. Contract SHA-256 is
  `dd13e383b15b17426098759cdab886623eee3fc19b24611aaa5c711fbbc4834d`;
  committed contract blob is `1377dd194ae10dcd36f41c67b05b7f6d7ef38552`.
  The fresh namespace is
  `20260722_model_based_corrective_teacher_case30_pair_v1_exclusive`.
- An explicit `.98 --execute` attempt exits `4`, creates no namespace, and
  leaves GPU ownership empty. The complete authoritative suite passes `744`
  tests with nine intentional skips in `63.68 s`. Runtime/GPU authorization,
  token, label capture, dataset, BC, PPO, training, validation, and holdout
  access remain false.
- Review decision: **GO** only to implement a bounded baseline/candidate runner
  and paired finalizer behind a still-empty authorization gate. **NO-GO** to
  launch Isaac or use any output as a teacher until that implementation is
  separately tested, reviewed, authorized once, and dynamically admitted.

## Next round after Round 166

- Add the bounded execution route and paired finalizer without issuing an
  authorization hash. Baseline must run first; candidate may start only after
  baseline dynamic passage and complete GPU release. Each rollout has a
  `600 s` timeout and must preserve its own log, heartbeat, exit code, gate JSON,
  command-source/profile identity, and zero-capture evidence.
- The finalizer must call the pure paired-admission contract, independently
  report dynamic and corrective-target outcomes, require identical plan/seed/
  physics/perturbation identities, and stop with label capture still false.
  Failure or weak improvement must reject the candidate without gate relaxation.
- Keep the runtime token/hash empty during implementation and authoritative CPU
  testing. Issue at most one token only after another explicit go/no-go review;
  do not start BC/PPO or open validation/holdout cases.

## Round 167: paired runner and finalizer are sealed but unauthorized

- Commit `03a833a` adds the bounded case-30 baseline/candidate route and paired
  finalizer while keeping `AUTHORIZATION_SHA256` empty. `--execute` therefore
  still exits before namespace creation, Python/Isaac, or GPU ownership.
- The future execution route consumes a mode-0600 one-use token before Isaac,
  runs the model-based exact-zero baseline first with a `600 s` timeout, waits
  for complete GPU release, and starts the corrective candidate only after the
  baseline gate and dynamic-quality result pass. The candidate uses the same
  plan, seeds, physics, perturbation, gains, controller arguments, and gates.
- Each rollout has an independent heartbeat, gate JSON, log, and exit code.
  Runtime or GPU-release failure skips the candidate and still reaches the
  finalizer. No dataset/raw-teacher/policy-trace/shadow-trace option exists in
  the route.
- The finalizer binds the copied contract and runtime admission, requires both
  command sources and profile identities, exact zero baseline action, bounded
  nonzero candidate action, exact perturbation execution, both heartbeats,
  capture absence, and GPU release. It calls the pure paired-admission contract
  and rejects weak improvement, physical regression, identity mismatch, or any
  capture path while leaving all learning flags false.
- Focused paired-runner/finalizer and existing runtime tests pass `99/99`.
  Real `.98` preflight passes every check at clean pushed commit
  `03a833aa2223124d0e248973a0833b04b1f33847`; resealed contract SHA-256 is
  `426b625b47ad6fd38cf3b56b04a4d65017b87dc4a8eab20623761a7b7a8a2882`.
- Explicit `.98 --execute` again exits `4` with
  `runtime_authorization_not_issued`, creates no namespace, and starts no GPU
  owner. The complete authoritative suite passes `747` tests with nine
  intentional skips in `63.66 s`. Label capture, dataset, BC, PPO, training,
  validation, and holdout access remain false.
- Review decision: **GO** for exactly one separately authorized case-30 pair.
  This is not target admission and not training authorization; the measured
  candidate must still beat the baseline under every frozen paired gate.

## Next round after Round 167

- Add exactly one authorization SHA in a separate commit, create one mode-0600
  token outside the repository, rerun the complete preflight immediately, and
  consume the token before the baseline Isaac process. Preserve exclusive GPU
  ownership throughout the two-rollout sequence.
- Stop after the finalizer regardless of outcome. If baseline fails, candidate
  is skipped. If candidate fails or improvement is below `3 mm` and `2%`, keep
  corrective targets rejected and diagnose CPU-only; do not tune gates in the
  same run.
- Even on a paired pass, keep label capture, BC, and PPO closed. A pass may open
  only a new CPU-reviewed training-split corrective-label capture contract.

## Round 168: the corrective case-30 pair passes its frozen dynamic gate

- Commit `4d47219` issued exactly one external mode-`0600` authorization token;
  commit `cf19c7b` then scoped the token-mode regression test to POSIX while
  retaining the real WSL mode check. The complete authoritative `.98` suite
  passed `747` tests with ten intentional skips and two warnings in `62.74 s`.
- The guarded runner consumed the token before Isaac and created only
  `20260722_model_based_corrective_teacher_case30_pair_v1_exclusive`. The
  copied contract SHA-256 is
  `7406335fa98f3d446ed92585d2811fd10f5d08fed3cab5dad4cf81afa446a391`;
  runtime commit and upstream are both
  `cf19c7b3496ef2090762f1550804f8a225c5e3a1`.
- The exact-zero baseline passed at `11,485` steps with position p95/max
  `0.1419144676/0.1694491894 m`, attitude max `0.2297383535 deg`, pitch max
  `7.0542488300 deg`, riser max error `0.0145935347 m`, and zero action
  saturation. Its gate JSON SHA-256 is
  `63554002a9fa9b10cdc52428cf5e55c47c916132ee6302fdb609a2378a1d43eb`.
- After complete GPU release, the corrective candidate passed at `11,411`
  steps with position p95/max `0.1355908849/0.1597901051 m`, attitude max
  `0.2215529015 deg`, pitch max `7.0453213458 deg`, riser max error
  `0.0118128467 m`, and zero action saturation. Its normalized residual-action
  maxima were `[0.39490636,0.65749963,0.9]`, below the frozen `0.95` admission
  margin. Its gate JSON SHA-256 is
  `a3fe35831f532d6e689f942df616dfdbc7213a440a7bbb367a0fe07bf560a13e`.
- The paired position-p95 improvement is `0.0063235827 m` or `4.455911%`,
  exceeding the frozen `0.003 m` and `2%` minima. Position max, attitude max,
  pitch max, riser error, and saturation all did not regress. Both executions
  used case 30, plan SHA-256
  `1722bfdc7c1aeabc5a9d3920cf6a47bc789afbc96e6ef5c8e540695dc3c97dcb`,
  physics seed `20260746`, source duration `18.144412 s`, execution duration
  `29.2224881939 s`, and the same measured `20 N`/`20`-step wrench pulse.
- Final status SHA-256 is
  `a966c9a5407a0d5ef91ca0553d00a0c47040fdfdc1c4f9bd9814aa33447fc083`.
  The token is absent, GPU/process ownership is released, and no NPZ, dataset,
  raw-teacher capture, policy trace, shadow trace, BC, PPO, or training output
  exists. The final status remains `valid_for_training=false` and label capture
  remains unauthorized.

## Next round after Round 168

- Build a new CPU-only, disabled-by-default corrective-label capture proposal
  for training case 30. Bind the admitted pair and hashes above, the exact plan,
  profile, perturbation, seeds, controller/robot identities, causal observation
  contract, residual scales, unchanged dynamic gates, and an immutable fresh
  namespace. Do not issue a runtime authorization hash or token in that change.
- Define the capture schema before runtime: policy-rate causal observations,
  complete model-planner commands, bounded corrective targets, final applied
  commands, source/execution clocks, initialization separation, amplitude/slew
  flags, perturbation telemetry, and per-sample plan/commit identities. The
  proposal must reject any validation/holdout case or any action outside the
  admitted profile.
- Add negative and synthetic tests for capture-disabled defaults, conflicting
  output modes, noncausal/future observations, identity mismatch, missing
  clocks, action-envelope overflow, initialization leakage, and dirty/diverged
  Git state. A future capture may be reviewed only after the authoritative CPU
  suite passes; BC and PPO remain closed.

## Round 169: seal the disabled corrective-label capture contract CPU-only

- Commit `a673ea0` adds a dedicated case-30 corrective capture schema and a
  disabled runtime seam. It does not reuse the historical raw/shadow teacher
  formats. Each row binds the current physical-`cam_link` pre-action
  observation, complete model-planner command, bounded corrective residual and
  normalized target, final applied high-level command, elapsed/execution/source
  clocks, amplitude/slew flags, perturbation activity, and repeated plan/commit
  identities.
- The archive validator requires exact command reconstruction and rejects
  command clipping, any normalized action at or above `0.95`, non-finite or
  mismatched rows, initialization samples, nonmonotonic or conflated clocks,
  validation/holdout cases, identity drift, noncausal metadata, or any opened
  normalized-dataset/BC/PPO/training flag. A dynamically failed rollout cannot
  save the archive.
- Commits `ab2df7b`, `54426a8`, and `904652e` add the canonical no-token
  preflight, accepted-pair evidence, normalized evidence mode, and updated
  capture-closure regression contract. The wrapper contains no playback/Isaac
  command; `--execute` exits `4` with `runtime_authorization_not_issued` and
  cannot create the namespace.
- Real `.98` preflight passes every check at clean pushed commit
  `904652e25fd8626ec6915fcec31605c53859f43d`. Contract SHA-256 is
  `8729b84d764f1a98e7c36212c80e5b12bd828bcc83907a2da1477deefbea9836`;
  committed contract blob is `1ea3cea2625b1b89ebe450a37795312da99e65d0`.
  It binds accepted-pair final-status SHA-256
  `a966c9a5407a0d5ef91ca0553d00a0c47040fdfdc1c4f9bd9814aa33447fc083`
  and case-30 plan SHA-256
  `1722bfdc7c1aeabc5a9d3920cf6a47bc789afbc96e6ef5c8e540695dc3c97dcb`.
- The focused local capture/playback suite passed `90/90`; after repairing one
  obsolete static assertion, the complete authoritative `.98` suite passed
  `766` tests with ten intentional skips and two warnings in `71.90 s`.
  The future namespace is absent, GPU/process ownership is empty, and runtime,
  label capture, normalized dataset creation, BC, PPO, training, and
  valid-for-training remain false.

## Next round after Round 169

- Add a separately reviewed bounded execution wrapper and capture finalizer
  while keeping the authorization hash empty. The eventual route must consume
  a one-use mode-`0600` token before Isaac, run only case 30 at the frozen
  commit/plan/profile/seed/perturbation, apply the admitted corrective teacher,
  and save only after unchanged dynamic, thermal, controller, perturbation, and
  archive-schema gates pass.
- The finalizer must independently hash and reopen the NPZ, require exact
  source/execution clocks and per-row identities, prove exactly 20 perturbation
  rows, reject initialization leakage or command clipping, preserve the gate
  JSON/heartbeat/log/exit code, and leave the captured archive
  `valid_for_training=false`. It may authorize later dataset conversion only;
  it must not start BC or PPO.
- Run focused negative tests and the complete authoritative CPU suite, then
  perform another explicit go/no-go review. Do not issue a token, create the
  namespace, or launch Isaac in the implementation commit.

## Round 170: guarded capture runner and finalizer pass CPU review

- Commit `37ee83a` adds the latent one-shot case-30 runner and independent
  capture finalizer while leaving `AUTHORIZATION_SHA256` empty. The wrapper
  contains the exact admitted case/plan/profile/perturbation/controller route,
  a `600 s` timeout, exclusive GPU checks, one-use token consumption before
  Isaac, heartbeat/log/exit-code preservation, and final GPU-release checking.
- The finalizer reopens the NPZ through the canonical archive validator. It
  requires dynamic, thermal, controller, and perturbation passage; exact
  admission/contract/plan/profile/pair/runtime identities; complete source and
  execution clocks; sample/heartbeat/telemetry count equality; exactly `20`
  perturbation-active rows; action margin below `0.95`; initialization
  exclusion; and all legacy dataset/BC/PPO/training paths closed. Even a pass
  authorizes only later dataset conversion and remains
  `valid_for_training=false`.
- Real `.98` preflight passes every check at clean pushed commit
  `37ee83a693176a22333cd0543c1020d6380d5545`. Contract SHA-256 is
  `c04614ed965900b318e416a52c383513a5b81b995dd2465977e5eb8631fa4806`.
  Explicit `--execute` exits `4` with `runtime_authorization_not_issued`, the
  namespace remains absent, and no Isaac/GPU process starts.
- Focused archive/runner/finalizer tests pass `86/86`; the complete
  authoritative `.98` suite passes `771` tests with ten intentional skips and
  two warnings in `72.79 s`. A further POSIX regression proves that only the
  exact mode-`0600` token can turn runtime/capture admission true.
- Diff review from capture-seam parent `a673ea0` finds no change to playback,
  controller, plan generation, robot/USD, LQR gains, corrective profile,
  residual scales, or physical gates. Review decision: **GO** for exactly one
  separately authorized case-30 corrective capture; **NO-GO** for normalized
  dataset creation, BC, PPO, validation/holdout cases, or any second capture.

## Next round after Round 170

- In a separate commit, issue exactly one token SHA and update only the wrapper
  identity plus runtime/GPU/capture authorization fields. Rerun the focused and
  complete CPU suites, canonical preflight, fresh-namespace check, and exclusive
  GPU check before creating the external mode-`0600` token.
- Run the one bounded capture and stop after the finalizer regardless of
  outcome. Independently hash/reopen the gate, heartbeat, admission, contract,
  capture, final status, and logs; verify token consumption and GPU release.
- If the archive is admitted, open only a CPU-reviewed conversion proposal. If
  it fails, diagnose CPU-only without changing gates or issuing a second token.
  BC and PPO remain closed in either case.

## Round 171: the one-shot case-30 capture is rejected at the v1 archive boundary

- Commit `a898eb4` issued exactly one separately reviewed mode-`0600` token for
  namespace
  `20260722_model_based_corrective_teacher_case30_capture_v1_exclusive`. The
  authoritative `.98` suite passed `771` tests with eleven intentional skips
  and two warnings in `72.52 s` before launch. The wrapper consumed the token
  before Isaac and did not leave a reusable authorization file.
- The physical execution reached the same complete corrective trajectory at
  `11,411` steps, source duration `18.144412 s`, execution duration
  `29.2224881939 s`, position peak `0.1597901051 m`, attitude peak
  `0.2215529015 deg`, and pitch peak `7.0453213458 deg`. The GPU/process owner
  was released after the bounded run.
- Archive creation then failed closed with
  `ValueError: final command was clipped or does not reconstruct`. The v1
  schema required the final safety-supervised command to equal the complete
  model command plus the requested corrective residual. At least one requested
  channel was clipped by the unchanged runtime supervisor, so this equality is
  not a valid training-target contract. No NPZ was written.
- The rejected gate JSON SHA-256 is
  `9efeec937f9e4445b1e9c7d3004aa781205a1493ad8332d095574f42a795efd1`;
  heartbeat SHA-256 is
  `159e8513bdd6f0cc2ef58c2da24b5441e04021a798cff9d46577cfbbf25e6a45`;
  final-status SHA-256 is
  `7a68218bc7851801f936d25ecdc8b9bc9148328a92ef2a11ac6726c915a635a4`;
  playback-log SHA-256 is
  `801d86f47ffde3fe503872cbfff36a9b86e192243364abc7b4fe800c18e67816`.
  The final status preserves contract SHA-256
  `143be187f83364d3475f638251aa9201e002ad88af6ea64f4c9077df51195c3a`
  and admission SHA-256
  `9f5010c36a34c828169afaa47560f2bb298a0b39fb9134ab05054c2686bdc138`.
- Two evidence defects were also exposed: the failure JSON reports
  `deterministic_teacher` instead of the active
  `model_based_planner_plus_corrective_teacher` source, and the heartbeat
  reports capture outputs disabled because its check omits the dedicated
  corrective-capture directory. Both must fail closed in the replacement
  contract.
- The rejected evidence is preserved under
  `evidence_20260722_case30_corrective_capture_v1_rejected`. It remains
  `capture_admitted_for_dataset_conversion=false`,
  `valid_for_training=false`, with normalized-dataset creation, BC, PPO, and
  training all false.

## Next round after Round 171

- Replace the v1 archive contract CPU-only. Preserve requested teacher intent
  and separately record the effective residual equal to final supervised
  command minus complete model command. The effective normalized residual is
  the only candidate BC target because it exactly reconstructs behavior that
  the bounded runtime can apply.
- Record per-channel clipping and requested-versus-effective deltas without
  relaxing command, safety, dynamic, thermal, or reserved-action limits. Fix
  corrective-source and heartbeat evidence, and require both in the finalizer.
- Move to a fresh v2 namespace, revoke the v1 token/hash, run focused and full
  authoritative CPU tests, and stop for a new explicit go/no-go. Do not issue a
  token or launch Isaac; dataset conversion, BC, PPO, validation, and holdout
  access remain closed.

## Round 172: v2 supervisor-aware capture contract passes CPU review

- Commit `bd05014` replaces the rejected archive format with
  `cinebotrl_two_wheel_riser_corrective_teacher_capture_v2`. Every policy-rate
  row now preserves requested corrective intent and the effective residual
  equal to final safety-supervised command minus the complete model-planner
  command. The effective normalized residual is explicitly the only candidate
  future BC target.
- The archive separately seals requested/effective residuals and normalized
  actions, requested-versus-effective delta, per-channel clipping mask, final
  command, causal observation, source/execution clocks, initialization
  exclusion, perturbation state, and repeated plan/runtime identities. Both
  requested and effective labels must remain below the unchanged exclusive
  `0.95` margin; final command must reconstruct from model command plus the
  effective residual.
- The runtime heartbeat now treats the dedicated corrective directory as an
  enabled capture path, and runtime-failure evidence reports
  `model_based_planner_plus_corrective_teacher` when that source is active. The
  finalizer rejects a hidden capture heartbeat, missing supervisor contract,
  inconsistent clip telemetry, or either action-envelope overflow. Legitimate
  supervisor clipping no longer changes commands or invalidates an otherwise
  exact effective target.
- The fresh closed namespace is
  `20260722_model_based_corrective_teacher_case30_capture_v2_exclusive`.
  Contract SHA-256 is
  `9d2df8b7659979a323737f88f322d326ab63edb21ce36e82dede587a5fa2d014`
  and committed blob is `868da616e9c921608befc19f33c80b376838f156`.
  Runtime commit is
  `bd05014a5af9c866b881fda1943af92da846eddb`; the contract remains bound to
  reviewed parent `f54db86768464c2d83feda9b2ec48c4ea2e732bf`.
- Sealed implementation SHA-256 identities are playback
  `83b585f46526f7dd535416ef513f606c978ac82c575fc8d021cff72dd863f47e`,
  capture runtime
  `b74e29b236d5195d9b877279a329c532beadc9bef6ede18c5a26f0d4d3d23c3f`,
  validator
  `12405025a4f759825835e5d60a0474d12f31a34daf74825380a24a81b88cb39b`,
  wrapper
  `280cee3da4d5152047fec6bd325b0ae9f23fbbd824d1ed571c19bf6b78e27eb6`,
  and finalizer
  `d19e68fc812e53b79491ae0c0ffc572c5663258051ae0dabb8055184ff454d24`.
- Focused local tests pass `51/51`. The Mac full suite cannot collect six
  unrelated mobile-manipulator modules because that host lacks `gymnasium`.
  On authoritative `.98`, focused tests pass `50` with one intentional skip;
  the complete pinned suite passes `775` with eleven intentional skips and two
  config warnings in `68.56 s`.
- Canonical `.98` preflight passes every check. Explicit `--execute` exits `4`
  with `runtime_authorization_not_issued`; the v2 namespace is absent and WSL
  and NVIDIA compute ownership are empty. The old v1 authorization SHA is
  absent, no replacement token exists, and capture, dataset conversion, BC,
  PPO, training, validation, and holdout access remain closed.
- Diff audit finds no plan, planner-command, LQR/controller, robot/USD, safety
  limit, perturbation, or dynamic/thermal gate change. Review decision:
  **GO** for exactly one separately authorized v2 case-30 corrective capture;
  **NO-GO** for dataset conversion, BC, PPO, a second capture, or any other
  case.

## Next round after Round 172

- Only after explicit authorization, add one fresh v2 token SHA in a separate
  commit, rerun canonical preflight and the complete CPU suite, create one
  external mode-`0600` token, and execute the sole case-30 capture. Stop after
  the independent finalizer regardless of outcome.
- If the archive passes, independently hash/reopen it and then open a separate
  CPU-only conversion proposal whose labels come only from
  `effective_corrective_normalized_actions`. Do not train directly from
  requested labels.
- Do not start BC or PPO until conversion is reviewed and the training-split
  dataset contract is admitted. Keep holdout cases `[3,5,13,19,24]` unopened.

## Round 173: effective-label case conversion is sealed CPU-only

- Commit `cbecb6a` adds a distinct
  `model_based_corrective_case_dataset_v1` path instead of reusing the legacy
  merged residual dataset. This closes a semantic mismatch: historical
  phase-feedforward residual data declares scales `[0.30,0.40,0.10]`, while
  the admitted complete-model-planner residual contract uses
  `[0.05,0.05,0.02]`. These schemas and labels may not be silently mixed.
- Conversion requires a passed v2 capture final status, all dynamic/archive
  checks true, an exact source-capture SHA-256 match, matching runtime commit,
  and all dataset/BC/PPO/training flags closed. It refuses an existing output
  and does not write anything unless `--execute` is explicitly supplied.
- Only `effective_corrective_normalized_actions` become candidate labels.
  Requested pre-supervisor actions remain audit-only. Effective residuals must
  reconstruct final command from the complete model command; requested and
  effective residuals, their delta, and the per-channel clipping mask must all
  remain mutually consistent and below the unchanged reserved margin.
- The converter repairs a causal-history mismatch exposed by the v1 run:
  capture observations contain the previously requested policy action, but the
  intended target is the effective post-supervisor action. The converted case
  dataset therefore resets the first previous-action observation to zero and
  rebuilds every later row from the previous effective label. Any recurrence
  mismatch fails closed.
- A converted case is only `valid_for_case_merge=true`; it remains
  `merged_dataset_created=false`, `valid_for_training=false`, with BC, PPO, and
  training unauthorized. The existing BC trainer does not accept this case
  schema directly.
- Runtime commit is
  `cbecb6a2d5f706dddf056b204907c166344652b2`. Converter-runtime SHA-256 is
  `a6b10fb888e9da46bd919c3d421a495f9cf7b2ec98bea84af223968711c82aef`;
  CLI SHA-256 is
  `0dded23cc3b6c14ff43f7aaad747bbd6d165d0a8de70aa9259ad06fc30da42bc`;
  focused-test SHA-256 is
  `9fb959851c20588e5c273ef5741571d0cb86439ff8453693d9f228167311657d`.
- Expanded local capture/conversion/trainer tests pass `77/77`. The complete
  authoritative `.98` suite passes `782` with eleven intentional skips and two
  config warnings in `73.50 s`. Canonical capture preflight still passes at
  clean pushed HEAD with runtime and label capture unauthorized; the v2
  namespace remains absent and no conversion or training artifact was created.

## Next round after Round 173

- Preserve the same explicit boundary: authorize and run exactly one v2
  case-30 capture, then independently seal it. Do not execute conversion unless
  the real final status admits it.
- If admitted, run converter preflight, verify all source/output hashes, execute
  one case conversion, and stop for review. One case is not a train/validation
  curriculum and must not be passed directly to BC.
- Design the multi-case merge/split contract only after real converted cases
  exist. Preserve case-disjoint splits, keep holdout unopened until its gate,
  and bind the model-based action scales and command contract end to end.

## Round 174: BC artifacts and legacy rollout gates bind action semantics

- Commit `c296392` closes an independent DNN deployment ambiguity. Previous BC
  checkpoints and reports identified architecture and observations but omitted
  the command base and action scales. A normalized policy could therefore be
  paired accidentally with the wrong physical residual semantics.
- The trainer now records `dataset_schema`, `policy_command_base`,
  `policy_residual_contract`, and `residual_action_scales` in both checkpoint
  and report. Current supported legacy merged datasets are explicitly bound to
  `phase_feedforward`,
  `phase_feedforward_plus_bounded_policy_residual_v1`, and either the frozen
  v2 `[0.30,0.40,0.10]` scales or the exact admitted v3 metadata scales.
  Invalid or non-positive scales fail closed.
- The legacy exact-source BC, holdout, and all-79 wrappers now require those
  report fields. Holdout/all-79 playback explicitly passes
  `--policy-command-base phase_feedforward` and
  `--residual-action-scales 0.30,0.40,0.10`; resumable gates must report the
  same values. These wrappers can no longer silently evaluate the new
  model-planner residual policy.
- The future model-based path remains separate and must eventually require
  `model_based_planner`,
  `model_based_planner_plus_bounded_policy_residual_v1`, and
  `[0.05,0.05,0.02]`. No model-based merged dataset or policy is claimed by
  this change.
- Runtime commit is
  `c296392386c7e19e8264b61fd27b852c8074b805`. Trainer SHA-256 is
  `8a4f0e582deb3e6b7388d9877944133f482a73acb0e8d6c2606704183f85673b`;
  legacy BC/holdout/all-79 wrapper SHA-256 values are respectively
  `e80a4954a4c699afd74d6750b671d674cb21b3d3c767eb2925f0ab83fed4f70e`,
  `5731d22d52dc6d5f6da6fa6c76d325611115c5ac08899a5269a287a5b75b5d88`,
  and `0daf68098211494125147e1a0a020bfa665a2403ca630be9bf26f803315115b6`.
- Focused semantics and wrapper tests pass `67/67`. The complete authoritative
  `.98` suite passes `783` with eleven intentional skips and two config
  warnings in `71.07 s`. The corrective-capture preflight still passes at
  clean pushed HEAD with runtime and label capture unauthorized; no Isaac,
  conversion, BC, PPO, or learned rollout ran.

## Next round after Round 174

- Do not add another policy abstraction in place of evidence. The next useful
  step is the already reviewed, exactly one-use v2 case-30 corrective capture.
- After a real archive passes, execute the sealed effective-label conversion
  once and audit its clipping distribution and rebuilt recurrence. Only then
  decide whether the generic corrective profile can be admitted on additional
  train cases and whether a model-based multi-case merge contract is justified.

## Round 175: one authorized case-30 corrective capture v2 passes

- The user authorized exactly one v2 case-30 corrective capture. Standalone
  authorization commit `ca755bdcf4498fe39f19735ed666e96ca11bed96`
  pinned token SHA-256
  `26c519b95a729f71b68b3dc4a0fed4f9cc90c0f73944e754476e9e1a3b345e72`.
  Before token creation, canonical `.98` admission passed, a missing-token
  execution failed closed with exit `4`, and the complete authoritative CPU
  suite passed `783` with eleven skips and two config warnings in `71.46 s`.
- One external mode-`0600` token was created and consumed before Isaac. The
  fixed namespace is
  `20260722_model_based_corrective_teacher_case30_capture_v2_exclusive`.
  Exactly one case ran; playback and finalizer both exited `0`, GPU ownership
  was released, and no second capture, conversion, BC, PPO, holdout, or
  training process started.
- The finalizer passed every admission, dynamic, thermal, controller,
  perturbation, source, heartbeat, archive, and GPU-release check. Dynamic
  quality passed with peak position error `0.159790 m`, peak pitch
  `7.045321 deg`, zero action/riser/proxy saturation, and no termination.
- The v2 archive contains `11,411` aligned samples with `65` observation
  features. Both clocks are monotonic and end at source `18.144412 s` and
  execution `29.22248819392579 s`; initialization rows are excluded and
  exactly `20` perturbation-active rows are present.
- Requested normalized action maxima are
  `[0.394906,0.657500,0.900000]`; effective post-supervisor maxima are
  `[0.394906,0.657500,0.267339]`. Command clipping is explicit on
  `[200,308,333]` rows by channel, with requested/effective residual deltas,
  amplitude limiting, and slew limiting preserved. Only effective actions are
  eligible for the later converter.
- The admitted capture SHA-256 is
  `ec0f13030ce755c38e31c138507537f461126312b0c268832bc6bf9a40e4e8cb`;
  final-status SHA-256 is
  `e0b9ec3186e677c34289a85e72e4bc91e3cd3d8ce5cfdea16d74e1c0be0554b2`.
  Immutable evidence is copied to
  `evidence_20260722_case30_corrective_capture_v2/` with a hash manifest.
- Authorization is revoked immediately after evidence preservation. The
  capture is admitted only for a separately reviewed CPU conversion; it is
  not a normalized training dataset and remains `valid_for_training=false`.

## Next round after Round 175

- Run the sealed effective-label converter in preflight against the preserved
  archive and final status. If it passes, authorize exactly one CPU-only case
  conversion, independently audit recurrence reconstruction and clipping, and
  stop for review.
- Do not train from this one case. A later multi-case capture/merge proposal
  must establish case-disjoint train/validation coverage while leaving holdout
  cases `[3,5,13,19,24]` closed.
- Keep BC and PPO disabled until the merged model-based dataset and action
  semantics are explicitly admitted.

## Round 176: case-30 effective-label conversion passes CPU-only

- Local and authoritative `.98` preflights both reopened the sealed capture
  and final status, reported `11,411` rows, and bound source SHA-256 values
  `ec0f13030ce755c38e31c138507537f461126312b0c268832bc6bf9a40e4e8cb`
  and
  `e0b9ec3186e677c34289a85e72e4bc91e3cd3d8ce5cfdea16d74e1c0be0554b2`.
  Focused authoritative capture/converter tests passed `24/24` with two
  config warnings.
- Exactly one CPU conversion wrote
  `case_0030_model_based_corrective_case_dataset_v1.npz` in namespace
  `20260723_model_based_corrective_case30_conversion_v1_cpu`. No Isaac, GPU,
  capture, merge, holdout, BC, PPO, or training process ran.
- Candidate labels are byte-for-byte equal to
  `effective_corrective_normalized_actions`; requested actions are preserved
  only as audit data. Effective action maxima remain
  `[0.394906,0.657500,0.267339]`, below the unchanged reserved margin.
- The converter rebuilt the three previous-action observation channels from
  the preceding effective label. Independent audit found exact recurrence,
  `841` changed history rows, maximum history correction `0.900000`, and exact
  preservation of all other observation channels, clocks, case IDs, and the
  clipping mask `[200,308,333]`.
- Final-command and requested/effective-delta reconstruction errors are bounded
  by `1.1921e-07` and `3.0994e-09`. Output SHA-256 is
  `191a44147bc44038a0645bf48a63609463bf280d97b37ddaf884200bd8b52447`.
- The artifact is `valid_for_case_merge=true` only. It remains
  `merged_dataset_created=false`, `valid_for_training=false`, with BC, PPO,
  and training unauthorized.

## Next round after Round 176

- Do not train one case. Design and test a fail-closed multi-case capture and
  merge contract using the same model-based command semantics, effective-only
  labels, `[0.05,0.05,0.02]` scales, and case-disjoint splits.
- Before another GPU capture, perform CPU-only eligibility analysis over the
  accepted train cases to select the smallest diverse corrective-teacher
  tranche. Keep holdout cases `[3,5,13,19,24]` unopened.
- Require every additional capture to pass its own dynamic gate and v2 archive
  audit before conversion; do not infer admission from case 30.

## Round 177: diverse corrective pair tranche selected CPU-only

- A new fail-closed selector binds the v16 plan portfolio, the sealed dynamic
  selection, current admitted split, every selected plan hash, and the passed
  case-30 conversion audit. It uses twelve normalized duration, path,
  kinematic, camera-height, and historical dynamic-residual features with
  deterministic farthest-point selection anchored on case 30.
- The first authoritative attempt stopped before output because it required
  the enclosing batch summary to pass for case 6. Audit showed case 6's own
  row, gate, and result dynamic checks all pass; its batch summary was false
  only because a later case rejected. Commit `d7fe4a9` correctly uses sealed
  per-case dynamic evidence and adds regression coverage for this condition.
- Focused local selector/converter/teacher tests pass `29/29`; authoritative
  selector tests pass `7/7`. Recomputing the final selection produced
  byte-identical JSON and independent flag/split checks passed.
- The eligible pool contains 31 current training cases. The selected five are
  `[30,23,6,2,7]`: case 30 is the converted pilot anchor; cases 23, 6, 2, and
  7 are marked `same_seed_paired_canary_required`.
- Validation cases remain `[8,16,22,32,78]`; holdout remains
  `[3,5,13,19,24]`. The selector explicitly sets case-30 profile reuse,
  generic-profile creation, runtime, GPU, capture, merge, BC, PPO, training,
  and training validity to false.
- Selection SHA-256 is
  `93aa25e99409ad926d4c4cf0b15075e6ea3532dd1c01d5217a45daffc37db7c0`.
  No Isaac or GPU process ran.

## Next round after Round 177

- Build a CPU-only per-case paired-canary contract for the first new candidate,
  case 23. It must use case 23's pinned plan and a case-specific profile
  proposal; do not reuse the case-30 profile identity.
- Keep the baseline-first, same-seed, same-plan, same-clock admission rule and
  require measurable position-p95 improvement without regressions in position
  max, attitude, pitch, riser error, or saturation.
- Do not authorize the case-23 GPU pair until that contract, negative tests,
  identities, and clean pushed state are independently reviewed.

## Round 178: case-23 paired-canary proposal is sealed CPU-only

- Case 23 is a useful diversity canary because it combines low plan error and
  modest base/riser rates with a `0.558596 m` camera-height span. Its baseline
  may already be too accurate for a corrective teacher to clear the measurable
  improvement gate; such a paired reject is an acceptable result.
- The corrective-profile loader now supports an explicitly supplied positive
  expected case while preserving case 30 as the default. Playback accepts a
  profile only when it matches the sole requested case. The new case-23
  profile is therefore a distinct identity, not reuse of the case-30 file.
- The CPU proposal binds tranche selection SHA-256
  `93aa25e99409ad926d4c4cf0b15075e6ea3532dd1c01d5217a45daffc37db7c0`,
  portfolio SHA-256
  `8351514a361d3be4e5fbf57f2dbb019a7d8d2f5b86e89cea2553a1cfda3c64a1`,
  case-23 plan SHA-256
  `ad76ada4cdb9f874da615aa0c6e441be62d9a768b813c597c5dc4e20894042b6`,
  and profile SHA-256
  `808e2e295bb4639288efc84f5bdf7a1954a6e2aa96f6854d808b02306e9324e3`.
- It proposes the unchanged conservative gains/limits and a case-specific
  20-step, 20 N pulse at execution midpoint `4.964847 s`. Baseline must run
  first; candidate requires baseline dynamic pass; same plan, seeds, physics,
  clocks, and perturbation are mandatory.
- The measurable-improvement and no-regression gates remain unchanged:
  position p95 at least `0.003 m` and `2%` better, position max allowance
  `0.005 m`, attitude `0.10 deg`, pitch `0.50 deg`, riser error `0.002 m`,
  and no saturation regression.
- Focused local and authoritative proposal/profile/runtime tests pass `36/36`.
  Proposal SHA-256 is
  `ef520558f4240add67667e1cbd3146c987d7b67aafaf6ad781b2dc9c576a2387`.
  Runtime route, token issuance, GPU, capture, merge, BC, PPO, and training all
  remain false.

## Next round after Round 178

- Independently review the case-23 proposal, especially whether the midpoint
  pulse is sufficient to create measurable but safe corrective demand on this
  already accurate trajectory.
- Only after review, implement a fresh case-23 guarded pair wrapper, finalizer,
  canonical validator, identities, and no-token preflight. Keep execute mode
  fail-closed until a separate explicit authorization.
- Do not infer capture eligibility from a passed pair implementation. Label
  capture would require an actual same-seed candidate improvement and a later
  separate one-use capture contract.

## Round 179: case-23 guarded pair route passes CPU-only

- A fresh case-23 wrapper, canonical validator, pair finalizer, tracked wrench
  profile, and sealed contract are implemented at commit
  `1a626d37ebcf0b8e4fafc9ff814aaa2512c1a354`. No case-30 evidence or runtime
  route was modified.
- The contract pins the reviewed parent, proposal, tranche selection, case-23
  plan, case-specific corrective and perturbation profiles, robot assets,
  controller/runtime code, wrapper, validator, and finalizer. Contract
  SHA-256 is
  `94e87b8eaae122f0c236c3812d9f2c184b54e1dd609713d52a9eb313ab59dd67`
  and Git blob is `33115952da2c70cadec2275b8561f4915d55b8ac`.
- The canonical `.98` no-token preflight passed every identity, proposal,
  selection, controller-argument, unchanged-gate, split, clean-HEAD, and fresh
  namespace check. It reports `cpu_contract_ready=true` while runtime, GPU,
  capture, dataset, BC, PPO, training, and training validity remain false.
- The authoritative `.98` suite passes `808 passed, 11 skipped, 2 warnings`
  in `78.25 s`. Focused new-route tests pass `11/11`. The sparse Mac riser
  subset passes `580` tests; its 31 failures are all missing local robot asset
  files and are covered by the authoritative `.98` run.
- A direct `--execute` attempt stops before Python/Isaac with exit code `4`
  and `runtime_authorization_not_issued`. The authorization hash is empty and
  the target namespace remains absent. No Isaac, capture, dataset, BC, PPO, or
  training process ran.

## Next round after Round 179

- Stop at the CPU review boundary. Do not issue a token or launch the pair
  unless the user separately authorizes exactly one case-23 paired canary.
- If authorized, create a later one-use token/contract revision rather than
  modifying this no-token evidence. Run baseline first; admit the candidate
  only if baseline dynamically passes; stop after the candidate and finalizer.
- A passed pair would establish only that the case-specific correction offers
  safe measurable improvement. Corrective label capture, conversion, corpus
  merge, BC, PPO, and training each remain separate later decisions.

## Round 180: case-23 route repaired and quantitatively admitted CPU-only

- Review found a real route defect after Round 179: the deterministic wrench
  loader still hardcoded case 30, so the tracked case-23 wrench profile would
  have failed before simulation after any future authorization. The loader now
  keeps case 30 as its default but accepts another case only through an
  explicit positive `expected_case`; playback supplies the sole requested
  case. Multi-case and mismatched profiles remain fail-closed.
- A new deterministic readiness audit binds the exact case-23 proposal,
  tranche selection, plan SHA, historical dynamic gate SHA, provisional plant
  prior, case-30 paired precedent, corrective profile, and wrench profile.
  Audit SHA-256 is
  `335e4cd0181edce3c1bb0dd7c0b13c99df040f648b788d65b0fce1a34bf3a198`.
- The pulse is `20 N` for `0.10 s`, exactly `2.0 N*s`, matching the edge of
  the provisional accepted signed impulse envelope. Its free-body displacement
  estimate is `0.003571 m`; this is used only as an observability screen, not
  as a closed-loop prediction.
- At the exact midpoint, local limit fractions are base `0.5919`, yaw
  `0.0120`, riser `0.0772`, and proxy `0.1217`. Historical case-23 margins to
  unchanged gates remain `0.04455 m` p95 position, `0.14436 m` max position,
  `6.255 deg` pitch, `0.01902 m` riser error, and `0.19882` saturation ratio.
- The same pulse previously produced a safe case-30 paired p95 improvement of
  `0.006324 m` (`4.456%`). This is precedent for one measurement, not proof
  that case 23 will improve.
- Implementation commit `f74d9223c626f47da9ad97e2f891417572076bdf`
  passes the authoritative `.98` suite with `818 passed, 11 skipped, 2
  warnings` in `77.62 s`. The canonical no-token preflight passes all
  readiness and identity checks.
- The repaired contract SHA-256 is
  `510d34d2e50d2708b654dbb348e4797a88132014b78a2d99023d807c7dd8e949`
  with Git blob `cbf9ff0e1635b2dc62b98d8760005dc8acd8f469`. The Round 179 contract hash is
  retained as historical evidence and is superseded for future case-23 use.
- Decision is `recommend_exactly_one_bounded_case23_pair_canary`. Runtime,
  GPU, capture, dataset, BC, PPO, training, and training validity remain false;
  no namespace or Isaac process was created.

## Next round after Round 180

- The next useful runtime action is exactly one authorized case-23 paired
  canary under the repaired contract: baseline first, candidate only after a
  baseline dynamic pass, unchanged thresholds, no capture, then stop.
- Do not infer authorization from this CPU recommendation. A later explicit
  one-use token revision and user authorization are still required.
- If the pair passes measurable-improvement and no-regression gates, separately
  review one corrective capture. If it rejects, preserve the pair and diagnose
  case 23 rather than changing thresholds or escalating the pulse.

## Round 181: riser engineering-sample procurement boundary sealed CPU-only

- Current official vendor specifications are frozen in
  `RISER_VENDOR_SPEC_SNAPSHOT_20260723.json`; the snapshot covers the Leadshine
  400 W brake motor, ELD2 CANopen drive, igus 70 mm/rev belt-axis catalog data,
  and the generic SITEMA fail-safe catcher principle. Catalog data is not
  treated as vertical mobile-robot approval.
- A deterministic procurement audit recomputes `292.397 N` rated linear force,
  `398.982 W` mechanical rated power, `2571.43 rpm` at `1 m/s`, and the existing
  `1.5 m` mechanical-stroke recommendation.
- The 8 kg emergency case needs `1.20295 N m` versus the motor's `1.27 N m`, so
  the nominal margin is only `1.05574`. A new fail-closed production threshold
  of `1.15` permits only about `7.2336 kg` under the same assumptions.
- Decision: one 400 W + ELD2 unit is recommended only for an instrumented,
  non-riding engineering bench sample. Production defaults to a 750 W-class or
  otherwise resized solution unless measurements prove at least 15% emergency
  margin and close regeneration, thermal, vertical-duty, gearbox, and anti-fall
  requirements.
- The report separately requires an external regeneration path, independent
  anti-fall device, hard limits, absorbing end stops, and safety-rated power
  removal. Motor phase current must not be used directly to size the battery
  branch.
- Evidence SHA-256 is
  `0733dac3f19ccbb6aa74986efa09f70f609c7bb008f5050feff12dea0bdbcf0b`.
  Production procurement, hardware transfer, training, and GPU work remain
  false.
- The host-independent evidence writer was repaired to force LF JSON and POSIX
  repository-relative paths. A fresh authoritative `.98` generation is
  byte-identical to the committed summary at the same SHA-256.
- The exact implementation checkpoint through
  `545ffe967884d481ab6e8cfd66d7b07e81971539` passes the authoritative `.98`
  CPU-only repository suite: `825 passed, 11 skipped, 2 warnings` in `77.93 s`.
  No Isaac process, runtime namespace, authorization, capture, dataset, BC,
  PPO, or training was created.

## Next round after Round 181

- Do not convert the engineering-sample recommendation into bulk procurement.
  The next hardware step is a measured bench campaign: moving mass, carriage
  force, current, temperature, regeneration voltage/energy, stopping distance,
  brake behavior, and anti-fall proof.
- Runtime case-23 authorization remains a separate decision. Do not launch
  Isaac, capture, BC, PPO, or training as part of this hardware audit.

## Round 182: measured riser bench acceptance contract implemented CPU-only

- A new machine-readable measurement template covers complete moving mass,
  friction, counterbalance, physical stroke, exact camera-height software
  limits, continuous duty, phase/DC current, thermal behavior, regeneration,
  emergency stopping, service-brake hold, independent anti-fall, hard limits,
  end stops, and safety-rated power removal.
- A result cannot pass from typed summary values alone. It must bind raw logs,
  force/current/temperature/position calibration records, supplier approval
  documents, and safety-test video with seven SHA-256 identities.
- The deterministic auditor preserves the current 400 W motor, ELD2 drive,
  `3:1`, 70 mm/rev, 1.0 m/s, 1.50 m mechanical stroke, and 0.60--1.80 m camera
  contracts. It recomputes emergency force margin from measured mass,
  friction, and counterbalance and requires at least `1.15`.
- Project engineering gates require a 30-minute/60%-duty run, at least
  `0.95 m/s` achieved speed, phase-current and `65 V` DC-bus limits, bounded
  temperature rise and terminal thermal slope, ten full-speed stops within
  `0.12 m`, a 600-second brake hold, and ten independent anti-fall catches
  within `0.03 m`.
- Even a complete pass only sets
  `ready_for_production_design_review=true`. Automatic production procurement,
  hardware transfer, runtime, GPU, dataset, BC, PPO, and training are never
  authorized by this audit.
- The canonical unmeasured template fails closed with 34 missing numeric/hash
  fields. Summary SHA-256 is
  `7e9c48539e9d9ae77fad9143a9bb91936ed195f92a60b517f22c469c74af37df`;
  decision is `collect_complete_calibrated_bench_measurements`.
- Exact implementation commit
  `79dc596b8202cd43e2fb5310696865633c4a2fc1` reproduces that summary
  byte-for-byte under authoritative `.98` Windows Python. The full `.98`
  CPU-only suite passes `837 passed, 11 skipped, 2 warnings` in `80.88 s`.
  No Isaac, playback, capture, dataset, BC, PPO, or training process ran.

## Next round after Round 182

- When hardware exists, copy the template to a dated evidence namespace,
  replace every placeholder from calibrated raw evidence, run the auditor, and
  preserve failing results rather than editing thresholds.
- On the learning path, the next runtime action remains exactly one separately
  authorized case-23 paired canary. Do not skip it to capture cases 6, 2, or 7,
  and do not open BC/PPO from the single converted case-30 teacher.

## Round 183: active goal status reconciled with current learning evidence

- The machine-readable goal no longer reports the July-17 state of zero
  dynamically qualified cases and an unmeasured residual envelope. It now
  distinguishes 79 corrected references, 42 dynamically qualified candidates,
  one converted model-based corrective case, and zero trainable model-based
  corrective corpus cases.
- The old 41-case planner-imitation BC is explicitly classified as encoder
  initialization only. It completed case 78 but failed position p95 at
  `0.165018 m` versus the unchanged `0.15 m` gate, and its command layering is
  not the target final policy contract.
- The active learned contract is
  `model_based_planner_plus_bounded_policy_residual_v1` with actions
  `[delta_vx,delta_wz,delta_riser_target]` and scales `[0.05,0.05,0.02]`.
  Exact-zero TorchScript preservation passed cases 8 and 78 before any nonzero
  corrective capture was admitted.
- Case 30 remains the sole converted effective-label teacher: `11,411` rows,
  case-merge eligible but not training eligible. The diverse paired tranche is
  `[30,23,6,2,7]`; case 23 is next and is CPU-ready but runtime unauthorized.
- Hardware status is included in the same current checkpoint: one 400 W
  instrumented bench sample is allowed, production remains 750 W-class/resized
  pending measurements, and the unmeasured bench template still lacks 34
  fields.
- Goal JSON SHA-256 is
  `4c8473fc3688231f018637526daf237056bb213531481aa684a015a7fb25717c`.
  New regression tests prevent candidate/corpus, planner-imitation/residual,
  case-23 authorization, and hardware-readiness states from being conflated.
- Exact implementation commit
  `d28bfde7194d427cdb665d5738aa29ee358fd9b0` passes the complete
  authoritative `.98` CPU-only suite with `842 passed, 11 skipped, 2 warnings`
  in `78.67 s`. Runtime authorization remains absent and no Isaac, capture,
  dataset, BC, PPO, or training process ran.

## Next round after Round 183

- Do not issue a case-23 token without explicit user authorization. When
  authorized, revise only the one-use runtime authorization identity, rerun the
  canonical preflight and CPU suite, execute baseline first, candidate second
  only after baseline pass, finalize, and stop.
- A pair pass opens only a separate corrective-capture review. A pair reject
  must be preserved and diagnosed without threshold relaxation, pulse
  escalation, or skipping directly to cases 6, 2, or 7.
- BC, PPO, holdout access, multi-case merge, production procurement, and
  hardware transfer remain closed.

## Round 184: corrective archive and conversion route generalized CPU-only

- Audit found that the v2 corrective archive and effective-label converter
  still hardcoded case 30. A future admitted case-23 capture would therefore
  have failed conversion even though the profile and paired-canary routes were
  already case-specific.
- Capture admission, save/load, final-status conversion, case-dataset
  validation, save/load, and the conversion CLI now accept an explicit positive
  expected case and an explicit `train` or `validation` split. Case 30/train
  remains the default for complete backward compatibility.
- Non-default evidence fails under the default route. Case 23 succeeds only
  when every boundary receives `expected_case=23`; a default loader rejects the
  resulting dataset until the same explicit case is supplied. Holdout is not a
  legal route and remains structurally rejected.
- The command contract, effective-only labels, `[0.05,0.05,0.02]` scales,
  supervisor reconstruction, previous-effective-action recurrence, source
  hashes, reserved action margin, and all closed learning flags are unchanged.
- Focused archive/converter tests pass `31/31`; the broader capture,
  finalizer, goal, and trainer set passes `67/67`. Capture, conversion, merge,
  BC, PPO, holdout, runtime, and training remain unauthorized.
- SHA-256 identities are capture runtime
  `9d2a9bbced92c53849df9fbe0809bc5dd98f26e54feb5f5595edb96ceedebbe6`,
  case converter
  `2f73df2eb54bbbb560eaf18b7264c8378da25b5a19b4f04443c3300a75f0e9ca`,
  CLI `09a58ec62fede0800afc70df1f8e6c47e618ed097ef96a0fedfcba8a6c88b706`,
  and refreshed goal JSON
  `16297e048015484b22b26c3959b2f848f0a4234a339a66dd274a1f0b8e3285ca`.
- On authoritative `.98`, the sealed real case-30 capture passes default-route
  conversion preflight with `11,411` rows and no output creation. The existing
  converted dataset reopens under the unchanged default route at SHA-256
  `191a44147bc44038a0645bf48a63609463bf280d97b37ddaf884200bd8b52447`;
  case/split remain `30/train` and `valid_for_training=false`.
- Exact implementation commit
  `8911ee870a8d05598ba3ec6c06d14e3b4f298d8a` passes the complete
  authoritative `.98` CPU suite with `849 passed, 11 skipped, 2 warnings` in
  `77.30 s`. No runtime, capture, conversion output, merge, BC, PPO, holdout,
  or training process ran.

## Next round after Round 184

- Do not create case-23 archive or conversion evidence from synthetic tests.
  Case 23 still requires the paired dynamic pass, then a separate capture
  proposal and explicit capture authorization before this route may be used.
- Multi-case merge, DNN training, validation rollout, holdout, and PPO remain
  closed until multiple real converted case datasets exist.

## Round 185: model-based corrective corpus boundary implemented CPU-only

- The multi-case merge design is no longer an unspecified future step. A new
  fail-closed corpus builder accepts only independently converted model-based
  corrective case datasets whose paths and SHA-256 identities are pinned in an
  exact manifest.
- Corpus admission requires at least four distinct training cases and two
  distinct validation cases. Reserved holdouts `[3,5,13,19,24]` are not a
  legal split and are rejected both from the manifest and from corpus rows.
- Each source block preserves the converted dataset, capture, final-status,
  runtime-commit, plan, corrective-profile, and paired-status identities. The
  loader rechecks dense one-to-one source/case/split mapping, contiguous rows,
  clocks, command reconstruction, clipping, and previous-effective-action
  recurrence.
- The learning target remains the effective post-supervisor residual over the
  model-based planner, with action scales `[0.05,0.05,0.02]`. Requested actions
  remain audit-only; no legacy planner-imitation labels can enter this schema.
- The BC loader can audit the two-split corpus and reports the correct
  `model_based_planner_plus_bounded_policy_residual_v1` semantics. The training
  entrypoint deliberately refuses this review-only schema before creating an
  output directory; a later explicit promotion must define a separately
  approved training schema.
- Synthetic coverage proves four-train/two-validation build and round-trip,
  CLI preflight and no-overwrite behavior, and rejection of holdout, duplicate,
  insufficient, hash-mismatched, source-leaking, recurrence-broken, and
  command-tampered inputs. Focused local coverage passes `42/42`.
- Exact implementation commit
  `872d3e7a6430785ac6b06b45ad51c7b8e0a54523` passes the complete
  authoritative `.98` CPU suite with `863 passed, 11 skipped, 2 warnings` in
  `80.91 s`. Corpus module SHA-256 is
  `1365959dde90030c657dac24f53a3dfa22486dae69939cc1027a6d68b6c4bc9d`,
  builder CLI SHA-256 is
  `5730ed3040247e0cb8ab10e04a48614c82bfe2a5e668cb9449b55fb2be61efc1`,
  and trainer SHA-256 is
  `a3da3405915f8ccd913437d73f64d88d7660a6929dd9a03adba281cbe8a45a2a`.
- This round created no real corpus because case 30 remains the only admitted
  converted source. It created no runtime namespace, Isaac process, capture,
  BC, PPO, policy artifact, holdout read, or training process.

## Next round after Round 185

- Keep the corpus builder unused for real output until at least four admitted
  train cases and two admitted validation cases exist. Do not fill missing
  cases with synthetic data or the historical planner-imitation corpus.
- The next runtime measurement remains exactly one separately authorized
  case-23 paired canary. A pass opens a separate capture review only; it does
  not authorize corpus construction or BC.

## Round 186: case-23 paired corrective target passed

- User authorization was implemented as a one-use, mode-`0600`, SHA-256-bound
  token at exact runtime commit
  `d77a1d494be79e442798e34368d865de1cf7ce25`. The token was consumed before
  Isaac and the fresh namespace was
  `20260723_model_based_corrective_teacher_case23_pair_v1_exclusive`.
- The authorization commit passed the authoritative `.98` CPU suite with
  `862 passed, 12 skipped, 2 warnings` in `93.61 s`. The additional Windows
  skip is the expected POSIX token-mode test; WSL token mode, hash, non-symlink,
  canonical contract, clean HEAD/upstream, and exclusive GPU checks all passed
  before launch.
- Baseline ran first and passed all unchanged dynamic gates with `3,273` steps,
  position p95/max `0.0593848022 / 0.0740444738 m`, and zero learned residual.
  Only then did the wrapper launch the single candidate.
- Candidate passed with the same `3,273` steps and position p95/max
  `0.0534133640 / 0.0679118294 m`. P95 improved by `0.0059714383 m`
  (`10.055499%`), exceeding both the `0.003 m` and `2%` admission thresholds.
- Every no-regression check passed: position max, attitude max, pitch, riser
  error, saturation, same plan/seed/clocks, deterministic perturbation, and
  dynamic quality. GPU release also passed.
- Final status SHA-256 is
  `67c8e99a0629a4b1cb4a2981abfe8360c5d9979c4757582dab6d4fb22cd00deb`;
  baseline/candidate SHA-256 identities are
  `b2c4ef1f3bb39086e6bcaf015c10b8f9740c5497030f38321dfb4836b90ace72`
  and
  `130da066f623eb588790fe2467ba44ee1f6918a69841bdeb328b28635c708474`.
- No labels, capture, normalized dataset, corpus, BC, PPO, policy rollout, or
  training were authorized or created. The pass opens only a separate case-23
  corrective-capture review.

## Next round after Round 186

- Build and review a case-23 capture-only contract bound to this exact passed
  pair, the case-23 plan/profile, effective post-supervisor label semantics,
  and a fresh namespace. Do not reuse the consumed pair token.
- Do not launch capture until separately authorized. Conversion, corpus merge,
  BC, PPO, holdouts, and cases 6/2/7 remain closed.

## Round 187: case-23 corrective capture contract passes CPU-only

- Commit `1bbcdbda3239db88c55b1ae36cd6b941e98ee7ed` adds a dedicated
  no-token case-23 capture contract, canonical validator, exclusive wrapper,
  and fail-closed archive finalizer. The generic case-30 validator/finalizer
  accept explicit case identities so the same archive semantics are enforced
  without retaining a hidden case-30 loader default.
- The contract is bound to the passed pair SHA-256
  `67c8e99a0629a4b1cb4a2981abfe8360c5d9979c4757582dab6d4fb22cd00deb`,
  case-23 plan SHA-256
  `ad76ada4cdb9f874da615aa0c6e441be62d9a768b813c597c5dc4e20894042b6`,
  the tracked case-23 corrective and perturbation profiles, frozen LQR gains,
  robot assets, capture runtime, validator, wrapper, and finalizer identities.
- The authoritative `.98` CPU suite passed with `870 passed, 12 skipped,
  2 warnings` in `79.45 s`. The committed no-token preflight then passed every
  pair, identity, plan, profile, asset, namespace, clean-HEAD/upstream, capture
  schema, holdout, and training-closure check. Contract SHA-256 is
  `1d290b6be77e86e69a5ecf025f616bf5cd3c53c336b2ca4d94185b76f5422756`.
- Preflight explicitly reports `runtime_authorized=false`,
  `gpu_launch_authorized=false`, `label_capture_authorized=false`,
  `dataset_creation_authorized=false`, `bc_authorized=false`,
  `ppo_authorized=false`, and `training_started=false`. No runtime namespace,
  token, Isaac process, capture archive, dataset, or training artifact was
  created in this round.

## Next round after Round 187

- Independently review this exact CPU-ready contract. Only explicit user
  authorization may add one mode-`0600`, SHA-256-bound, one-use token for the
  single case-23 capture in the already pinned fresh namespace.
- A successful capture may be reviewed for dataset conversion; it does not
  authorize conversion, corpus merge, BC, PPO, holdout access, or cases 6/2/7.

## Round 188: independent case-23 capture-route review passes

- Commit `86ceb7d31b97c3e8325fe1960042112f5c16fd95` adds the independent
  CPU review and real, non-mocked case-23 archive tests. The audit decision is
  **GO for exactly one separately authorized capture**, not for runtime under
  the current no-token commit and not for conversion or training.
- A synthetic archive now traverses the actual case-23 finalizer and conversion
  implementation. It proves exact case/split preservation, effective
  post-supervisor targets, previous-effective-action recurrence, and continued
  `valid_for_training=false`. The implicit case-30 conversion route rejects the
  archive, and the case-23 finalizer rejects a case-30-labeled archive.
- The downstream corpus code was re-audited. It still requires at least four
  train and two validation cases, keeps the splits case-disjoint, verifies all
  source identities, and excludes holdouts `[3, 5, 13, 19, 24]`.
- The authoritative `.98` suite passed with `872 passed, 12 skipped,
  2 warnings` in `82.82 s`. The no-token preflight passed again at exact clean
  `HEAD == upstream == 86ceb7d31b97c3e8325fe1960042112f5c16fd95`.
- The output namespace remains absent and GPU ownership remains empty. Runtime,
  GPU, capture, dataset, BC, PPO, and training authorization all remain false.

## Next round after Round 188

- Stop at the authorization boundary. Do not issue a token or launch Isaac
  unless the user explicitly authorizes exactly one case-23 corrective-label
  capture after this review.
- If authorized, retain every pinned identity, threshold, command, namespace,
  and capture-only gate; consume the one-use token before Isaac and stop after
  case 23 regardless of pass or rejection.

## Round 189: 750 W riser production-design candidate pinned CPU-only

- The previously vague `750 W class` fallback is now a pinned candidate:
  Leadshine `ELVM8075V48EH-M17-HD` plus `ELD2-CAN7020B`. Current official
  sources support 48 V, 750 W, 2.39/7.17 N m rated/peak torque, brake,
  multi-turn absolute encoder, and a matched 20 Arms/80 Apeak CANopen drive.
- With the unchanged `3:1`, 70 mm/rev, 90% transmission, and 95% reduction
  assumptions, the candidate computes to `550.258929 N` continuous equivalent
  linear force and `1650.776788 N` peak equivalent force. At `1 m/s`, motor
  speed is `2571.428571 rpm`, below the 3000 rpm rating.
- Against the unchanged conservative 8 kg emergency design force
  `276.960000 N`, rated-force margin is `1.986781`, above the project minimum
  `1.15`. The same formula gives a maximum moving mass of `14.803715 kg` at
  that minimum margin. These are calculated screening values, not measured
  carriage capability.
- The mechanism remains a supplier-qualified guided belt axis with `1.50 m`
  mechanical stroke, or a supplier-qualified synchronized two-stage
  telescoping mast if collapsed height requires it. Software camera height
  remains exactly `0.60--1.80 m`; `1.9 m` is explicitly rejected.
- The motor must remain base-mounted rather than moving with the carriage.
  External regeneration, independent anti-fall, hard limits, absorbing end
  stops, safety-rated power removal, gearbox verification, vertical-duty
  approval, and the calibrated bench campaign remain mandatory.
- Evidence SHA-256 is
  `bc4eaa8673cf389e11d639aec06414c5506fd2461e67bad6c5328d0363fd3bb7`.
  Classification is only
  `candidate_ready_for_supplier_and_bench_review=true`; production
  procurement, hardware transfer, simulation motor replacement, runtime, GPU,
  and training remain false.
- The first `.98` full suite correctly rejected CRLF output from the new
  evidence writer: `879 passed, 1 failed, 12 skipped`. Commit
  `50eb597eda7578e3e4f850aa47b3158ac5e203e7` forces LF output and updates the
  sealed script hash. Focused `.98` tests then passed `8/8`, every SHA-256
  verified, and the final authoritative suite passed
  `880 passed, 12 skipped, 2 warnings` in `84.72 s`.
- No robot asset, controller, trajectory, gate, authorization, Isaac process,
  capture, dataset, BC, PPO, or learned policy changed in this round.

## Next round after Round 189

- Hardware: send the pinned 750 W candidate and mechanism contract for supplier
  vertical-axis/gearbox review, but do not place production orders. Populate
  the calibrated bench template when hardware and instruments exist.
- Learning: remain at the separately reviewed case-23 capture boundary. Do not
  infer runtime authorization from the hardware candidate or its passing CPU
  audit.

## Round 190: residual DNN offline promotion is separated from runtime

- Commit `10174a2ca9fc3932317651cc1f6865a477b157e1` repairs the generic
  residual-BC admission semantics. A validation pass and emitted checkpoint now
  set `offline_policy_candidate_ready=true` while always retaining
  `learned_rollout_authorized=false`, `dynamic_holdout_authorized=false`, and
  `separate_dynamic_authorization_required=true`.
- Gate B verifies that separation instead of treating offline MSE as permission
  to launch a learned policy. Gate C therefore remains closed until a separate
  one-shot contract binds the policy SHA-256, source commit, case set, and fresh
  runtime namespace.
- Offline admission also requires every teacher-forced validation prediction,
  and every recursive validation prediction when scheduled sampling is
  enabled, to stay strictly below normalized magnitude `0.95`. This reserves
  action margin in addition to the existing `tanh` bound and dynamic safety
  supervisor.
- No policy architecture, observation, action scale, controller, plan, source
  trajectory, dataset, runtime command, or dynamic threshold changed. No
  checkpoint was trained and no Isaac, capture, holdout, BC campaign, or PPO
  process was launched.
- Focused Mac tests passed `60/60`. The Mac full suite could not collect six
  unrelated mobile-manipulator modules because local `gymnasium` is absent.
  The authoritative `.98` Isaac Python suite passed
  `880 passed, 12 skipped, 2 warnings` in `81.22 s`.

## Next round after Round 190

- Learning data remains the blocking gap: only case 30 has a converted
  model-based corrective case dataset and there are zero admitted multi-case
  training-corpus cases.
- Keep runtime and training closed. The next bounded runtime action remains
  exactly one separately authorized case-23 corrective-label capture using the
  already reviewed capture-only contract. A successful capture still requires
  separate conversion review and does not authorize BC, holdout evaluation, or
  PPO.

## Round 191: corrective validation tranche is selected CPU-only

- The corpus contract requires at least four train cases and two validation
  cases. The existing `[30,23,6,2,7]` tranche addressed only training cases, so
  it could not by itself satisfy the case-disjoint BC admission contract.
- A deterministic selector now reuses the sealed plan portfolio, per-case
  dynamic-quality evidence, split admission, case-30 conversion semantics, and
  the same normalized 12-feature diversity space used by the training tranche.
- Live `.98` evidence exposes four dynamically qualified validation candidates:
  `[8,16,22,32]`. Case `78` remains an honest exclusion because it is absent
  from the dynamic-qualified selection; no threshold was relaxed.
- Farthest-pair selection chose cases `[8,16]` with normalized feature distance
  `2.6496234766`. Case 8 covers the high riser-span/rate end, while case 16
  covers the higher base-speed/yaw-rate and low-height-span end.
- Selection SHA-256 is
  `5576c696e304eb9b9a173970e5fed06e887eccefe2d65a20678415148e22fa0b`;
  selector SHA-256 is
  `8f95c022743cd633d2399953060a8836d7e901f26ae7265258ffb8b72e8dd460`.
  Focused `.98` tests passed `10/10`; the final authoritative suite at
  implementation/evidence commit `4091214` passed
  `884 passed, 12 skipped, 2 warnings` in `83.23 s`.
- The selection is proposal evidence only. Cases 8 and 16 each still require a
  separately reviewed same-seed pair before capture. Runtime, GPU, label
  capture, conversion, merge, BC, PPO, training, and holdout access remain
  false, and no Isaac process was launched.

## Next round after Round 191

- Preserve the current ordering: the already reviewed case-23 capture is still
  the next bounded runtime action and requires separate explicit one-shot
  authorization.
- After case 23 is captured and separately converted, prepare case-specific
  paired-canary proposals for validation cases 8 and 16. Do not reuse the
  case-30 or case-23 corrective profiles without a new quantitative review.

## Round 192: active 400 W and candidate 750 W plants are separated

- A new machine-readable audit traces the generated riser URDF, Isaac actuator,
  thermal monitor, 400 W engineering envelope, and 750 W production-candidate
  evidence together.
- The active simulation remains
  `leadshine_400w_engineering_sample_v1`: `300 N / 1.0 m/s`, continuous thermal
  reference `292.397004 N`, and peak reference `877.191013 N`. This is the
  plant identity under which current dynamic and corrective evidence was
  produced.
- The `ELVM8075V48EH-M17-HD + ELD2-CAN7020B` option is explicitly
  `leadshine_750w_production_candidate_v1`: calculated `550.258929 N` rated and
  `1650.776788 N` peak, but simulation, runtime, training, procurement, and
  hardware-transfer status all remain false.
- No environment or CLI profile switch exists. Activating 750 W requires
  supplier and calibrated bench evidence, coordinated URDF/Isaac/thermal/USD
  changes, and complete static, dynamic, exact-source, and full-79
  requalification. Existing dynamic evidence, corrective captures, and BC
  checkpoints are explicitly non-reusable across that plant change.
- Live `.98` focused tests passed `17/17`. Audit SHA-256 is
  `39a700de3985175e4e8415f1f23beef4264b103daa7ce8847f4ac0fe69f879f7`;
  audit-script SHA-256 is
  `f8bcea857b84104fb5cdbf79aab7b3681fd569aac2ff4e0bf6b6a01e75443eff`.
  The final authoritative suite at commit `c79107b` passed
  `890 passed, 12 skipped, 2 warnings` in `96.08 s`. No Isaac or GPU work was
  started.

## Next round after Round 192

- Hardware remains at supplier/bench review; do not upgrade the simulated plant
  or reuse current training evidence for the 750 W candidate.
- Learning remains at the separately authorized case-23 capture boundary under
  the active 400 W plant identity.

## Round 193: case-23 capture is bound to the active 400 W plant

- Commit `eff05387a93fd8281ae32482121c67105d85819d` adds the active drive-profile
  evidence as a required, hash-pinned case-23 capture identity. The validator
  now requires `leadshine_400w_engineering_sample_v1`, `300 N / 1.0 m/s`, and
  `leadshine_400w_first_order_monitor_v1`.
- The same semantic gate requires the 750 W candidate to remain disabled for
  simulation, runtime, and training. It rejects silent plant upgrades and
  records that a plant switch invalidates existing dynamic, capture, and BC
  evidence.
- Current no-token capture contract SHA-256 and Git blob are
  `18210efd27ba7d6001dc2d81f070f95df5dacd36f78ae417184771d2208f05d8`
  and `01a7f0e957b49c21ccaf68a91c8e476bfaa894aa`. The prior
  `1d290b6be77e86e69a5ecf025f616bf5cd3c53c336b2ca4d94185b76f5422756`
  capture-contract hash is superseded.
- Live `.98` focused tests passed `15/15` with two configuration warnings.
  The no-token preflight passed every pinned identity and the new drive-profile
  semantics at clean
  `HEAD == upstream == eff05387a93fd8281ae32482121c67105d85819d`.
- The authoritative `.98` suite at evidence commit
  `b801ae02c6beb03cc05cfa70017683541057d23e` passed
  `892 passed, 12 skipped, 2 warnings` in `81.10 s`.
- The pair namespace remains the single completed case-23 pair. The capture
  namespace and token remain absent. Runtime, GPU, label capture, dataset
  conversion, corpus merge, BC, PPO, and training remain unauthorized.

## Next round after Round 193

- Do not rerun the passed case-23 pair. The only prepared runtime action is one
  separately authorized case-23 corrective-label capture in
  `20260723_model_based_corrective_teacher_case23_capture_v1_exclusive`.
- Authorization wording must name the corrective-label capture rather than the
  already-completed paired canary. After a capture pass or reject, stop and
  audit before any conversion or learning action.

## Round 194: effective labels require projection-aware temporal handling

- A CPU-only audit reopens the admitted case-30 converted dataset and its
  corrective profile. Across `11,410` transitions, requested teacher intent
  has physical slew violation counts `[0,0,0]` under the unchanged
  `[0.10,0.10,0.04]` limits.
- Effective post-supervisor labels have per-channel violation counts
  `[30,49,8]` across `87` transitions. Every violation touches a transition
  where the same channel was clipped by the deterministic command supervisor;
  there are zero unclipped violations.
- Added `model_based_residual_safety_projection_v1`, a differentiable and
  TorchScript-compatible projection from requested normalized residuals plus
  model commands to final commands and effective actions. It reconstructs the
  tracked case-30 final commands within `1.2e-7`, effective actions within
  `3.9e-6`, and the clipping mask exactly.
- Future model-based BC must keep effective post-supervisor actions as
  pointwise targets but compute loss after this projection. Requested network
  output slew must be regularized/gated independently; clipped effective-label
  transitions must not be naively classified as teacher chatter.
- Audit summary SHA-256 is
  `349e61367603d1945774edfc7839040a6eaef66261991425eeeb43bc4a561c47`.
  It is valid only for BC contract review. Case-23 capture, corpus merge, BC,
  PPO, learned rollout, and training remain unauthorized, and no Isaac/GPU
  process was launched.
- The first `.98` suite correctly found non-portable Windows path separators
  in regenerated evidence: `896 passed, 1 failed, 12 skipped`. Commit
  `d5b936678b50b0af82511e14fb2a79a3bdf818da` normalizes evidence paths to
  POSIX form without changing numerical results. The final authoritative
  `.98` suite passed `897 passed, 12 skipped, 2 warnings` in `81.60 s`.

## Next round after Round 194

- Keep the case-23 capture as the sole prepared runtime action. Do not use this
  temporal audit to bypass its separate authorization.
- After at least four train and two validation case datasets are admitted,
  implement the projection-aware loss and requested-output slew gate in a new
  explicitly authorized model-based training schema. Do not retrofit the
  legacy phase-feedforward BC corpus.

## Round 195: calibrated bench logs gain deterministic numeric reduction

- The existing hardware gate required manually entered RMS current, thermal
  slope, and stopping-distance values. Added
  `cinebotrl_two_wheel_riser_bench_log_reduction_v1` to derive those values
  directly from a calibrated raw CSV.
- The reducer validates exact columns, finite numeric values, strict time
  ordering, boolean flags, one contiguous continuous-duty block, and exactly
  one stop trigger per positive emergency-stop trial. Each trial must continue
  until measured speed is at most `0.02 m/s`.
- It derives time-weighted duty cycle, steady-state minimum speed, RMS/peak
  currents, bus-voltage maximum, temperature maxima, final 300 s motor/drive
  slopes, per-trial trigger speed, and stopping distance. The raw-log SHA-256
  is sealed into the output.
- Script and empty CSV-template SHA-256 values are
  `04870ab00f2ca6a8f24e5a0f062b04d3c68aba92667cd9bc36497667f52fd01c`
  and
  `cf2196889bc3c973055d45c9c50d1f819e4a4c4287014cb5998110b77638acb3`.
  Focused contract tests pass `33/33`.
- The authoritative `.98` Windows-Python suite at implementation commit
  `96b8a4198c660bdce557cee04c0f71e6dba982ac` passes
  `903 passed, 12 skipped, 2 warnings` in `80.46 s`.
- Reducer output is only a numeric merge fragment. Calibration, supplier,
  brake, anti-fall, hard-limit, end-stop, and safety-power evidence remain
  manual and mandatory. No real measurements were collected, and production
  design review, procurement, hardware transfer, simulation profile switch,
  runtime, GPU, BC, PPO, and training remain false.

## Next round after Round 195

- When physical hardware exists, acquire one calibrated raw log using the
  fixed CSV contract, reduce it, merge only the numeric sections into a fresh
  measurement JSON, and run the existing bench audit without relaxing gates.
- Until then, keep the active 400 W simulation profile and the separately
  unauthorized case-23 capture boundary unchanged.

## Round 196: 750 W supplier responses become machine-auditable

- Added `cinebotrl_two_wheel_riser_supplier_response_v1` and a fixed unanswered
  response template for the pinned
  `ELVM8075V48EH-M17-HD + ELD2-CAN7020B` production-design candidate.
- The response gate binds the axis to a supported guided or synchronized
  telescoping architecture, exact `0.60--1.80 m` camera range, `1.20 m`
  software stroke, at least `1.50 m` mechanical stroke, `3:1`,
  `70 mm/rev`, 8 kg, `1 m/s`, `2 m/s2`, `8 m/s3`, `5 m/s2`, and
  `300 N` continuous-force requirements.
- It also requires gearbox speed and emergency-torque margin, written vertical
  duty/tooth-jump approvals, named regeneration and independent anti-fall
  designs, at most `30 mm` declared catch distance, hard limits, absorbing end
  stops, external safety power removal, and seven supporting SHA-256 records.
  A motor holding brake declared for dynamic stopping is rejected.
- Even a complete passing response produces only a fail-closed
  `supplier_evidence` merge fragment for the later bench JSON. It does not
  approve production procurement, hardware transfer, the 750 W simulation
  profile, runtime, training, BC, or PPO.
- The committed blank template has `52` missing fields and correctly returns
  `collect_complete_signed_supplier_response`. Script/template SHA-256 values
  are `ef3ffa688589533e4c12dcef8f41d7ce10f6484369a8601b606281b79acc8ce8`
  and
  `12bb571bfe58d5f224c7ef84b9f21afa9695a05cc5354dd0566b3a6232b5d47d`.
  Focused hardware tests pass `38/38`.
- The authoritative `.98` Windows-Python suite at implementation commit
  `a48a2914c41988ae6006d210809702e487da8823` passes
  `915 passed, 12 skipped, 2 warnings` in `78.56 s`.

## Next round after Round 196

- Send the fixed template to candidate axis/gearbox suppliers and require a
  signed, hash-bound response. Do not infer positive approvals from a quote or
  a catalog speed line.
- The separately authorized case-23 capture remains the next software runtime
  action. Supplier preparation does not authorize capture, conversion, merge,
  BC, PPO, or any GPU work.

## Round 197: 400 W and 750 W bench evidence become disjoint

- Re-audit found that the existing bench template and auditor were pinned to
  the 400 W engineering sample while the new supplier response is pinned to the
  750 W production-design candidate. The Round-196 generic merge flag could
  therefore have encouraged an invalid cross-candidate manual merge and is now
  superseded.
- Added explicit candidate routes to the bench auditor. The 400 W route binds
  the legacy motor/drive and procurement-calculation schema; the 750 W route
  binds `ELVM8075V48EH-M17-HD`, `ELD2-CAN7020B`, its production-candidate
  calculation summary, and an explicit
  `leadshine_750w_production_candidate_v1` profile.
- The historical 400 W template remains byte-identical at SHA-256
  `ec89f0787f081df24d1db60cd114432f3210783e274283f3611e3229e31dabe6`.
  Its omitted profile is accepted only on the fully matching legacy route.
- Added a separate blank 750 W template with SHA-256
  `de3c9f08f35e5a03aea63fd02c5433c1800d74cbc36662dc7b72fd6748657b9e`.
  The supplier merge fragment now carries exact required-candidate identity;
  generic and 400 W merge eligibility are always false.
- Its sealed blank-template audit has SHA-256
  `1ffc8ccae050bd152c276025e0cce842e12accc318a3661e47b4f3c9bf075ad9`,
  passes all candidate-route identity checks, reports `34` missing physical
  fields, and remains rejected.
- Crossed motor/drive/profile/vendor/calculation combinations fail closed.
  Focused candidate-routing and supplier tests pass `44/44`. Neither template
  contains physical measurements, and procurement, hardware transfer,
  simulation switching, runtime, GPU, BC, PPO, and training remain false.
- The authoritative `.98` Windows-Python suite at implementation commit
  `681977133fc8c07e790f8c43832f3e06f0dbde42` passes
  `921 passed, 12 skipped, 2 warnings` in `85.84 s`.

## Next round after Round 197

- Do not build a merge utility until it verifies the supplier fragment's exact
  required-candidate identity against the selected 750 W bench template.
- Runtime progress still requires the separate case-23 corrective-label capture
  authorization; hardware progress requires a signed 750 W supplier response
  or real calibrated bench data.

## Round 198: candidate-bound 750 W bench assembly

- The raw telemetry reducer now has a v2 mode that records an explicit 400 W or
  750 W candidate profile. Legacy v1 output remains available for numeric
  inspection but is always invalid for candidate-bound final assembly.
- Added `cinebotrl_two_wheel_riser_750w_bench_assembly_v1`. It accepts only a
  clean manual 750 W calibration/safety record, a passing v2 reduction bound to
  `leadshine_750w_production_candidate_v1`, and a passing supplier audit whose
  exact required-candidate identity matches the 750 W template.
- Automation-owned manual fields must remain `null/false`; prefilled telemetry,
  supplier approvals, or evidence hashes are rejected rather than silently
  overwritten. Legacy-v1, 400 W, crossed drive, or open authority inputs are
  rejected before output creation.
- The supplier package hash in the merge fragment must exactly equal the
  supplier-response input hash sealed by the supplier audit; a well-formed but
  substituted hash is rejected.
- Structurally valid inputs are assembled and passed directly to the final
  bench audit. Missing manual safety/calibration evidence produces a preserved
  fail-closed result. Even a complete gate pass authorizes only production
  design review, never procurement, hardware transfer, simulation switching,
  runtime, GPU, training, BC, or PPO.
- Assembler and reducer SHA-256 values are
  `977ac4e099ac34834b42ebe72bccb2b33b3faefb545e91e89e30462394bf950c`
  and
  `498aa34d22b0b4c952df35a3b8869d926d4c107e54517a35adf807712d7919b7`.
  Focused assembly/routing tests pass `47/47`.
- Only synthetic healthy/rejection fixtures were used. No real supplier,
  calibration, safety, or telemetry evidence was assembled.
- The authoritative `.98` Windows-Python suite at implementation commit
  `e189e0732e4a650c01f6dbd7edcf1b0b8f25e69f` passes
  `932 passed, 12 skipped, 2 warnings` in `81.86 s`.

## Next round after Round 198

- Real hardware progress then requires the signed 750 W supplier response and
  calibrated physical logs; software runtime still requires the separate
  case-23 corrective-label capture authorization.

## Round 199: case-23 v1 rejection, v2 recovery, and secret-safe boundary

- The exactly-one authorized case-23 v1 corrective-label capture was consumed
  but rejected during Windows output-path argument validation before Isaac
  initialization. No labels, dataset, conversion, BC, PPO, or training were
  produced. The rejected namespace and machine-readable evidence remain
  preserved.
- V2 uses a fresh namespace and fixes the shell-to-Windows path construction.
  The rejected v1 route remains permanently no-token and cannot be retried.
- The security audit isolated a recently committed high-entropy one-use
  authorization digest as the likely GitGuardian trigger. The nine affected
  branch commits were rewritten, the consumed digest was redacted from
  reachable history, and the branch was replaced using an exact
  `--force-with-lease`.
- V2 no longer commits token-derived authorization hashes. A future token must
  be a non-symlink mode-`0600` file outside the repository, with its lowercase
  SHA-256 provided only through the runtime environment and compared in
  constant time.
- Gitleaks scanned the final reachable branch history: `825 commits`,
  approximately `48.39 MB`, and no leaks. The new GitHub secret-scan workflow
  passed at final evidence commit
  `4cd197ec7542aaed5430b77699856505afff0e06`.
- The authoritative `.98` suite at implementation commit
  `4a6e45643a2efd6b713054fa5862b0ae4a506e8d` passed
  `941 passed, 12 skipped, 2 warnings` in `85.23 s`.
- A new CPU-only no-token preflight at clean
  `HEAD == upstream == 4cd197ec7542aaed5430b77699856505afff0e06`
  passed every source, plan, contract, robot, drive-profile, and namespace
  check. Runtime, GPU, label capture, conversion, merge, BC, PPO, and training
  remain false.

## Next round after Round 199

- The sole prepared software runtime action is exactly one separately and
  explicitly authorized case-23 v2 corrective-label capture in
  `20260723_model_based_corrective_teacher_case23_capture_v2_exclusive`.
- Authorization must name that exact one-shot v2 capture. A pass opens only a
  separate conversion audit; a rejection stops for diagnosis. Neither outcome
  automatically authorizes corpus merge, BC, PPO, holdout evaluation, or
  training.

## Round 200: authorized case-23 v2 capture rejects on case-route propagation

- The user authorized exactly one case-23 v2 corrective-label capture. The
  wrapper consumed one out-of-repository mode-`0600` token at clean
  `HEAD == upstream == 526952133a784ad653f4cfebd3e618a23fd4b291`.
- Admission passed case `23`, split `train`, all pinned identities, fresh
  namespace, active 400 W plant, authorization, and closed-training checks.
- Playback rejected before Isaac initialization because it called the generic
  capture loader without forwarding the expected case. The loader's
  compatibility default is case 30, so its internal `case_split` check rejected
  the otherwise valid case-23 admission.
- Playback exited `2`; the finalizer exited `6`; no capture file, heartbeat,
  labels, dataset, BC, PPO, or training was created. The token was deleted and
  GPU ownership was empty after finalization.
- The v2 namespace and evidence are preserved at
  `evidence_20260723_case23_corrective_capture_v2_rejected_case_split`. V2 is
  consumed and must not be retried.
- CPU repair now requires playback to accept an independently pinned
  train/validation split and call the admission loader with both the corrective
  profile's case and that split. The first focused regression suite passes
  `38/38`.

## Next round after Round 200

- Finish a fresh no-token v3 route around the repaired playback identity.
- Do not issue a token or launch v3. A future retry requires a new explicit
  authorization naming exactly one case-23 v3 corrective-label capture.
- Conversion, corpus merge, BC, PPO, holdouts, and training remain closed.

## Round 201: fresh case-23 v3 route passes CPU-only

- V3 uses fresh namespace
  `20260723_model_based_corrective_teacher_case23_capture_v3_exclusive` and
  binds the consumed-v2 rejection manifest. V2 remains non-retryable.
- Playback now requires capture directory, admission, and an independently
  pinned `train` or `validation` split together. It forwards the corrective
  profile's case and the expected split into `load_capture_admission` before
  `AppLauncher`.
- The v3 wrapper pins case 23 and split `train`, retains the exact plan,
  corrective/wrench profiles, LQR, robot, active 400 W plant, safety gates, and
  out-of-band authorization contract, and contains no token or token hash.
- The no-token `.98` preflight passes at clean
  `HEAD == upstream == 90d329cdb1ebaadefefd3696862873eb49f5fd37`.
  The v3 namespace is absent and runtime/GPU/capture authorization remains
  false.
- Focused tests pass `37/37`; the authoritative `.98` suite passes
  `949 passed, 12 skipped, 2 warnings` in `84.07 s`.
- Contract SHA-256 and Git blob are
  `990a20518288f2878fbd7c495dcdc17b8972ffab067f012db7e70a20cc9e3c7c`
  and `6b25a90beb4b7f2c6f7ba769a60e246466c131dd`.

## Next round after Round 201

- Do not launch v3 without a new explicit instruction authorizing exactly one
  case-23 v3 corrective-label capture.
- A v3 pass opens only a conversion audit. A rejection stops for diagnosis.
  Corpus merge, BC, PPO, holdouts, and training remain separately closed.

## Round 202: authorized case-23 v3 completes phase but archive rejects

- The user authorized exactly one case-23 v3 corrective-label capture. The
  wrapper consumed one out-of-repository mode-`0600` token at clean
  `HEAD == upstream == 71ed62558dc4588b4f9a39a3b598e3faf636bd5f`.
- Isaac initialized and reached the full `9.929694 s` trajectory phase in
  `3273` steps. The final heartbeat recorded peak position error
  `0.067912 m`, peak attitude error `0.257883 deg`, peak pitch
  `5.642538 deg`, no pending termination, and no dataset.
- Post-execution archive validation rejected before the dynamic gate result was
  written. Playback did not forward case/split into `save_corrective_capture`,
  so the case-30 compatibility default rejected case-23 rows.
- The v3 wrapper independently invoked the v2-hardcoded finalizer, which
  reported the obsolete v2 namespace.
- No capture archive, conversion, corpus merge, BC, PPO, holdout evaluation, or
  training was created. V3 is consumed and non-retryable.
- Immutable evidence is preserved in
  `evidence_20260723_case23_corrective_capture_v3_rejected_save_route`.

## Round 203: case-23 v4 save/finalizer route is CPU-ready

- Playback now forwards the admitted case and split into the archive save.
  The case-30 compatibility default remains unchanged.
- V4 uses fresh namespace
  `20260723_model_based_corrective_teacher_case23_capture_v4_exclusive` and a
  v4-specific finalizer pinned to case 23 and that namespace.
- The reviewed parent is
  `472130ef622ef90afd6f470783f834d014e41ac0`; the implementation and
  authoritative CPU commit is
  `2eb9604b7e2c030a867d9ab64e536240561c652f`.
- Focused tests pass `37/37`. The authoritative `.98` suite passes
  `959 passed, 12 skipped, 2 warnings` in `90.22 s`.
- A later real-path synthetic case-23 test exercises the actual v4 finalizer:
  archive reopening, explicit case/split validation, every archive/gate/contract
  check, and conversion-only admission all pass. The latest authoritative
  `.98` suite at `d4fb8b4fea89f953a699e4a090d33049c49936dc`
  passes `960 passed, 12 skipped, 2 warnings` in `82.03 s`.
- The `.98` no-token preflight passes every identity and closure check. The v4
  namespace remains absent, with zero GPU owners and zero runtime processes.
- V4 contains no token or token hash. Runtime, label capture, conversion,
  corpus merge, BC, PPO, holdouts, and training remain unauthorized.

## Next round after Round 203

- Do not launch v4 without a new explicit instruction authorizing exactly one
  case-23 v4 corrective-label capture.
- A pass opens only a separate conversion audit. A rejection stops for
  diagnosis. Corpus merge, BC, PPO, holdouts, and training remain separately
  closed.

## Round 204: 750 W external evidence collection is packaged fail-closed

- The production design-review candidate remains Leadshine
  `ELVM8075V48EH-M17-HD` plus `ELD2-CAN7020B`, with `3:1` reduction,
  `0.07 m/rev` effective lead, `1.50 m` minimum mechanical stroke, and the
  immutable `0.60-1.80 m` camera-height contract.
- A machine-readable collection checklist now binds the supplier template,
  production calculation, 750 W bench template, and vendor snapshot by path
  and SHA-256. It preserves all `52` supplier and `34` bench missing fields,
  grouped by section.
- The checklist fixes the evidence order: signed supplier package, complete
  instrumented axis, calibrated raw log and manual safety evidence, numeric
  reduction, then hash-bound final assembly.
- `external_collection_package_ready=true` means only that the empty
  collection package is internally consistent. Real supplier evidence, real
  bench evidence, hardware qualification, production design review,
  procurement, hardware transfer, simulation profile switching, runtime,
  capture, dataset creation, BC, PPO, and training all remain false.
- Focused hardware/checklist tests pass `48/48`. The authoritative `.98` CPU
  suite at `8c8f627f6e4a2d51eefee3a8ccccfaa496d51bd3` passes
  `965 passed, 12 skipped, 2 warnings` in `81.77 s`.

## Next round after Round 204

- Hardware: send the pinned template to the supplier and collect the signed
  response before building the physical 750 W axis and starting bench tests.
  Do not treat the public vendor ratings or empty templates as vertical-axis
  qualification.
- Runtime: case-23 v4 remains the sole prepared capture action and still
  requires a new explicit exactly-one authorization. No v4 namespace or token
  exists. Conversion, corpus merge, BC, PPO, holdouts, and training remain
  separately closed.

## Round 205: projection-aware model-based BC loss is implemented CPU-only

- Added `model_based_projected_effective_action_bc_loss_v1`. Network output
  remains a requested normalized residual; the loss first applies
  `model_based_residual_safety_projection_v1`, then compares the projected
  action with the effective post-supervisor teacher label.
- Added independent
  `requested_physical_residual_slew_hinge_v1`. It uses explicit previous
  requested predictions, positive transition time, case-boundary masks, and
  case-balanced sample weights. Effective-label jumps caused by supervisor
  clipping do not become false requested-output slew penalties.
- The loss rejects non-finite or out-of-bound actions, invalid timing, negative
  weights, non-boolean transition masks, and valid transitions with zero total
  weight. It is differentiable and TorchScript compatible.
- The real case-30 audit covers `11,411` rows. Projected pointwise loss is
  `2.86e-13`; the incorrect direct requested-to-effective MSE is `0.00538`.
  Requested slew violations are `0/0/0`; effective-label violations remain
  `30/49/8`, with `0/0/0` outside projection-clipped transitions.
- The existing trainer still rejects
  `cinebotrl_two_wheel_riser_model_based_corrective_merged_v1` as
  admission-review-only. No corpus, checkpoint, TorchScript policy, BC, PPO,
  learned rollout, capture, or training was created or authorized.
- Focused tests pass `66/66`. The authoritative `.98` suite at
  `da0653d39509839bdd48c5eb81de36ca8e391838` passes
  `976 passed, 12 skipped, 2 warnings` in `89.44 s`.

## Next round after Round 205

- First obtain the separately authorized case-23 v4 corrective capture and
  convert only if its finalizer admits it. Continue diverse train and
  validation captures until the implemented `4` train plus `2` validation
  case-disjoint corpus contract can be populated.
- Only after that corpus receives a separate training-schema admission may the
  projected loss be wired into an authorized BC run. Keep the review-only
  corpus rejection, holdout closure, PPO closure, and learned-rollout closure
  unchanged until their own gates are met.

## Round 206: projection-aware corrective training schema is fail-closed

- Added
  `cinebotrl_two_wheel_riser_model_based_corrective_training_v1` and its
  independent admission schema. Promotion requires the implemented
  case-disjoint review corpus with at least `4` train cases and `2` validation
  cases while holdouts `[3,5,13,19,24]` remain unopened.
- Admission binds the source corpus, promotion commit, projection-aware loss
  module and audit, promotion module, and CLI by SHA-256. The checked-in
  template deliberately has no corpus SHA or commit and keeps
  `training_schema_promotion_approved=false`.
- Promotion preserves every review-corpus array and effective
  post-supervisor label. It derives only same-case previous-row indices,
  positive transition times, transition masks, and per-case sample weights
  that sum to one.
- Tamper tests reject forged admissions, holdout leakage, broken transition
  mappings, invalid timing, unbalanced case weights, and opened learning
  flags. The CLI is preflight-first, refuses overwrite, and the current BC
  trainer still rejects the new schema.
- The implementation is commit
  `6dd9027568ab7afdca68615ad08a9934191ec874`. The local corrective-pipeline
  suite passes `161 passed, 3 warnings` in `17.59 s`; the authoritative `.98`
  suite at the same commit passes `989 passed, 12 skipped, 2 warnings` in
  `95.37 s`.
- No real multi-case corpus or promoted dataset was created. Runtime, capture,
  conversion, BC, PPO, learned rollout, checkpoint creation, and training
  remain closed.

## Next round after Round 206

- The sole prepared runtime action remains exactly one separately authorized
  case-23 v4 corrective-label capture. V3 is consumed and non-retryable.
- If v4 passes, audit and convert it separately, then gather enough disjoint
  train and validation cases to populate the `4+2` review corpus. Only a fresh
  hash-bound promotion admission may create the training schema, and a further
  explicit BC authorization is still required before any learning run.

## Round 207: projection-aware BC entry preflight is integrated CPU-only

- The BC loader now recognizes
  `cinebotrl_two_wheel_riser_model_based_corrective_training_v1` and validates
  it through the production training-dataset loader instead of treating it as
  an unknown archive.
- Explicit `--preflight-only --device cpu` runs
  `projection_aware_effective_label_bc_adapter_v1`. It evaluates the frozen
  projection-aware loss, previous-row timing, case-balanced weights, clipping,
  and requested-action slew diagnostics without creating an output directory,
  optimizer, checkpoint, TorchScript artifact, rollout, or training run.
- `requested_actions_audit` is used only to exercise the projection and slew
  contract. Effective post-supervisor actions remain the future pointwise
  targets. Audit-request reconstruction error is reported independently
  because historical temporal supervisor limits need not be reproduced by the
  pointwise projection.
- A normal invocation with this schema still fails before optimizer creation
  and requires a separate hash-bound BC authorization. Legacy datasets cannot
  use the projection preflight switch.
- Adapter and trainer SHA-256 values are
  `3777aefc6f1fbc14a165f7eb9a897da6db46be4b8c56116e2ddaa69be7d52cb1`
  and
  `ea7d6cc881f54467296286fced2522c07739450961943fe84344cb02c35785a4`.
  The implementation commit is
  `5f46832e706356a7587cd78d67822af0679ed51d`.
- Focused tests pass `64 passed, 5 warnings` in `20.97 s`; the authoritative
  `.98` suite passes `990 passed, 12 skipped, 2 warnings` in `100.07 s`.
- The optimizer path is not integrated, no real promoted dataset exists, and
  BC, PPO, learned rollout, runtime capture, and training remain unauthorized.

## Next round after Round 207

- Runtime remains gated on exactly one separately authorized case-23 v4
  corrective-label capture. V3 cannot be retried.
- CPU-only development may next add the projection-aware optimizer and
  validation-metric kernel behind a hash-bound BC admission contract, but it
  must remain unusable until a real `4+2` promoted dataset and separate BC
  authorization both exist.

## Round 208: projection-aware optimizer and validation kernels pass CPU review

- Added
  `exact_case_balanced_projection_aware_gradient_accumulation_v1`. Each current
  row is paired only with its same-case, same-split predecessor observation.
  Cross-case or cross-split predecessor mappings fail closed.
- Minibatch gradients are scaled against whole-split pointwise and valid
  transition weights, accumulated across the complete split, clipped once,
  and followed by one optimizer step. A deterministic test proves that one
  full batch and uneven seven-row batches produce equivalent parameter
  updates.
- Added `projected_effective_action_case_balanced_validation_v1`. Validation
  passes requested outputs through the frozen command supervisor before
  comparing them with effective targets and reports projected loss,
  case-balanced per-action MSE, zero-request baseline, clipping, requested
  output magnitude, and slew violations.
- A deterministic synthetic three-channel teacher is learned on CPU with
  effective-target MSE reduced below the bounded test gates. The first invalid
  fixture placed the riser at its lower command bound while asking for negative
  residual motion; the safety projection correctly exposed that unreachable
  target. The corrected fixture uses a valid `0.6 m` mid-stroke command rather
  than weakening the learning threshold.
- The adapter SHA-256 is
  `946e205daf98c36c7141536ec2ddd3073190b7d76f0e8224d066e3e309284a48`.
  Implementation commit
  `9dad263fb14ec767cabe912276e80461c9bf4b77` passes the focused suite with
  `67 passed, 5 warnings` in `21.22 s` and the authoritative `.98` suite with
  `993 passed, 12 skipped, 2 warnings` in `102.76 s`.
- The kernel has no executable CLI route and creates no artifact. A real
  promoted dataset, BC admission, optimizer execution, checkpoint, rollout,
  PPO, and training all remain absent or unauthorized.

## Next round after Round 208

- The only prepared runtime action remains one separately authorized case-23
  v4 corrective-label capture.
- CPU work may define the hash-bound BC execution admission and trainer report
  schema around the reviewed kernel. The checked-in template must remain
  unusable, and optimizer execution must stay closed until a real promoted
  `4+2` dataset receives explicit BC authorization.

## Round 209: hash-bound BC execution admission and report contracts

- Added
  `cinebotrl_two_wheel_riser_model_based_corrective_bc_execution_admission_v1`.
  It binds the promoted dataset and promotion commit, exact execution commit,
  trainer, adapter, loss, policy, training-dataset and admission modules, the
  reviewed optimizer/validation contracts, immutable hyperparameters, disjoint
  train/validation cases, and unopened holdouts.
- The checked-in template has null dataset/promotion/execution identities,
  empty splits, `bc_execution_approved=false`, and `bc_authorized=false`.
  It cannot pass even CPU review until populated against a real promoted
  dataset, and cannot pass execution admission without separate approval.
- Added strict execution-report v1 validation. A success claim requires
  contiguous nonnegative epoch history, one optimizer step per epoch, projected
  train/validation metrics, three-channel improvement over zero request,
  requested-action margin below `0.95`, closed holdouts, and matching
  checkpoint/TorchScript files and SHA-256 identities.
- Tests reject forged dataset/code hashes, stale execution commits, changed
  learning rate, holdout leakage, PPO opening, mismatched approval, missing
  artifacts, false success, holdout metrics, and prediction-margin equality.
- Contract module and template SHA-256 values are
  `0c8787a091aaa8051368aeed176e31e2327c0e1c6afa26d70a5e3a88d32b6c49`
  and
  `bdeba23dc8730cc8ee6ce5e36f4335d3450edd2a25c346b9026a181ca4657edf`.
  Implementation commit
  `ebb89fc6ae911329537b238427aa0c104fbe0f4d` passes `82` focused tests
  and the authoritative `.98` suite with
  `1008 passed, 12 skipped, 2 warnings` in `96.17 s`.
- The contract is not connected to the trainer execution path. No real
  promoted dataset, admission approval, optimizer run, artifact, BC, PPO,
  rollout, or training was created or authorized.

## Next round after Round 209

- Exactly one separately authorized case-23 v4 corrective-label capture remains
  the only prepared runtime action.
- CPU work may integrate the admission validator and reviewed optimizer kernel
  into the trainer behind an explicit admission argument. The checked-in
  template and absent real `4+2` dataset must continue to make that path
  unusable by default.

## Round 210: admission-gated projection BC trainer is integrated

- `train_riser_residual_bc.py` now accepts `--bc-admission` only for the
  projection-aware corrective training schema. Legacy datasets cannot use the
  option.
- Before optimizer creation, the route requires clean tracked state with
  `HEAD == upstream`, exact source/execution commit equality, a fully authorized
  admission, matching dataset/code/config hashes, the fixed CUDA training
  configuration, and no legacy masking, scheduled sampling, or attenuated
  previous-action variant.
- The route builds same-case train/validation splits, normalizes observations
  from train rows only, trains with the exact case-balanced projected kernel,
  selects on projected validation loss, and gates all three channels against
  zero-request improvement, normalized output margin, and zero requested-slew
  violations.
- Only a passing offline gate writes a checkpoint and TorchScript policy. The
  resulting execution report is revalidated against admission, dataset,
  history, split metrics, artifact hashes, closed holdouts, and closed
  PPO/learned-rollout state before it is written.
- A bounded synthetic end-to-end test exercises the real policy architecture,
  optimizer, projected metrics, artifact writes, TorchScript export, and final
  report validator on CPU. It is test-only and does not authorize the real
  corpus or GPU training.
- Integration commit
  `02dbff8bca1b8ed2fee3eb2598a3382c0adce0af` passes `84 passed,
  10 warnings` in `21.41 s`; the authoritative `.98` suite passes
  `1010 passed, 12 skipped, 2 warnings` in `93.62 s`.
- No real promoted `4+2` dataset or approved execution admission exists.
  Real optimizer execution, checkpoint creation, dynamic canary, BC, PPO,
  learned rollout, and training remain absent or unauthorized.

## Next round after Round 210

- The sole prepared runtime operation remains exactly one separately
  authorized case-23 v4 corrective-label capture.
- If v4 passes and converts, continue the disjoint corrective tranche toward
  the required `4` train plus `2` validation cases. Only then may a populated
  promotion admission and a separately approved BC execution admission be
  considered.

## Round 211: end-goal evidence is machine-audited

- Added `audit_riser_goal_completion.py`. It independently reads and hashes the
  robot asset audit, 28 kg LQR gate, static/dynamic riser baseline,
  exact-source all-79 portfolio, 750 W design recommendation, and physical
  bench template.
- Optional future learning evidence is fail-closed. A training dataset must
  pass the projection-aware loader; dataset, BC admission, and BC report must
  be supplied together; the existing BC validators must accept them; all-79
  and learned-render reports must bind the resulting TorchScript SHA-256.
- The implementation supports Windows Python running inside a WSL-created
  worktree by falling back to WSL Git. It emits explicit LF bytes, so macOS and
  `.98` Windows Python produce identical reports.
- The clean-parent report at
  `d83fa50f29761f967d204dfeadd7090631f535e4` has SHA-256
  `3b00bad293b061a323d97ced6010bb0313423421b7deaa089202d29dfcdd8c6d`.
  Mac and `.98` both pass `13` focused tests with two known pytest-config
  warnings.
- The audit passes `6/10` required gates: isolated worktree/branch, arm-free
  asset, frozen LQR baseline, 0.60--1.80 m and 1 m/s riser baseline,
  exact-source 79-case reference, and the 750 W motor/mechanism recommendation.
- Four required gates remain false: real disjoint `4+2` corrective corpus,
  authorized projection-aware BC, learned-policy all-79 dynamics, and
  learned-policy rendered rollout audit.
- Physical 750 W bench qualification remains false but is not substituted for
  the requested engineering recommendation. No runtime, capture, BC, PPO,
  learned rollout, or GPU work was started.

## Next round after Round 211

- The only prepared runtime operation remains exactly one separately
  authorized case-23 v4 corrective-label capture.
- A passing converted v4 case advances the first missing gate; it does not
  authorize the remaining train/validation captures, BC execution, learned
  rollout, all-79 evaluation, rendering, or PPO.

## Round 212: future learned evidence is fail-closed

- The completion auditor now hashes its own source file, so a report cannot
  silently survive validator-code drift.
- A learned all-79 report must contain exactly cases `1--79` in order, one row
  per case, all true per-case checks, complete finite teacher/learned metrics,
  three bounded residual channels, and no metric beyond the fixed regression
  budget. The auditor independently recomputes the position-error means and
  aggregate budget.
- A learned-render report must bind the validated BC TorchScript file and
  all-79 report by path and SHA-256. It requires at least three unique case
  videos with verified file hashes, codec, resolution, frame rate, and
  duration, plus true checks for an intact robot, visible riser motion, visible
  camera/gimbal, and no detached links.
- Healthy synthetic reports pass. Tests reject a wrong policy hash,
  duplicate/misaligned case, false row check, metric regression, forged mean,
  modified video hash, and detached-robot review.
- Implementation commit
  `4b92a72750996439458e475c9d75be86332892bb` and status commit
  `83c9b3f9e2d02378694482d12d0cf8907d9e0989` pass `17` focused tests
  on macOS and `.98`.
- The regenerated current report is byte-identical across hosts at SHA-256
  `95e144e77d6ee94e08de2622ad005f3a7a6ac15b0ff9322eb3ee587c68d8558d`.
  The result remains honestly `6/10`; no missing learned-policy gate changed
  state, and no runtime, capture, BC, PPO, rollout, or GPU work started.

## Next round after Round 212

- The only prepared runtime operation remains exactly one separately
  authorized case-23 v4 corrective-label capture.
- Do not construct all-79 or render evidence until a real `4+2` corpus,
  authorized BC report, and separately admitted learned rollout exist.

## Round 213: learned all-79 rollout admission is independently bound

- Added a fail-closed learned all-79 admission contract. It binds the passing
  BC execution report and TorchScript policy, separate validation and reserved
  holdout reports, exact execution commit, evaluator code hashes, cases
  `1--79`, and the fixed tracking, residual-scale, and regression configuration.
- The admission preserves the rollout gate's aggregate null-action semantics:
  every hard/regression check must pass, while null-action comparison uses the
  existing required mean improvement and majority-of-cases criteria.
- The completion auditor refuses any all-79 report unless the BC report,
  learned-rollout admission, validation report, and holdout report are all
  supplied and hash-valid. The checked-in template is intentionally unusable:
  identities are empty and all runtime authorization fields are false.
- Implementation commit
  `abd9d616da4881b844f8506aeff1c9443dcf0b86` and status commit
  `7dc88c0be03e7cf173766746a9e756be789484e3` pass `34` focused tests
  with two known pytest-config warnings on macOS and `.98` Windows Python.
- The completion report was regenerated byte-identically on both hosts at
  SHA-256
  `3d78e8f9179f35fd9117842257877ec1e8deb5260b0ec7d03cd16d40e5f73317`.
  It remains `6/10`; the real corpus, BC policy, all-79 dynamic policy gate,
  and rendered learned-policy audit are still missing.
- No runtime authorization token, Isaac process, capture, BC, PPO, learned
  rollout, or GPU work was created or started.

## Next round after Round 213

- The only prepared runtime operation remains exactly one separately
  authorized case-23 v4 corrective-label capture.
- A passing v4 capture still advances only the corrective corpus. Validation,
  holdout, learned all-79, rendering, and PPO remain separately gated.

## Round 214: current model-based learned all-79 route is fail-closed

- Replaced the historical phase-feedforward learned-rollout assumptions with a
  future-only model-based route using residual scales `[0.05, 0.05, 0.02]`,
  tracking profile `riser_recovery_direction_v4_camera_lever_arm_v1`, and the
  current fixed controller and lever-arm settings.
- The admission now binds the exact-source manifest, all-79 plan manifest and
  every plan file, LQR gains, robot build audit, robot USD, drive profile,
  validation and holdout reports, execution commit, evaluator and wrapper code,
  and every raw teacher and learned rollout JSON.
- Added a CPU-only preflight receipt and a guarded execute/resume wrapper.
  Preflight requires clean `HEAD == upstream == BC execution commit`; resume
  requires byte-identical admission and preflight evidence. A missing or
  mismatched identity rejects before Isaac or a runtime namespace.
- The rollout gate and completion auditor independently revalidate provenance,
  raw rollout hashes, all 79 case rows, model-based command semantics, and the
  fixed regression gates. The legacy all-79 wrapper remains historical and is
  explicitly invalid for this route.
- Implementation commit
  `09df01ab7a0db3581ba36fa2a16d3b70852e97aa` passes `36` focused tests
  with two known pytest-config warnings on macOS and `.98` Windows Python.
  The real `.98` exact-source and v16 plan manifests pass the new CPU-only
  source, timestamp, ordered-anchor, transition, and file-hash checks.
- No learned admission, runtime token, namespace, Isaac process, capture, BC,
  PPO, learned rollout, or GPU workload was created. Goal completion remains
  `6/10`.
- Status commit `e6912d621b909298519e26ac478dee15f6a06326`
  regenerates a byte-identical completion report on macOS and `.98` Windows
  Python at SHA-256
  `20e7b456e9fe481857a3cf3a0ddc1092b41b96fc01254d4c3338fa5f1ec294d0`.

## Next round after Round 214

- The only prepared runtime operation remains exactly one separately
  authorized case-23 v4 corrective-label capture. The consumed v3 instruction
  cannot authorize a retry or v4.
- After a real `4+2` corpus and separately admitted BC exist, populate the
  model-based learned all-79 admission and run preflight before considering a
  learned-policy rollout.

## Round 215: learned validation and holdout use the current model-based route

- Added one fail-closed split-evaluation admission for `validation_canary` and
  `holdout`. Validation cases must come exactly from the passing BC report.
  Holdout remains fixed at `[3,5,13,19,24]` and requires a separately
  hash-bound passing validation report plus explicit model-selection
  completion.
- Both modes use the current
  `model_based_planner + [0.05,0.05,0.02]` residual contract, frozen controller
  settings, source/plan/runtime-asset identities, clean synchronized execution
  commit, CPU preflight, fresh namespace, GPU ownership guard, bounded timeout,
  and validated resume semantics.
- Every split evaluates the complete model-based zero-residual baseline and
  learned TorchScript policy. The baseline is the teacher and null-action
  reference; the historical phase-feedforward holdout wrapper is explicitly
  invalid for this policy.
- The final all-79 admission now opens and validates the contents of each
  split admission and preflight receipt rather than trusting their hashes
  alone. Tests reject rehashed admissions that enable PPO and rehashed
  preflights whose clean-state check is false.
- Implementation commit
  `15a58801ae0fab3fdc5686a1d0a7b0da8b42e6ef` passes `61` focused tests
  on macOS and `.98`; the authoritative `.98` suite passes
  `1068 passed, 12 skipped, 2 warnings` in `111.96 s`.
- No real BC policy, split admission, runtime token, namespace, Isaac process,
  validation rollout, holdout access, capture, PPO, or GPU workload was
  created. The completion result remains `6/10`.
- Status commit `e6ceab046f9bab5954697fca01fa774c0642d92f`
  regenerates a byte-identical completion report on macOS and `.98` at
  SHA-256
  `387d15eb4a812bff8461bd5480be5b5864bae34e0b93205fbc4d1203eb20d51f`.

## Next round after Round 215

- The nearest data-path action remains exactly one separately authorized
  case-23 v4 corrective-label capture. A passing conversion advances the
  training corpus; it does not authorize validation, holdout, BC, all-79, or
  PPO.

## Round 216: learned-policy render evidence is fail-closed

- Added a render admission and CPU preflight tied to a passing learned all-79
  report, the exact TorchScript policy, source/plan/assets, clean execution
  commit, rendering code, and representative cases `[1,15,31,50,73,79]`.
- The guarded wrapper uses the current model-based residual contract and
  D3D12 offscreen RTX rendering under exclusive GPU ownership. It supports
  only a fresh namespace or byte-identical validated resume.
- The machine media auditor hashes every rollout and MP4 and derives codec,
  resolution, frame rate, and duration with `ffprobe`. It cannot set visual
  checks.
- A separate explicit review must confirm intact robot geometry, visible
  riser and camera/gimbal, plausible wheel contact, no detached links, and no
  abnormal oscillation. The v2 finalizer and completion auditor open and
  revalidate admission, preflight, rollout, media, and review artifacts.
- Implementation commit
  `172d4efa8d43418b7eba656117b5004a7df7e708` passes `75` focused tests
  on macOS and `.98`; the authoritative `.98` suite passes
  `1082 passed, 12 skipped, 2 warnings` in `113.60 s`.
- No render admission, namespace, Isaac process, recording, policy, capture,
  BC, PPO, or GPU workload was created. Goal completion remains `6/10`.
- Status commit `1f34408063025397e6376666713b7a4e91765133`
  regenerates a byte-identical completion report on macOS and `.98` at
  SHA-256
  `7de87987498744eba5a48d787559d5c80aa55c20b660d0c752f7f8c0f65053df`.

## Next round after Round 216

- The nearest data-path action remains exactly one separately authorized
  case-23 v4 corrective-label capture. Rendering remains downstream of a real
  admitted BC policy and passing all-79 learned rollout.

## Round 217: case-23 v4 corrective capture passes

- Consumed exactly one out-of-repository mode-`0600` authorization token and
  ran only case 23 in
  `20260723_model_based_corrective_teacher_case23_capture_v4_exclusive` at
  clean synchronized runtime commit
  `31bb9afbf3e9ce6c17e0fc1d2f06b5990e130d1c`.
- The v4 finalizer passed every admission, dynamic, thermal, controller,
  perturbation, heartbeat, GPU-release, archive, clock, and identity check.
  The run completed 3,273 aligned samples and exactly 20 perturbation-active
  rows.
- Position p95/max was `0.053413/0.067912 m`, attitude p95/max was
  `0.148905/0.257883 deg`, and peak pitch was `5.642538 deg`. No command or
  amplitude clipping occurred; eight riser-label rows were slew-limited and
  remain explicit.
- Preserved evidence under
  `evidence_20260723_case23_corrective_capture_v4/`. Capture SHA-256 is
  `f0ea5c59e1f2f0e5f6f91336788d0e0228d079f74a53a4a50d442751b8b23796`;
  final-status SHA-256 is
  `8f7589cdc31b5b6369fea8fda7fbd8b743b57afa78709cf03b2bd600a25833e3`.
- Evidence commit `8e3ea24482cb03eefcc0d55e3acfc0846148d196` and mode-normalization
  commit `46370ecf03957b6921c9dd93bff86ae1cdf54df1` pass the authoritative
  `.98` CPU suite: `1082 passed, 12 skipped, 2 warnings in 114.24 s`.
- Status commit `302bf5ad2fe6ce5b1a760d80608f2934effebb9d` regenerates the
  completion audit at SHA-256
  `0b9159e409494430e8c43be468ad02efb12bd41cd6c907b2d505688175c0915b`.
  The result remains honestly `6/10`: the raw case-23 capture does not count as
  a converted multi-case corpus or a learned policy.
- The one-shot authorization is consumed and no retry is authorized. The
  archive is admitted only for a separately reviewed CPU conversion. No
  conversion, corpus merge, BC, PPO, holdout, or training was started.

## Next round after Round 217

- Review and bind a CPU-only case-23 v4 conversion route, then convert only if
  every archive, case/split, clock, projection, and provenance check passes.
- A successful conversion would make case 23 the second admitted case dataset;
  it would not authorize cases 6/2/7, validation 8/16, corpus merge, BC, PPO,
  holdout, or learned-policy runtime.

## Round 218: case-23 v4 conversion is reviewed without writing output

- Added a hash-bound case-23 conversion review contract and reviewer. They pin
  the passed capture/final status/gate/admission/capture contract, converter,
  dataset/capture modules, reviewed parent, case/split, and the sole prospective
  output path.
- The first `.98` attempt correctly failed before source review because Windows
  Python could not use Windows Git against the WSL-mounted repository. Commit
  `dfeb84b9e1def0ae41b2e1bfe7f32efa8dbd1a95` delegates only repository
  provenance queries to WSL Git while retaining Windows NumPy conversion
  validation.
- Canonical no-write preflight passes at clean synchronized `dfeb84b`: all
  repository, contract, identity, source, effective-label, previous-action,
  non-history observation, case-ID, and three-clock checks are true.
- The prospective dataset is `3273 x 65`; action maxima are
  `[0.216787, 0.084475, 0.284235]`, no rows are clipped, and the source already
  has exact effective previous-action recurrence, so zero history rows would
  change.
- Review evidence is preserved under
  `evidence_20260723_case23_corrective_conversion_review_v1/` at SHA-256
  `d1d18672aa3c5922d04d55df49a903051e395328ca2828c39b538dc28581f270`.
  No converted output, corpus merge, BC, PPO, or training was created.
- Review evidence commit
  `34ca577f5c83ce3d3cf229d261ba467a85e9b5e8` passes the authoritative
  `.98` CPU suite: `1087 passed, 12 skipped, 2 warnings in 109.31 s`.
- Status commit `cc1cd3f3cb055843f16a2e95a590169714f252d5` regenerates the
  completion audit at SHA-256
  `372b0a90e43dbabf5ba1701956eff8694b1b4ded459529c919f7f09c033d538e`.
  The result remains `6/10`; conversion readiness is not a converted case
  dataset and does not satisfy the multi-case corpus gate.

## Next round after Round 218

- A new explicit authorization is required for exactly one CPU-only case-23 v4
  conversion. Reopen and independently validate the produced archive before
  admitting case 23 as the second case dataset.
- The capture authorization is consumed and does not authorize conversion,
  cases 6/2/7, validation 8/16, corpus merge, BC, PPO, holdout, or runtime.

## Round 219: case-23 v4 conversion execution route is guarded

- Added a committed execution contract, fail-closed validator, one-use shell
  wrapper, and post-conversion finalizer for exactly one case-23 train-split
  CPU conversion. Implementation commit is
  `02a090e02f03523c0274151202ab7af204585c32`.
- The route pins the passed source capture/final status/review, converter,
  dataset module, validator, wrapper, finalizer, case, split, namespace, and
  output path. Execution requires an out-of-repository mode-`0600` token and
  a separately supplied SHA-256; the token is consumed before conversion.
- Focused negative, healthy, and synthetic round-trip tests pass:
  `22 passed, 2 warnings`.
- Canonical `.98` no-token preflight passes at clean synchronized `02a090e`.
  Every repository, contract, and identity check is true, and
  `cpu_contract_ready=true`.
- Preserved preflight evidence under
  `evidence_20260724_case23_corrective_conversion_execution_cpu/` at SHA-256
  `2034291914a515ee633d36d46bcce5d457aec630abbf5da9f4bcd3efc00623d2`.
- No authorization token was issued or consumed. The production conversion
  namespace and output are absent; conversion, merge, BC, PPO, and training
  remain false. Goal completion remains `6/10`.

## Next round after Round 219

- Require a new exact authorization for one CPU-only case-23 v4 conversion.
  If authorized, run the guarded route once, reopen the produced archive, and
  admit only if every clock, label, recurrence, provenance, and closed-training
  check passes.
- Do not infer conversion permission from the consumed capture authorization
  or this CPU-ready preflight.

## Round 220: case-23 conversion token route is repaired for `.98`

- The first authoritative full `.98` suite exposed one cross-platform failure:
  `1094 passed, 1 failed, 12 skipped, 2 warnings`. The healthy authorization
  test could not satisfy mode `0600` because `.98` mounts both `/mnt/c` and
  `/mnt/g` as DrvFS without Linux metadata; files there report mode `0777`.
- This was a real execution-contract defect, not relaxed as a test exception.
  Commit `298805562202320c72319f7adb0f955fd9568116` requires authorization
  tokens on WSL ext4 and supports the matching pinned-Ubuntu
  `\\wsl.localhost` path for Windows Python.
- The validator rejects alternate WSL distributions, and the shell wrapper
  still independently verifies WSL mode `0600`, non-symlink status, and the
  out-of-band SHA-256 before consuming the token.
- The corrected real `.98` focused suite passes:
  `9 passed, 2 warnings`.
- Fresh no-token preflight evidence is preserved under
  `evidence_20260724_case23_corrective_conversion_execution_cpu_v2/` at
  SHA-256
  `b1d76609bac8982d3bd4af818c15e18ad58dc88b24093617a8fe90018069f739`.
  Every repository, contract, and identity check passes.
- Evidence commit `3040a6db2b70b1fced0fd306ea17e2a008009bd3`
  passes the authoritative `.98` suite:
  `1096 passed, 12 skipped, 2 warnings in 130.96 s`.
- Status commit `5801a61aeffc4f9bb8caa85d0c32ac124b11ca29`
  regenerates a byte-identical completion audit on macOS and `.98` at
  SHA-256
  `19151b0fabd21fc7c72d9e97e54445a2812c476454fd8370c9bfb2a4bb07c347`.
  It remains `6/10` with the same four learning blockers.
- No authorization token, conversion namespace, output, merge, BC, PPO,
  training, Isaac process, or GPU workload was created. The route is CPU-ready
  but conversion remains unauthorized.

## Round 221: real corrective corpus intake is machine-readable

- Added a fail-closed intake auditor for the real model-based corrective
  corpus. It reopens case datasets, binds the selected train and validation
  tranches, verifies case-23 capture/conversion state, and refuses to treat
  captures or preflights as converted data.
- The auditor supports a future case-23 dataset only when a separate passing
  conversion final status binds the exact dataset SHA-256 and keeps merge, BC,
  PPO, and training closed.
- Current evidence proves `1/4` converted train cases (`[30]`) and `0/2`
  validation cases. The pending minimum train path is `[23, 6, 2]`; validation
  remains `[8, 16]`; case 7 remains an additional diverse train candidate.
- Implementation commit
  `e6a3688de943864f043691f407de90eb0e51f75d` emits byte-identical reports on
  macOS and `.98`. Evidence is preserved under
  `evidence_20260724_model_based_corrective_corpus_intake_v1/` at SHA-256
  `3d8f3da9c23ddb9d63a26afb3bec15324d8ce61a3e0b900c7cdf67f91c9e20bf`.
- Evidence commit `fa4c834ae78ed65c74bd1e369c9e4868ea0c2d44`
  passes the authoritative `.98` suite:
  `1107 passed, 12 skipped, 2 warnings in 112.64 s`.
- Status commit `0982661880c389ab6d5f56f8d2001d3674a29e23`
  regenerates a byte-identical completion audit on macOS and `.98` at
  SHA-256
  `5dda6698bed9459329d25a3d6356891dd8a1cd91c6abe2a516fc3b07b336eefa`.
  It remains `6/10`; the partial intake is not the required real corpus.
- The intake audit passes, but `corpus_manifest_ready=false`. It authorizes no
  conversion, merge, capture, runtime, BC, PPO, or training.

## Round 222: case-6 paired readiness is audited CPU-only

- Preserved and hash-bound the selected case-6 smoothed exact-source plan and
  its passing zero-residual dynamic evidence. The plan contains `807` source
  states, `806` transitions, a `15.942736 s` source clock, and a
  `17.737275 s` execution clock. Camera height remains within
  `0.600000-1.528812 m`.
- The zero-residual gate remains healthy at `0.118125/0.127080 m` position
  p95/max and `6.393436 deg` peak pitch. No learned residual is applied and no
  dataset is produced.
- Case 6 is not paired-profile ready. Base-linear, base-yaw, and proxy-rate
  commands reach their current limits, the camera lever-arm correction
  saturation ratio is `0.958710`, and only one conservative `0.105 s`
  low-motion window is available.
- Reusing the case-23 perturbation profile is explicitly forbidden. The next
  case-6 step is a CPU-only, case-specific corrective and perturbation profile
  design; it is not a runtime authorization.
- Implementation/evidence commit
  `95666c94930eba4f9726a5d8ff3dbb7dcea83a40` passes `24` focused tests on
  macOS and `.98`. Both platforms regenerate byte-identical evidence at
  SHA-256
  `0f74c92b93040b126f8e25ac2470603203f8e07df6e923c1462c3cd24891b5ec`.
- Status commit `993b81a5bb11e18d5a08d79b667beddb5d9a3b10` passes the
  authoritative `.98` CPU suite:
  `1114 passed, 12 skipped, 2 warnings in 114.32 s`.
- Verification status commit `b269676a0f5d2afd8184851657cbf65597871e0e`
  regenerates a byte-identical completion report on macOS and `.98` at
  SHA-256
  `ee548294277aa14eaf4e4e140c0b4ed26cdc89849a056e23d52291315cb6a6ab`.
  The required-gate result remains `6/10`.
- The requested case-23 v4 capture was not repeated: Round 217 already consumed
  the sole authorization and sealed the successful capture. No token, Isaac
  process, GPU workload, conversion, merge, BC, PPO, or training was created.
  Goal completion remains `6/10`.

## Next round after Round 222

- The nearest data-path action remains a separately authorized, exactly-once
  case-23 v4 CPU conversion. The capture authorization cannot be reused for
  conversion or another capture.
- In parallel, continue CPU-only design of the case-6-specific paired profile.

## Round 223: case-6 corrective profiles are formula-bound

- Added a reproducible CPU-only builder for case-6 corrective and wrench
  profiles. It retains `75%` of the observed raw case-6 residual envelope under
  policy scales `[0.05,0.05,0.02]`, producing maximum residuals
  `[0.028768,0.007953,0.001787]` with a `0.30 s` slew horizon.
- The proven `20 N`, `20`-step pulse starts at phase `17.185169 s` and fits
  within the only verified low-motion window. Local base/yaw headroom is
  `0.110688/0.182392`, riser-target headroom is `0.279227 m`, and the recovery
  tail is `0.452106 s`.
- The provisional `28 kg` lower model gives a `2 Ns` impulse and
  `0.003571 m` free-body displacement. This remains an observability screen,
  not a closed-loop prediction.
- Commit `8667320d83c3fd3518927bfca3819b061532cb50` passes `50` focused
  tests on macOS and `.98`. Both platforms regenerate byte-identical profile
  and proposal artifacts. Proposal SHA-256 is
  `649aeaa56333b9172e4a25d0f34a716d09a51da7cfa1f269ca14ab32da384b64`.
- Status commit `3e3a4f070b1384a8a798d908e4bc174060921ba7` passes the
  authoritative `.98` CPU suite:
  `1124 passed, 12 skipped, 2 warnings in 113.57 s`.
- Verification status commit
  `89d74defd0783cb492a03388592cd908d09d3050` regenerates a
  byte-identical completion report on macOS and `.98` at SHA-256
  `b9804473371407657a1206ba18806b508e5745fc879a3d9071c0af6a8bb8c0ce`.
  The required-gate result remains `6/10`.
- `pair_profile_cpu_ready=true`, but no runtime route or authorization exists.
  No token, GPU launch, capture, conversion, merge, BC, PPO, or training was
  created. Goal completion remains `6/10`.

## Next round after Round 223

- Implement a disabled-by-default, hash-bound case-6 paired-canary runtime
  contract and CPU validator. Do not issue a token or launch Isaac.
- The nearest actual data-path action remains the separately authorized
  case-23 v4 CPU conversion.

## Round 224: case-6 paired-canary contract is fail-closed

- Added a canonical, hash-bound case-6 paired-canary contract and CPU
  validator. The contract pins `17` identities covering the selected plan,
  readiness and profile evidence, profiles, provisional `28 kg` gains, robot
  assets, playback, and corrective/perturbation runtime code.
- The committed contract contains no usable authorization:
  `runtime_authorization_token_sha256=""`, `authorization_token_issued=false`,
  `runtime_authorized=false`, and `gpu_launch_authorized=false`.
- The wrapper exposes CPU preflight only. Its `--execute` path rejects with
  `runtime_authorization_not_issued` before Python or Isaac and cannot create
  the reserved runtime namespace.
- The evidence contract explicitly records
  `runtime_route_contract_ready=true` separately from
  `execution_route_complete=false`. CPU readiness therefore cannot be
  presented as an executable canary.
- Implementation commit
  `10bb5db127efdcff5518f530c5bf3d54ab509be8` passes `26` focused tests on
  macOS and `.98`. Both hosts validate all `17` identities with no failed
  checks.
- The authoritative `.98` CPU suite passes:
  `1133 passed, 12 skipped, 2 warnings in 123.36 s`.
- Evidence is preserved under
  `evidence_20260724_case6_pair_contract_cpu_v1/`. The contract SHA-256 is
  `664be51b4b0504292e35a3dc0d227abcc6de99bf07a141d307d7ae9232eb1c70`.
- No token, runtime namespace, Isaac process, GPU workload, label capture,
  dataset creation, BC, PPO, or training was created. Goal completion remains
  `6/10`.

## Next round after Round 224

- The nearest actual data-path action remains exactly one separately
  authorized case-23 v4 CPU conversion. The consumed capture authorization
  cannot authorize that conversion.
- Case 6 may advance only through a separate CPU-only implementation of the
  paired-result finalizer and guarded execution route. That work must remain
  tokenless and must not launch Isaac until separately reviewed and authorized.

## Round 225: case-6 paired route is complete but unauthorized

- Added a v2 contract with a fresh
  `20260724_model_based_corrective_teacher_case6_pair_v2_exclusive` namespace.
  It pins `18` identities, adding the paired-result finalizer to the previous
  plan, profile, controller, robot, playback, validator, and wrapper set.
- Completed the guarded baseline/candidate route. Both rollouts use the same
  case-6 plan, reset seed, frozen controller settings, dynamic thresholds, and
  deterministic wrench. Each rollout has a `600 s` timeout and candidate
  execution requires a dynamically passing baseline.
- The finalizer reuses the shared paired-assessment contract and additionally
  binds case 6, source/execution clocks, plan and profile identities,
  perturbation telemetry, closed capture paths, heartbeats, and GPU release.
- Synthetic CPU tests prove admission of one safe measurable improvement and
  fail-closed rejection of weak improvement, source-clock mismatch, corrective
  profile mismatch, any capture path, failed GPU release, and unauthorized
  admission.
- Implementation commit
  `0f3e13ee3817d9216bf5cef70541493e519e0abb` passes `50` focused tests on
  macOS and `.98`. Both hosts validate all `18` identities with no failed
  checks.
- The authoritative `.98` CPU suite passes:
  `1141 passed, 12 skipped, 2 warnings in 124.94 s`.
- Contract SHA-256 is
  `ab5760179f4f2f79bce1ef06525316a0127dd97c939325c102d6538fbfa39a1f`.
  Evidence is preserved under
  `evidence_20260724_case6_pair_route_cpu_v2/`.
- The committed authorization hash is empty. `--execute` rejects before
  Python/Isaac, the v2 namespace remains absent, and no token, capture, dataset,
  BC, PPO, or training was created. Goal completion remains `6/10`.

## Next round after Round 225

- The nearest actual data-path action remains exactly one separately
  authorized case-23 v4 CPU conversion.
- The independent case-6 runtime step is now reviewable as exactly one
  separately authorized v2 paired canary. Do not issue that authorization or
  launch Isaac implicitly.

## Round 226: case-2 readiness exposes a structural profile constraint

- Imported and hash-bound the selected case-2 exact-source smoothed plan and
  its passing zero-residual dynamic gate from `.98`. Their SHA-256 values
  exactly match the tranche-selection contract.
- The plan contains `480` source states and `479` transitions, with separate
  `9.439314 s` source and `18.241928 s` execution clocks. The source-anchor map
  is exactly one-to-one, initialization is separate and empty, and camera
  height remains within `1.174803-1.362249 m`.
- The zero-residual gate passes at `0.139830/0.153514 m` position p95/max and
  `6.632458 deg` peak pitch, but its p95 margin is only `0.010170 m`.
  Base-linear, base-yaw, and proxy-rate commands reach their frozen limits,
  while camera lever-arm correction saturation is `0.943907`.
- No interval of at least `0.1 s` satisfies the existing conservative
  low-motion envelope. Case 2 is therefore evidence-ready but not
  profile-ready. Reusing either the case-23 or case-6 profile is explicitly
  forbidden.
- Implementation commits
  `9a16e5ca18ab4da96d78269234b674eec88d103c` and
  `b7d02c1a5b72e61394921f837d9845a850fc5e64` pass `15` focused
  tests on macOS and `.98`.
- The authoritative `.98` CPU suite passes:
  `1149 passed, 12 skipped, 2 warnings in 124.40 s`. Both hosts regenerate
  byte-identical evidence at SHA-256
  `5e346af90c7d4888914baf6bcf8adaef65957bb20a97a11b8666590ca9067f5c`.
- No token, runtime namespace, Isaac process, GPU workload, label capture,
  conversion, merge, BC, PPO, or training was created. Goal completion
  remains `6/10`.

## Next round after Round 226

- Design a CPU-only structural corrective profile for case 2 that excludes
  saturated segments and proves command, clock, and supervisor invariants.
- The nearest actual data-path action remains exactly one separately
  authorized case-23 v4 CPU conversion. Case 6 remains separately reviewable
  as one v2 paired canary.

## Round 227: case-2 uses natural error and projection, not another wrench

- Rejected the case-6 pulse-profile pattern for case 2 because no shared
  low-motion interval exists and the baseline already runs close to its
  position-p95 gate.
- Built a natural-error corrective profile that retains `25%` of the observed
  raw envelope. Maximum residuals are
  `[0.010247,0.004541,0.000614]` with a `0.4 s` slew horizon.
- The existing zero-residual trace provides sufficient excitation without an
  external wrench: `42/47` trace samples exceed `0.03 m` position error and
  the trace maximum is `0.153167 m`.
- Bound the profile to the existing deterministic model-based safety
  projection. Outward negative linear corrections require projection on
  `430` transitions, and outward positive yaw corrections on `103`.
  Projection is contractive, preserves all command limits, and leaves riser
  corrections unprojected over the frozen plan envelope.
- Any future corrective capture must store effective projected residuals,
  never unavailable requested commands. This matches the existing
  projection-aware BC contract and does not grant capture or training.
- Implementation commit
  `2911a0bdb45cf4c83d57a490dfdff9d9a9e90a58` passes `50` focused
  tests on macOS and `.98`. The authoritative `.98` CPU suite passes:
  `1161 passed, 12 skipped, 2 warnings in 122.95 s`.
- The profile SHA-256 is
  `b08434850b63c3370172788372f9b2f89ee388c07e8292f837838354ba111b50`;
  proposal SHA-256 is
  `6af820c7a26094bb2770be2ca80923cf9f378386fca0252c4cbe37eeaf50fdfb`.
- No token, runtime namespace, Isaac process, GPU workload, label capture,
  conversion, merge, BC, PPO, or training was created. Goal completion
  remains `6/10`.

## Next round after Round 227

- Implement a disabled-by-default, hash-bound case-2 natural-error paired
  runtime contract and CPU validator. It must omit a wrench profile and pin
  the safety-projection source.
- The nearest actual data-path action remains exactly one separately
  authorized case-23 v4 CPU conversion.

## Round 228: case-2 paired route is complete but unauthorized

- Added a tokenless case-2 natural-error paired-canary contract with a fresh
  `20260724_model_based_corrective_teacher_case2_natural_error_pair_v1_exclusive`
  namespace and `19` pinned identities.
- Kept the shared playback and corrective-teacher runtime byte-stable. A
  case-specific adapter observes requested versus effective post-supervisor
  residuals while returning the original command object unchanged.
- Corrected the case reset seed to `20260718`, which is the runtime invariant
  `20260716 + case`. The draft had incorrectly copied case 6's `20260722`.
- The paired route uses no external wrench. It runs the exact-zero baseline
  before the natural-error candidate, requires the baseline dynamic gate to
  pass, uses unchanged safety and quality thresholds, and forbids every
  capture and training output.
- Implementation commit
  `39b63d8efbe7954c4f0fcc3aa762caee7695497e` passes `28` focused
  route tests on macOS and `.98`, plus `320` macOS corrective/runtime
  regressions including historical case-23 v4 identity validation.
- The authoritative `.98` CPU suite passes:
  `1183 passed, 12 skipped, 2 warnings in 137.70 s`.
- macOS and `.98` preflights pass all `19` identity and contract checks at
  clean pushed `HEAD`. Contract SHA-256 is
  `1a5b9190bf656cd52b973193efd90bd60aa316dda2b67c41f93f21376626872c`.
- `--execute` rejects before Python or Isaac with exit code `4`; the namespace
  remains absent. No token, Isaac/GPU run, label capture, dataset, BC, PPO, or
  training was created. Goal completion remains `6/10`.

## Next round after Round 228

- The nearest actual data-path action remains exactly one separately
  authorized case-23 v4 CPU conversion. The consumed v4 capture authorization
  cannot be reused.
- Case 6 remains separately reviewable as one v2 paired canary.
- Case 2 is now separately reviewable as one natural-error paired canary; no
  case-2 runtime authorization has been issued.

## Round 229: case-7 readiness exposes usable windows and lever-arm risk

- Imported the selected case-7 exact-source plan and zero-residual Gate C
  result from `.98`. Their SHA-256 identities exactly match the diverse
  tranche manifest.
- The plan preserves `663` source states and `662` transitions, with separate
  `12.940941 s` source and `18.1173174 s` execution clocks and an exact
  one-to-one source-anchor map.
- Camera height stays within `0.600000-1.605452 m`. Maximum base, yaw, riser,
  and proxy rates remain below the frozen limits with material headroom.
- The zero-residual gate passes at `0.130904/0.142948 m` position p95/max and
  `6.169692 deg` peak pitch.
- Four conservative low-motion windows are available; the longest lasts
  `3.431994 s`. Unlike case 2, case 7 can support a bounded perturbation
  window.
- Camera lever-arm correction saturation is still `0.919479`, so case 7
  requires its own profile. Case-23, case-6, and case-2 profile reuse remains
  forbidden.
- Implementation commit
  `b98d1fe1c008372373b300502fb6f0bdbd6200d4` passes `29` focused
  tests on macOS and `.98`. Both hosts regenerate byte-identical evidence at
  SHA-256
  `8652d92eaf8196d77bc3aab33e011029008b0ebdc3b31d035dda29417d2a3df1`.
- The authoritative `.98` CPU suite passes:
  `1191 passed, 12 skipped, 2 warnings in 138.28 s`.
- No token, runtime namespace, Isaac/GPU run, label capture, conversion,
  merge, BC, PPO, or training was created. Goal completion remains `6/10`.

## Next round after Round 229

- Design a CPU-only case-7 corrective and perturbation profile inside one of
  the audited low-motion windows. Prove command headroom, source/execution
  clocks, post-supervisor label margin, and recovery-tail invariants.
- The nearest actual data-path action remains exactly one separately
  authorized case-23 v4 CPU conversion.

## Round 230: case-7 profile is CPU-ready without opening runtime

- Built a case-specific corrective profile that retains `50%` of the observed
  raw envelope. Maximum linear, yaw, and riser residuals are
  `[0.019165, 0.010077, 0.001263]`, with a `0.35 s` slew horizon.
- Placed one deterministic `20 N`, `0.10 s` longitudinal pulse at execution
  phase `2.851306 s` in source window `27..125`. It covers four source
  samples and leaves `15.166011 s` for recovery.
- Proved the selected pulse window is fully unclipped. Over the complete plan,
  base and yaw stay unclipped and only four initial negative riser transitions
  project at the lower height bound.
- Kept the effective post-supervisor residual as the only admissible future
  label. The maximum effective normalized action remains `0.383306`.
- Implementation commit
  `9d4777ac7e9b90d3a57061c6f8e1bb2b38b8a0c5` passes the focused
  `47`-test suite on macOS and `.98`. Builder, test, profile, and proposal
  hashes match across both hosts.
- The authoritative `.98` CPU suite passes:
  `1205 passed, 12 skipped, 2 warnings in 137.71 s`.
- A repeated case-23 v4 capture authorization was not replayed because the
  one-shot authorization was already consumed and its successful capture is
  sealed.
- No token, runtime namespace, Isaac/GPU run, label capture, conversion,
  merge, BC, PPO, or training was created. Goal completion remains `6/10`.

## Next round after Round 230

- Implement the CPU-only case-7 paired runtime contract and preflight around
  the new profiles, without issuing authorization or creating a namespace.
- The nearest actual data-path action remains exactly one separately
  authorized case-23 v4 CPU conversion. The consumed v4 capture authorization
  cannot be reused.

## Round 231: case-7 paired execution route is complete but unauthorized

- Added a fresh case-7 paired contract, validator, wrapper, and finalizer at
  implementation commit
  `7f311139c93247b78e0e4e503525ebe002f825c0`.
- Bound `18` plan, profile, robot, controller, runtime, and route identities.
  The route fixes the case-7 reset seed at `20260723` and uses a fresh
  `case7_pair_v1_exclusive` namespace.
- Preserved baseline-first execution. The candidate cannot run unless the
  exact-zero baseline passes its unchanged dynamic gate and releases the GPU.
- Preserved the same plan, seed, physics, and deterministic perturbation for
  both rollouts. No capture or dataset argument exists in the wrapper.
- Mac and `.98` route regressions pass `50` tests. The `.98` preflight passes
  every check with all `18` identities matching.
- The authoritative `.98` CPU suite passes:
  `1224 passed, 12 skipped, 2 warnings in 150.03 s`.
- `.98 --execute` rejects before Python or Isaac with code `4` and
  `runtime_authorization_not_issued`. The runtime namespace remains absent.
- No token, Isaac/GPU run, label capture, dataset, BC, PPO, or training was
  created. Goal completion remains `6/10`.

## Next round after Round 231

- The case-7 pair is separately reviewable for one future paired canary, but
  no case-7 runtime authorization has been issued.
- The nearest actual data-path action remains exactly one separately
  authorized case-23 v4 CPU conversion. The consumed v4 capture authorization
  cannot be reused.

## Round 232: case-8 validation readiness is hash-bound and CPU-verified

- Bound the first validation case to the dedicated validation-tranche
  selection instead of reusing the train-tranche selection or any train-case
  profile.
- Preserved `663` immutable source anchors and `662` transitions, with
  separate `12.940941 s` source and `18.1173174 s` execution clocks and an
  empty, separate initialization phase.
- Verified the selected zero-residual dynamic gate at
  `0.131254/0.143331 m` position p95/max and `6.147057 deg` peak pitch.
  The playback applied zero residual, did not terminate, and produced no
  training artifact.
- Found four conservative low-motion windows; the longest is `3.431994 s`.
  Camera lever-arm correction saturation is `0.920061`, so a dedicated
  validation profile is required. Reuse of case-30, case-23, case-6, case-2,
  or case-7 profiles remains forbidden.
- Implementation commit
  `155affbd7348d6eecae541a08a6908cc31e397ed` passes the focused `23`-test
  readiness suite on macOS and `.98`. Both hosts regenerate byte-identical
  evidence at SHA-256
  `8ba7d6613b53cca7f266cb0052680b2d0cc4a71e363e10abbadcb8c789526983`.
- The authoritative `.98` CPU suite passes:
  `1232 passed, 12 skipped, 2 warnings in 151.02 s`.
- The repeated case-23 v4 capture request was not replayed: that one-shot
  authorization was already consumed by the sealed successful capture.
- No token, runtime namespace, Isaac/GPU run, label capture, conversion,
  merge, BC, PPO, or training was created. Goal completion remains `6/10`.

## Next round after Round 232

- Design CPU-only case-8 validation corrective and perturbation profiles
  inside an audited low-motion window. Prove command headroom, unchanged
  source/execution clocks, post-supervisor label margin, and recovery-tail
  invariants without reusing a train-case profile.
- The nearest actual data-path action remains exactly one separately
  authorized case-23 v4 CPU conversion. The consumed v4 capture authorization
  cannot be reused.

## Round 233: case-8 has a dedicated validation profile

- Built a validation-only case-8 corrective profile from its own readiness
  evidence and selected plan. The implementation reuses the audited formula
  engine but not any case-30, case-23, case-6, case-2, or case-7 profile file
  or parameter set.
- Retained `40%` of the observed case-8 raw residual envelope with maximum
  residuals `[0.015366,0.008148,0.001003]` and a `0.40 s` slew horizon.
  This is more conservative than case 7's `50%` and `0.35 s` design.
- Selected an `18 N`, `0.10 s` longitudinal pulse at execution phase
  `2.851306 s`, covering four source samples and leaving `15.166011 s` for
  recovery. This is smaller than case 7's `20 N` pulse.
- Proved the pulse window is fully unclipped. The full-plan projection is
  contractive, with only four initial negative riser transitions projected at
  the lower bound. The maximum effective normalized action is `0.307323`.
- For the provisional `28 kg` plant, the pulse impulse is `1.8 Ns`; its ideal
  free-body displacement lower screen is `0.003214 m`. This remains an
  observability screen, not a closed-loop response claim.
- Implementation commit
  `0a671f3bd074a45e96648f22961c8b9a162e5545` passes `40` focused tests
  on macOS and `.98`. Windows regenerates all profile artifacts byte-for-byte.
- The authoritative `.98` CPU suite passes:
  `1244 passed, 12 skipped, 2 warnings in 152.91 s`.
- `pair_profile_cpu_ready=true`, but the runtime route is not implemented and
  no authorization token exists. No Isaac/GPU run, label capture, conversion,
  merge, BC, PPO, or training was created. Goal completion remains `6/10`.

## Next round after Round 233

- Implement a disabled-by-default, hash-bound case-8 validation paired runtime
  contract and CPU preflight. It must preserve the validation split, exact
  clocks, identical baseline/candidate seed and physics, and forbid every
  dataset or training output.
- The nearest actual data-path action remains exactly one separately
  authorized case-23 v4 CPU conversion. The consumed v4 capture authorization
  cannot be reused.

## Round 234: case-8 held-out validation route is complete but unauthorized

- Added a dedicated validation assessor rather than weakening the existing
  train-only corrective-teacher admission gate. A passing case-8 pair can
  report held-out validation improvement but cannot admit teachers, labels, or
  training data.
- Added a hash-bound contract, validator, wrapper, and finalizer with `19`
  identities, validation split, reset seed `20260724`, and the fresh
  `case8_validation_pair_v1_exclusive` namespace.
- Preserved baseline-first execution with identical plan, seed, physics,
  thresholds, and deterministic `18 N`, `0.10 s` perturbation. The candidate
  is unreachable unless the baseline dynamically passes and releases the GPU.
- Repaired cross-platform contract validation to use WSL Git from Windows
  Python and explicit LF serialization. This preserves exact blob and byte
  identities across macOS, WSL, and Windows.
- Mac and `.98` preflights pass every check with all `19` identities matching.
  The `.98` focused suite passes `27` tests.
- The authoritative `.98` CPU suite passes:
  `1259 passed, 12 skipped, 2 warnings in 157.03 s`.
- `.98 --execute` rejects before Python or Isaac with code `4`; the namespace
  remains absent.
- No authorization token, runtime, teacher admission, label capture, dataset,
  BC, PPO, or training was created. Goal completion remains `6/10`.

## Next round after Round 234

- The case-8 validation pair is separately reviewable for exactly one future
  held-out paired canary. No runtime authorization has been issued.
- Continue CPU-only preparation with case 16, the second selected validation
  case, unless a separate case-23 v4 CPU conversion or case-8 runtime
  authorization is explicitly issued.

## Round 235: case-16 readiness requires a structural natural-error design

- Bound the second selected validation case to the exact v12 explicit-preview
  plan and exclusive zero-residual Gate C result. Their SHA-256 identities
  match the validation selection.
- Preserved `896` exact source anchors and `895` transitions with separate
  `17.548706 s` source and `26.028630 s` execution clocks.
- The zero-residual dynamic gate passes at `0.080600/0.081492 m` position
  p95/max and `6.030922 deg` peak pitch.
- Unlike case 8, case 16 reaches the frozen base-linear, yaw, and proxy-rate
  limits. Lever-arm correction saturation is `0.958736`, and no low-motion
  interval lasts `0.10 s`.
- Marked `safe_window_absent_requires_structural_profile=true`. Reusing the
  case-8 external-wrench pulse is not suitable; the next design must use
  natural tracking error with deterministic safety projection.
- Implementation commit
  `7957e34871b2da9f84a555bcd46576048fbee658` passes `20` focused tests
  on macOS and `.98`, with byte-identical evidence.
- The authoritative `.98` CPU suite passes:
  `1265 passed, 12 skipped, 2 warnings in 161.84 s`.
- No token, runtime, capture, conversion, dataset, BC, PPO, or training was
  created. Goal completion remains `6/10`.

## Next round after Round 235

- Design a CPU-only case-16 structural natural-error corrective profile with
  no external wrench. Prove projection, command-limit, clock, and held-out
  validation invariants before implementing any runtime route.

## Round 236: case-16 structural natural-error validation profile is ready

- Built a case-specific held-out validation profile from case 16's own
  zero-residual gate rather than reusing a train-case or case-8 profile.
- Because the plan has no low-motion pulse window and already reaches the
  base-linear, yaw, and proxy-rate limits, no external wrench profile was
  created. Existing tracking error supplies the bounded excitation.
- The formula retains the smaller of `40%` and the p95 gate-margin fraction.
  Case 16 resolves to `40%`, with maximum residuals
  `[0.004255,0.007047,0.001022]` and a `0.40 s` slew horizon.
- Natural error exceeds `0.03 m` in `52/54` trace samples. The full-plan safety
  projection is contractive: negative projection counts are `[0,20,0]` and
  positive counts are `[607,174,0]`.
- Only effective post-supervisor residuals may be assessed. Requested
  residuals are not training labels, and held-out validation cannot admit a
  teacher, label archive, dataset, merge, BC, PPO, or training run.
- Implementation commit
  `77139d631ee05b3432d368ce478ff8f8af7bca93` passes the focused `19`-test
  suite on macOS and `.98`; Windows regenerates byte-identical artifacts.
- The authoritative `.98` CPU suite passes:
  `1278 passed, 12 skipped, 2 warnings in 155.25 s`.
- No token, namespace, Isaac/GPU run, capture, conversion, dataset, BC, PPO,
  or training was created. Goal completion remains `6/10`.

## Next round after Round 236

- Implement a disabled-by-default, hash-bound case-16 natural-error validation
  pair contract and CPU preflight without issuing authorization or creating a
  runtime namespace.
- The nearest actual data-path action remains exactly one separately
  authorized case-23 v4 CPU conversion. The consumed case-23 capture
  authorization cannot be replayed.

## Round 237: case-16 natural-error validation route is complete but unauthorized

- Added a dedicated validation-only natural-error pair contract, builder,
  validator, playback adapter, wrapper, and finalizer at implementation commit
  `fcec29fdd2d72ea4defe129a11cf9e089a7c0a57`.
- Bound the route to case `16`, validation split, the reviewed controller
  parent `c92c428785be987ab13e558aa07abc2713a7a0c5`, and `24` exact source,
  plan, controller, profile, robot, and route-code identities.
- Preserved baseline-first execution, configuration seed `20260716`, reset
  seed `20260732`, identical plan and physics, and the existing dynamic gates.
  External wrench, capture, dataset, teacher admission, BC, PPO, and training
  paths are forbidden.
- The candidate is unreachable unless the baseline dynamically passes and
  releases the GPU. The paired finalizer requires nonzero bounded effective
  projection telemetry and at least `0.003 m` and `2%` p95 improvement without
  exceeding any regression allowance.
- Mac and `.98` preflights pass every check. The `.98` focused route/profile
  suite passes `24` tests, and the authoritative `.98` CPU suite passes
  `1289 passed, 12 skipped, 2 warnings in 171.93 s`.
- An explicit `.98 --execute` attempt fails closed with exit code `4` before
  Python or Isaac. No runtime namespace was created.
- The repeated case-23 v4 capture request was not replayed because that
  one-shot authorization was already consumed by the sealed successful
  capture. No token, Isaac/GPU run, label, dataset, BC, PPO, or training was
  created. Goal completion remains `6/10`.

## Next round after Round 237

- The case-16 validation natural-error pair is separately reviewable for one
  future held-out paired canary. No runtime authorization has been issued.
- The nearest real-data action remains exactly one separately authorized
  case-23 v4 CPU conversion:
  `Authorize exactly one case-23 v4 CPU conversion.`
- Do not replay the consumed case-23 v4 corrective-label capture authorization.

## Round 238: the six pending routes are fresh, ordered, and closed

- Added a deterministic CPU audit for the complete pending corrective-data
  queue at implementation commit
  `3a8c6aa15b480f1e45354c2c95aa4beb7333e22f`.
- Regenerated all six route preflights on `.98` at clean synchronized
  `HEAD == upstream`. Every route passes all contract/check groups and all
  `107` pinned identities.
- Fixed the queue order to case-23 v4 CPU conversion, train paired canaries
  `6/2/7`, then held-out validation pairs `8/16`. This prevents later runtime
  work from bypassing the already accepted but unconverted case-23 capture.
- Verified all six reserved namespaces are absent. Runtime, GPU, label
  capture, dataset conversion/merge, BC, PPO, and training authorization
  remain false for the aggregate and each route.
- Focused macOS and `.98` suites pass `25` tests. Aggregate summary SHA-256 is
  `b516f8e2f0fe8bc7f21a2837c9d0a56387990a2c1db59247f60a62d0dbe65488`.
- The authoritative `.98` CPU suite passes:
  `1297 passed, 12 skipped, 2 warnings in 172.43 s`.
- The repeated case-23 v4 corrective-label capture was not replayed. No token,
  runtime namespace, Isaac/GPU workload, capture, conversion, dataset, BC,
  PPO, or training was created. Goal completion remains `6/10`.

## Next round after Round 238

- The queue is ready for review, not execution. The exact next action remains:
  `Authorize exactly one case-23 v4 CPU conversion.`
- After a passing conversion and independent reopen, case 23 can become the
  second train dataset. Cases 6 and 2 remain the next minimum-tranche paired
  canaries; case 7 is the additional train candidate; cases 8 and 16 remain
  held-out validation only.

## Round 239: the final residual BC architecture is explicit and fail-closed

- Audited the projection-aware BC runner against the real case-30 converted
  dataset and the case-23 v4 capture. Their observation matrices are
  `[11411,65]` and `[3273,65]`; both use the same ordered observation names.
- Corrected the active goal ledger: `26` is only the physical/base-state block.
  The admitted policy input is `65 = 26 + 3 * 13`, with three bounded residual
  outputs for base linear velocity, base yaw rate, and riser target.
- Pinned architecture
  `model_based_shared_encoder_zero_initialized_residual_v1`: state encoder
  `[128,128]`, shared lookahead encoder `[64,64]`, fusion `[256,128]`, and
  `142019` parameters.
- The projection-aware runner now initializes the residual action head to exact
  zero before optimization. Eager and TorchScript paths both emit bit-exact
  zero for nonzero synthetic observations before training.
- The checked-in admission template binds architecture, dimensions, zero-head
  initialization, and current trainer/contract hashes while remaining
  unusable: dataset/commit identities are null and all learning authorization
  flags remain false.
- Implementation commit
  `282c5998ec0c91982d2a1f610b18db7acc5f4e1b` and real-data contract test commit
  `8063f785ca683ac41686b06ea53fc3afb69a431a` are pushed and synchronized to
  `.98`.
- The focused `.98` CPU suite passes:
  `69 passed, 2 warnings in 27.87 s`.
- The authoritative `.98` CPU suite passes:
  `1300 passed, 12 skipped, 2 warnings in 170.69 s`.
- No token, runtime namespace, Isaac/GPU workload, capture, conversion, dataset
  merge, BC, PPO, checkpoint, or training run was created. Goal completion
  remains `6/10`.

## Next round after Round 239

- The nearest real-data operation remains separately authorized case-23 v4 CPU
  conversion. The exact authorization phrase is:
  `Authorize exactly one case-23 v4 CPU conversion.`

## Round 240: completion audit v2 binds the real pre-training state

- Upgraded the goal completion auditor to schema
  `cinebotrl_two_wheel_riser_goal_completion_audit_v2`.
- The auditor now compares the active goal ledger with the executable BC
  contract instead of accepting the ledger alone. It proves the admitted
  architecture is exact-zero `65 -> 3` with a `26 + 3 * 13` observation
  decomposition.
- The report makes the data gap explicit: one converted corrective case
  dataset exists, zero multi-case corpus cases exist, and the minimum corpus
  remains four train plus two disjoint validation cases.
- It records case 23 as the next operation and keeps conversion, runtime, BC,
  PPO, and training authorization false. Architecture readiness therefore does
  not imply readiness for BC execution.
- The v2 report was generated independently on macOS and `.98` from clean
  synchronized implementation commit
  `a3185e211f300cb02c784973931f1b502b5f31cb`. The files are byte-identical with
  SHA-256
  `507e5a0f7161f89610a551ef7e3fa7493744f9d00f816c42663005e41fa85157`.
- The focused `.98` CPU suite passes:
  `21 passed, 2 warnings in 9.93 s`.
- The authoritative `.98` CPU suite passes:
  `1301 passed, 12 skipped, 2 warnings in 173.12 s`.
- Completion remains honestly `6/10`. No token, runtime namespace, Isaac/GPU
  work, capture, conversion, dataset merge, BC, PPO, checkpoint, or training
  run was created.

## Next round after Round 240

- The nearest real-data operation is unchanged:
  `Authorize exactly one case-23 v4 CPU conversion.`

## Round 241: learned-policy artifacts fail closed before Isaac

- Audited the validation, holdout, all-79, and rendered-rollout admissions. They
  previously bound only the TorchScript file hash, so a corrupt or
  architecture-incompatible artifact could survive admission and fail only
  after the runtime route started.
- Added a CPU-only structural inspector that requires a loadable TorchScript
  policy with `65` observation channels, `26 + 3 * 13` state/lookahead
  structure, three bounded outputs, finite positive normalization, and exactly
  `142019` parameters.
- Validation, holdout, all-79, and render contracts now run this inspection and
  hash-bind the inspector module as an explicit code dependency.
- The all-79 admission now revalidates the complete BC execution report and its
  admission chain rather than trusting only selected result fields.
- Repaired a pre-existing `.98` execution defect: WSL system Python has neither
  NumPy nor PyTorch. Learned preflights now run in the Windows Isaac Python,
  convert every shared artifact path with `wslpath`, and route repository
  checks through `wsl.exe git` using an explicit `WSLENV` bridge.
- Live `.98` CPU evidence at commit
  `49296494a898b07f477f145dffde6f126ee64714` proves the valid policy loads with
  the exact structure, WSL git reports the synchronized commit, and malformed
  TorchScript is rejected.
- The focused `.98` suite passes:
  `65 passed, 2 warnings in 7.74 s`.
- The authoritative `.98` suite passes:
  `1309 passed, 12 skipped, 2 warnings in 175.46 s`.
- No runtime namespace, Isaac/GPU work, capture, conversion, dataset merge, BC,
  PPO, checkpoint training, or learned rollout was started. Completion remains
  `6/10`.

## Next round after Round 241

- The nearest real-data operation remains:
  `Authorize exactly one case-23 v4 CPU conversion.`

## Round 242: runtime action recurrence now matches corrective training data

- Audited the complete offline-to-runtime `65 -> 3` residual-policy contract.
  Corrective conversion rebuilds each observation's previous-action channels
  from the prior effective post-supervisor label, while runtime previously
  carried the requested pre-projection action.
- Added a projection-aware helper that returns the unchanged safety-clamped
  high-level command and the normalized residual actually realized after
  supervisor limits. Runtime now carries that effective action into the next
  observation under contract
  `previous_effective_post_supervisor_action_v1`.
- Kept requested network output separate from effective action and added
  requested/effective/projection-delta telemetry. The model-based planner,
  residual scales, safety limits, and final command path are unchanged.
- Added fail-closed coverage for unprojected parity, per-axis safety
  projection, invalid planner bases, recurrence ordering, and corrective
  dataset contract reuse.
- Implementation commit
  `22203a348bddfa4d8550ca64b9bbd6f0edc65e07` is pushed and synchronized to
  `.98`.
- Resealed every active tokenless route that pins the shared playback or
  residual modules at commit
  `be5cd3ca12751db6b23308691d5e8040731e721c`. No archived case-23 capture
  evidence was changed.
- Split historical executed/preflight identities from current active contract
  identities in the goal ledger at commit
  `09580eedbd4cf1013cb2737bc494449f978606d2`, preventing a current reseal from
  overwriting historical provenance.
- Focused CPU contract coverage passes `103` tests. The authoritative `.98`
  Windows-Isaac Python suite passes:
  `1313 passed, 12 skipped, 2 warnings in 172.00 s`.
- No token, runtime namespace, Isaac/GPU workload, capture, conversion, corpus
  merge, BC, PPO, checkpoint, or training run was created. Goal completion
  remains `6/10`.

## Next round after Round 242

- The case-23 v4 capture authorization is consumed and must not be replayed.
- The nearest real-data operation remains separately authorized CPU-only
  conversion:
  `Authorize exactly one case-23 v4 CPU conversion.`

## Round 243: current-code conversion and corrective queue admissions refreshed

- Regenerated the case-23 CPU-conversion no-token preflight and all five
  pending paired-canary preflights on `.98` from clean synchronized code.
- The first live sweep correctly found one transitive case-7 drift: its active
  route contract pinned the historical profile-proposal hash even though the
  proposal had been resealed after the residual-runtime update.
- Updated only the case-7 proposal SHA/blob identity and added a committed
  contract test that opens every pinned identity. The test computes Git blob
  IDs directly, so it works under both macOS and Windows Python.
- Repair commit
  `cec3cdb20904bbe37d5214e859759ff13a7de84a` is pushed and synchronized.
  Re-running all six no-token preflights at that commit produced six passes:
  case-23 conversion; train cases 6, 2, and 7; validation cases 8 and 16.
- Preserved the exact live admissions under
  `evidence_20260724_pending_corrective_route_queue_cpu_v2`. Its summary
  SHA-256 is
  `ff5f54f22877ec653da3aa933a4cc3eaceb083f4a0c1135a44586ed8353bae13`.
- The case-23 conversion preflight is also preserved independently under
  `evidence_20260724_case23_corrective_conversion_execution_cpu_v3`, SHA-256
  `0a7719f26849f2d08ffa8471b670c9b63068015c709541ca15ad22554e8a094a`.
  It proves CPU readiness while conversion authorization, output creation,
  corpus merge, BC, PPO, and training remain false.
- Evidence commit
  `049094a61a814fd5c2c4a31cf164f60478239f1d` and goal-binding commit
  `e190d8820303ebc7ef9185be56c66f561296d0fe` are pushed and synchronized.
- Focused CPU coverage passes `28` tests. The authoritative `.98`
  Windows-Isaac Python suite passes:
  `1315 passed, 12 skipped, 2 warnings in 175.14 s`.
- No token, runtime namespace, Isaac/GPU workload, capture, conversion,
  dataset output, corpus merge, BC, PPO, checkpoint, or training run was
  created. Goal completion remains `6/10`.

## Next round after Round 243

- The exact next data-producing operation remains:
  `Authorize exactly one case-23 v4 CPU conversion.`
- A future authorized wrapper invocation must recompute admission against its
  then-current clean synchronized `HEAD`; the preserved v2 queue is evidence,
  not a reusable authorization.

## Round 244: riser mass and drive recommendation is traceable end to end

- Added a CPU-only cross-layer auditor that derives the moving assembly from
  the current URDF `riser_joint` subtree instead of trusting a copied mass.
  It distinguishes the `28.000 kg` whole-robot balance plant from the
  `4.342 kg` modeled moving assembly, the conservative `8.000 kg` drive-sizing
  mass, and the `14.803715 kg` calculated ceiling at the required `15%` force
  margin.
- Recomputed the selected 48 V/750 W, `3:1`, `70 mm/rev` candidate against the
  existing conservative force model. The 8 kg emergency design force is
  `276.96 N`, continuous equivalent linear force is `550.258929 N`, and the
  resulting force margin is `1.986781`.
- The sizing calculation explicitly receives no counterbalance credit. It also
  records the `98.176 J` single-descent mechanical-energy bound for regeneration
  review, the `0.12 m` full-speed stopping distance, and the `1.50 m`
  recommended mechanical stroke while preserving the `0.60--1.80 m` camera
  height contract.
- Added fail-closed checks that reject treating all 28 kg as vertical payload,
  extending the camera ceiling to 1.9 m, silently enabling the 750 W simulation
  profile, or interpreting the empty bench template as physical qualification.
- The active Isaac profile remains the 400 W engineering profile. The 750 W
  unit remains only a supplier/bench design-review candidate; production
  procurement, hardware transfer, runtime, capture, BC, PPO, and training stay
  unauthorized.
- Implementation commit
  `71095494049c585df7afcede4718a030458ca261` is pushed and synchronized to
  `.98`. The machine-readable summary SHA-256 is
  `52ddff232c339d5cd3057cf680a98ce19939150943d9c45e21f14836d28c507a`.
- Focused macOS and `.98` suites both pass `24` tests. The authoritative `.98`
  Windows-Isaac Python suite passes:
  `1321 passed, 12 skipped, 2 warnings in 173.36 s`.
- Operational lesson: a long Windows pytest launched through an unlogged SSH
  pipe can outlive the client stream. One diagnostic retry briefly created a
  duplicate; the later duplicate was stopped, no unrelated process was
  touched, and the authoritative suite was rerun alone with an explicit log
  and exit-code file. Future long `.98` CPU suites should use that logged
  pattern from the outset.
- No Isaac/GPU workload, runtime namespace, label capture, CPU conversion,
  corpus merge, BC, PPO, checkpoint, or training run was created. Goal
  completion remains `6/10`.

## Next round after Round 244

- The physical riser recommendation is now calculation-traceable but still
  awaits supplier and bench measurements before hardware transfer.
- The exact next data-producing operation remains separately authorized:
  `Authorize exactly one case-23 v4 CPU conversion.`

## Round 245: BC promotion now tests the runtime action recurrence

- Audited the projection-aware `65 -> 3` BC path after the runtime previous
  action contract was corrected. Training and validation still consumed
  teacher-forced previous effective actions, so a model could pass the offline
  gate without proving stability when its own projected effective action is
  fed into the next observation.
- Added deterministic case-reset recursive validation under contract
  `case_reset_recursive_effective_action_validation_v1`. For each disjoint
  validation case, the evaluator starts with zero previous action, predicts a
  requested residual, passes it through the unchanged safety projection, and
  writes the resulting effective residual into the next observation's previous
  action channels.
- The recursive evaluator separately records requested action magnitude,
  projected action magnitude, projection-clipped rows, requested rate maxima,
  slew violations, row/case/reset counts, and case-balanced effective-action
  MSE against the zero-residual planner baseline.
- Artifact emission now requires both teacher-forced and recursive validation
  to improve every action channel by at least the admitted fraction, keep
  requested normalized magnitude below `0.95`, and produce zero request-slew
  violations. A teacher-forced-only pass is explicitly rejected.
- Recursive replay is limited to the held-out validation split. It does not
  serialize the full training corpus through one-row GPU inference and does not
  open the reserved holdout split.
- Bumped the validation contract to
  `projected_effective_action_case_balanced_recursive_validation_v2` and the BC
  execution report to
  `cinebotrl_two_wheel_riser_model_based_corrective_bc_execution_report_v2`.
  The unusable admission template was resealed with current code hashes while
  dataset, commit, split, approval, and authorization fields remain empty or
  false.
- Implementation commit
  `8ee358e045a8099384dae2556e66093f86d1aa05` and goal-binding commit
  `c995bea4718dbae47c3f02054952f45eb060f667` are pushed and synchronized to
  `.98`.
- Focused `.98` coverage passes `67` tests. The final authoritative `.98`
  Windows-Isaac Python suite passes:
  `1324 passed, 12 skipped, 2 warnings in 171.00 s`.
- This closes an offline promotion false-positive, not the data gap. No
  conversion, corpus merge, BC execution, checkpoint, learned rollout, Isaac,
  GPU workload, PPO, or training run was created. Goal completion remains
  `6/10`.

## Next round after Round 245

- The exact next data-producing operation remains separately authorized:
  `Authorize exactly one case-23 v4 CPU conversion.`
- After conversion, the route must be reopened against then-current hashes
  before cases 6/2/7 and validation cases 8/16 proceed.

## Round 246: learned-policy promotion is explicitly balance first

- Audited the validation, reserved-holdout, all-79, render, and final goal
  admission path against the goal priority `self-balance > tracking`.
- The playback already failed on termination, excessive pitch, wheel/riser/
  proxy saturation, thermal force/load, and missing controller evidence, but
  the learned gate retained only generic `passed` booleans plus tracking
  regression metrics. This made the final evidence unable to independently
  prove why a rollout was safe.
- Added `balance_first_rollout_safety_v1` snapshots for every model-based
  baseline, learned, and zero-policy rollout. Each snapshot records both
  payload/result dynamic and thermal outcomes, reference completion,
  termination absence, pitch, all saturation ratios, thermal load, peak-force
  violations, and the matching runtime check booleans.
- Model-based validation, holdout, and all-79 reports are now v2 schemas.
  Promotion fails closed if any safety field is missing, non-finite, above its
  bound, inconsistent with the runtime checks, or if tracking improves while
  balance safety regresses.
- The downstream render admission validates safety snapshots for all 79 rows,
  not only the six representative render cases. The completion auditor also
  recomputes the v2 safety contract before accepting a learned all-79 report.
- The wrappers pin the same `12 deg` pitch and `0.20` saturation limits for
  playback and report generation. No source plan, controller command,
  trajectory, residual scale, or physics setting changed.
- Implementation commit
  `5148ab60396eb62f4204b452670d3d95ce793546` and goal-binding commit
  `18a627ed0345074958c4b6601118029be2178c75` are pushed and synchronized.
- The first full `.98` run intentionally exposed one stale goal-ledger template
  hash (`1329 passed`, one failed). After rebinding the changed contract
  identities, focused `.98` coverage passed `62` tests and the final
  authoritative Windows-Isaac Python CPU suite passed:
  `1330 passed, 12 skipped, 2 warnings in 171.96 s`.
- No Isaac/GPU workload, runtime namespace, conversion, capture, corpus merge,
  BC, checkpoint, PPO, or training run was created. Goal completion remains
  `6/10`.

## Next round after Round 246

- The exact next data-producing operation remains separately authorized:
  `Authorize exactly one case-23 v4 CPU conversion.`
- After the real case-disjoint corpus exists, every learned policy must pass
  recursive offline validation followed by the new balance-first dynamic
  validation, holdout, all-79, and render chain.

## Round 247: the pending corrective queue is refreshed without conversion

- Re-ran the case-23 conversion no-token preflight and all five pending paired
  route preflights on `.98` from clean synchronized commit
  `a8a7533642694dfb05c7a999803ebd95fed456fc`.
- All six routes pass in the required order: case-23 conversion; train cases
  6, 2, and 7; validation cases 8 and 16. Their complete set of `107`
  SHA/blob identities is unchanged from the admitted v2 queue; only clean
  commit context advanced.
- Preserved the refreshed queue under
  `evidence_20260724_pending_corrective_route_queue_cpu_v3`. Its summary
  SHA-256 is
  `ef32652ac98d1103d431f7b3ae96d63c3311e3a7fba0b9a92061a1e1b16892cd`.
- Preserved the independent case-23 no-token preflight under
  `evidence_20260724_case23_corrective_conversion_execution_cpu_v4`. Its
  summary SHA-256 is
  `b9ca26d3aed077c216c785cc611d4e63d6708bd66050bdd07d6a57a40dd565c3`.
  CPU contract readiness is true while all authorization checks remain false,
  as required when no one-time token exists.
- Evidence/test commit
  `c094d0d9c63cba589bb04f5ede078ba2693fb219` and goal-binding commit
  `c6dd84b9cab94fefb2a215d2256c343b40ff72fa` are pushed and synchronized.
- Focused macOS coverage passes
  `30 passed, 2 warnings in 0.62 s`; focused `.98` Windows coverage passes
  `30 passed, 2 warnings in 2.56 s`. The authoritative `.98` Windows-Isaac
  Python CPU suite passes:
  `1331 passed, 12 skipped, 2 warnings in 177.86 s`.
- No authorization token, CPU conversion output, runtime namespace,
  Isaac/GPU workload, capture, corpus merge, BC, checkpoint, PPO, or training
  run was created. Goal completion remains `6/10`.

## Next round after Round 247

- The exact next data-producing operation remains separately authorized:
  `Authorize exactly one case-23 v4 CPU conversion.`
- The preserved queue is readiness evidence only. Any authorized conversion
  must reopen the fail-closed wrapper against its then-current clean,
  synchronized `HEAD`; conversion does not itself authorize merge or
  training.

## Round 248: the end-goal audit now proves pending-route readiness

- Upgraded the CPU-only end-goal auditor to
  `cinebotrl_two_wheel_riser_goal_completion_audit_v3`. It now loads the
  preserved pending corrective queue as a first-class input rather than
  reporting only a hardcoded next case.
- Pre-training readiness now fails closed unless the queue summary hash matches
  the recursive goal ledger, all six routes appear in the exact
  train-before-validation order, all `107` route identities are present, every
  queue check passes, and runtime/conversion/merge/BC/PPO/training
  authorization remains closed.
- The completion count remains honest: six foundation gates pass and four
  learning gates remain unresolved. Queue readiness is not counted as a
  completed learning gate and does not make the project training-ready.
- Auditor/test implementation commit
  `d845d029d12e33cc29543e4cad8627d5b2266f01` and evidence/goal-binding commit
  `4ee77623d64091af572cd21cf6ab255f0a23c20c` are pushed and synchronized.
- Preserved the report under
  `evidence_20260724_riser_goal_completion_audit_v3`; summary SHA-256 is
  `c27491ba9392396d76dc7797231186220dde75f6676c44c12df44ec6666651af`.
  macOS and `.98` generated byte-identical reports.
- Auditor-only coverage passes `14` tests on both hosts. Combined goal/auditor
  coverage passes `20 passed, 2 warnings in 3.17 s` on macOS and
  `20 passed, 2 warnings in 9.80 s` on `.98`. The authoritative `.98`
  Windows-Isaac Python CPU suite passes:
  `1332 passed, 12 skipped, 2 warnings in 173.50 s`.
- No authorization token, CPU conversion output, runtime namespace,
  Isaac/GPU workload, capture, corpus merge, BC, checkpoint, PPO, or training
  run was created. Goal completion remains `6/10`.

## Next round after Round 248

- The exact next data-producing operation remains separately authorized:
  `Authorize exactly one case-23 v4 CPU conversion.`
- After conversion, reopen the corpus intake and the remaining case
  6/2/7 plus validation 8/16 routes against the then-current synchronized
  commit. BC remains closed until the real case-disjoint corpus gate passes.

## Round 249: learned authority is pinned above the inner loops

- Audited the learned action path from policy output through playback,
  validation, holdout, all-79, render, and final goal admission.
- Added the explicit contract
  `frozen_lqr_high_level_residual_control_ownership_v1`. The learned policy may
  output only normalized residual `vx`, `wz`, and riser-target commands. It
  cannot directly command wheel effort or physical gimbal joints.
- Pinned the deterministic owners: the frozen cascaded LQR owns wheel effort,
  the semantic attitude adapter owns gimbal attitude, the command supervisor
  owns riser hard limits, and runtime gates own safety enforcement.
- Runtime evidence and all downstream learned-policy gates now fail closed if
  this ownership record is absent or altered. Validation, holdout, and all-79
  report schemas advanced to v3. No controller command, source plan,
  trajectory, residual scale, physics parameter, or dynamic threshold changed.
- Implementation commit
  `20694e8d5d238d1965a22e3eef4e14fe57682f05`, route-identity reseal commit
  `8c9d1e4b2de2fa0e2007f98004e01b989b6d6883`, queue evidence commit
  `5b269a729a8006a7bec5fd5d9bb6fa594e1e58e7`, and goal-binding commit
  `011b3e2f1c4c460de28318866c80662a91415953` are pushed and synchronized.
- The first authoritative run intentionally exposed eight stale playback or
  residual-dataset identities:
  `1328 passed, 12 skipped, 8 failed`. These were identity-only failures in
  active case-23/2/6/7/8/16 contracts and case-7/8 profile proposals; no
  behavioral test failed. The contracts were resealed with every token and
  authorization field still closed.
- Refreshed all six pending no-token routes on `.98` at clean commit
  `8c9d1e4b2de2fa0e2007f98004e01b989b6d6883`. All `107` current identities
  pass. Queue summary SHA-256 is
  `244377cb46a69d744f26449f74a4fa5301c0416c3142857f8213dfbacd05f041`.
- Preserved the byte-identical macOS/`.98` completion audit v5 at summary
  SHA-256
  `25ce15bb02bd5c43b4619d1b0ddf85f62a72ca4018d23bb587f74c46acd50ed1`.
  It still reports six of ten required gates complete.
- Final focused coverage passes
  `214 passed, 2 warnings in 16.45 s` on macOS and
  `214 passed, 2 warnings in 79.74 s` on `.98`. The authoritative `.98`
  Windows-Isaac Python CPU suite passes:
  `1337 passed, 12 skipped, 2 warnings in 180.63 s`.
- No authorization token, CPU conversion output, runtime namespace,
  Isaac/GPU workload, capture, corpus merge, BC, checkpoint, PPO, or training
  run was created. Goal completion remains `6/10`.

## Next round after Round 249

- The exact next data-producing operation remains separately authorized:
  `Authorize exactly one case-23 v4 CPU conversion.`
- After conversion, reopen corpus intake and then proceed in order through
  train cases 6/2/7 and validation cases 8/16. BC remains closed until the
  real case-disjoint corpus gate passes.

## Round 250: case-23 v4 is converted exactly once

- Consumed one ephemeral mode-`0600` authorization outside the repository at
  clean synchronized runtime commit
  `11fd27698955d277f4b926151bcca0cda2f4b27c`. The token was deleted before
  conversion and no second token or converter invocation was created.
- The converter succeeded once and produced 3,273 rows with 65 observation
  features and three effective post-supervisor residual targets. The dataset
  SHA-256 is
  `ee55db1c02e504e035a47532df8141ab142bb68a2c47b2795b6fd5ad7283ef01`.
  Effective actions, requested-action audit values, non-history observations,
  case IDs, and all three clocks reopen exactly; clipped rows are `[0, 0, 0]`.
- The original wrapper escaped `$NAMESPACE` in its Windows output path. The
  converter therefore wrote to a literal-dollar directory and the first
  finalizer exited `1`. The converter was not retried. Its output was copied
  byte-for-byte into the authorized namespace and only the finalizer was run
  again. The final status passes all provenance and data checks at SHA-256
  `31dc0e6126772fad3958b4759d9c3fe03ee0be7157b6f5b356e08060952a1943`.
- Preserved the complete admission, contract, original failure, finalizer-only
  recovery, canonical NPZ, and checksums under
  `evidence_20260724_case23_corrective_conversion_execution_cpu_v5`.
  The recovery audit SHA-256 is
  `17679574397ff911bfe971d5f3f906f3229b53ee9ee903ecc0edd89421fa3e0f`.
- Fixed the wrapper path construction and added a regression that rejects
  escaped namespace interpolation. The first authoritative post-conversion
  suite correctly exposed three tests that depended on the runtime namespace
  being absent. Their preflight fixtures are now isolated, while production
  still rejects namespace reuse.
- Evidence/fix commit `3052ca41a56a6cc4b5014787c8480aaded531224`,
  goal-binding commit `21686033153670ffecf73ddc073d7f7ec21b690e`,
  and test-isolation commit
  `224af37e3c6d870b0997a5ed67fec9e0096024cd` are pushed and synchronized to
  `.98`.
- Focused coverage passes `29 passed, 2 warnings in 1.01 s` on macOS and
  `29 passed, 2 warnings in 6.56 s` on `.98`. The final authoritative `.98`
  Windows-Isaac Python CPU suite passes:
  `1337 passed, 12 skipped, 2 warnings in 175.33 s`.
- The converted case is valid only for a later reviewed case-merge operation.
  No corpus merge, Isaac/GPU workload, capture, BC, PPO, checkpoint, or
  training run was created. Goal completion remains `6/10`.

## Next round after Round 250

- Reopen the CPU-only corrective corpus-intake audit against the sealed
  case-23 v4 conversion and the existing case-30 conversion. Do not merge a
  corpus during that audit.
- Continue the pending paired routes in order: train cases 6, 2, and 7, then
  validation cases 8 and 16. BC remains closed until the case-disjoint corpus
  gate passes and receives separate authorization.

## Round 251: corpus intake now binds converted cases 23 and 30

- Upgraded the CPU-only corpus-intake audit to schema
  `cinebotrl_two_wheel_riser_model_based_corrective_corpus_intake_v2`.
  The default route now reopens the real case-23 dataset, conversion final
  status, and path-recovery audit in addition to the existing case-30 dataset.
- Case-23 intake fails closed unless the recovery evidence proves one converter
  invocation, no converter retry, byte identity with the canonical NPZ, a
  matching final-status hash, and closed merge/BC/PPO/training state.
- Converted train cases are `[23, 30]`. The minimum train tranche still lacks
  cases `6` and `2`; validation still lacks cases `8` and `16`. Case `7`
  remains an additional train candidate. The corpus manifest is not ready.
- Preserved the byte-identical macOS/`.98` report under
  `evidence_20260724_model_based_corrective_corpus_intake_v2`. Its summary
  SHA-256 is
  `dc5ee5f97882c0551521703267d9897ce79731845cbd73e6883bc121183bcb62`.
- Implementation commit
  `4f370b4b6cc71a338386cc6f760a42b7e32ff085` and evidence/goal-binding
  commit `cd83adff011165221ac0fd4f260c2f4eee15c6b0` are pushed and synchronized
  to `.98`.
- Focused intake coverage passes `12 passed, 2 warnings in 0.85 s` on macOS
  and `12 passed, 2 warnings in 1.27 s` on `.98`. The authoritative `.98`
  Windows-Isaac Python CPU suite passes:
  `1338 passed, 12 skipped, 2 warnings in 181.34 s`.
- No dataset merge, new conversion, Isaac/GPU workload, capture, BC, PPO,
  checkpoint, or training run was created. Goal completion remains `6/10`.

## Next round after Round 251

- The next data-producing operation requires separate authorization:
  `Authorize exactly one case-6 paired canary.`
- If case 6 passes its paired dynamic gate, review and authorize its corrective
  capture separately. Do not skip directly to conversion, merge, or BC.

## Round 252: case-6 paired corrective target passes

- Consumed exactly one external mode-`0600` authorization and executed the
  pinned case-6 paired wrapper once at clean synchronized runtime commit
  `1a331133832bdf45cf4d8035f7b411e08c664522`. The token was removed before
  Isaac started and was not recreated.
- Baseline and candidate used the same case-6 plan, source/execution clocks,
  reset seed, physics, controller settings, and deterministic 20-step `20 N`
  body-x wrench. Both dynamic-quality gates passed without termination.
- The candidate reduced camera-position p95 error from `0.118123281 m` to
  `0.110559691 m`: `0.007563591 m` absolute and `6.4031%` relative
  improvement. Position max also improved from `0.127080200 m` to
  `0.126411242 m`; attitude max, pitch max, riser error, and saturation did
  not regress.
- The paired admission passed every check. Candidate normalized residual
  maxima were `[0.462029300, 0.159056047, 0.089325784]`, within the frozen
  action envelope.
- Preserved admission, contract, both raw rollout JSONs, heartbeats, logs,
  exit codes, finalizer output, and checksums under
  `evidence_20260724_case6_corrective_pair_execution_v2`. The final-status
  SHA-256 is
  `c723407d40072b0e0753036b65f66e64963268d5f2d01934a2b4a13d1aa96458`.
- Authorization implementation commit
  `f83041e3ee609a0a2bd9df96d0ac4033835a9479`, cross-platform test fix commit
  `1a331133832bdf45cf4d8035f7b411e08c664522`, evidence/status commit
  `0eb88c8aae225d9f4280ee76398279003910d1a6`, and evidence-mode commit
  `8109c87636736c868a89cd61e3b23a076d899510` are pushed and synchronized.
- Focused coverage passes `24 passed, 2 warnings in 4.04 s` on macOS and
  `24 passed, 2 warnings in 12.27 s` on `.98`. The authoritative `.98`
  Windows-Isaac Python CPU suite passes:
  `1339 passed, 12 skipped, 2 warnings in 178.86 s`.
- No label capture, dataset, conversion, merge, BC, PPO, checkpoint, or
  training output was created. GPU ownership was released and the case-6
  corrective target is admitted only for a separately authorized capture.

## Next round after Round 252

- The next data-producing operation requires separate authorization:
  `Authorize exactly one case-6 corrective-label capture.`
- Do not reuse the consumed paired-canary authorization and do not proceed
  directly to conversion, corpus merge, BC, PPO, or case 2.

## Round 253: case-6 corrective labels are captured exactly once

- Added a fresh, tokenless, fail-closed case-6 capture route bound to the
  sealed passing pair, exact case-6 plan/profile/wrench, current 400 W drive
  profile, LQR gains, robot assets, playback, capture runtime, and finalizer.
  The only generic change allows the existing capture validator to receive the
  expected case-specific corrective-profile envelope instead of assuming the
  case-30 envelope.
- Route commit `c528cd2a3ccfabcb38fffd99a5c1613d306161e9` is pushed and synchronized.
  Focused route coverage passed `27 passed, 2 warnings` on macOS and
  `26 passed, 1 skipped, 2 warnings` on `.98`; the real tokenless WSL
  preflight passed with all identities and authorization states closed.
- Consumed exactly one external mode-`0600` authorization and executed the
  case-6 capture wrapper once. The token was removed before Isaac started;
  there was no retry.
- The rollout passed dynamic, thermal, controller, perturbation, heartbeat,
  archive, clock, identity, and GPU-release checks. Position p95/max was
  `0.110559691/0.126411242 m`, attitude p95/max was
  `0.158318448/0.453032403 degrees`, and peak pitch was
  `6.383016685 degrees`.
- Preserved 7,933 aligned samples. Source/execution clocks end at
  `15.942736/17.737274606 s`, exactly 20 rows carry the deterministic wrench,
  and initialization contributes no capture samples. Requested/effective
  normalized maxima are `[0.462029308, 0.159056053, 0.089325786]`.
- The safety supervisor changed yaw residuals on 146 rows; requested intent,
  effective post-supervisor targets, clipping flags, amplitude/slew flags, and
  both clocks are retained. Future conversion must use only effective
  post-supervisor residuals as labels.
- Preserved all evidence under
  `evidence_20260724_case6_corrective_capture_v1`. Capture SHA-256 is
  `c51411a9686909c47af7eeabf46a61672d8b09432cfb35daabc46af5a5913f85`;
  final-status SHA-256 is
  `843981c82609d8d07cf1b532ce5978e279872649f4f7cd499092a0c7261376f9`.
- Evidence/status commit
  `96f20f6b5eff85fcba4c6ebebd2d00f784ee5f00` is pushed and synchronized.
  Final focused coverage passes `33 passed, 2 warnings in 2.42 s` on macOS
  and `32 passed, 1 skipped, 2 warnings in 6.56 s` on `.98`. The
  authoritative `.98` Windows-Isaac Python CPU suite passes:
  `1345 passed, 12 skipped, 2 warnings in 180.40 s`.
- The raw capture is admitted only for a separately authorized CPU conversion.
  No normalized dataset, corpus merge, BC, PPO, checkpoint, or training output
  was created. Goal completion remains `6/10`.

## Next round after Round 253

- The next data-producing operation requires separate authorization:
  `Authorize exactly one case-6 CPU conversion.`
- Conversion must reopen the sealed capture and final status, preserve all
  clocks and provenance, use effective post-supervisor labels, reconstruct
  previous-action channels, and remain separate from corpus merge or training.

## Round 254: case-6 corrective labels are converted exactly once

- Added a dedicated tokenless case-6 conversion route bound to the sealed
  capture/final status, generic converter and dataset modules, wrapper,
  finalizer, fresh namespace, clean `HEAD == upstream`, and reviewed parent
  `18c000aa23316dfa98eeb25348c5cadd06d59f27`.
- Route commits
  `fd395eb63e698499920fe9b11865586652d0da77` and
  `1283e1b0da405653564382d0aa47d767ce2f925b` are pushed and synchronized.
  The focused suite passes `20 passed, 2 warnings` on macOS and `.98`.
  The authoritative `.98` Windows-Isaac Python CPU suite passes
  `1356 passed, 12 skipped, 2 warnings in 204.52 s`.
- The real tokenless wrapper preflight passed every repository, identity,
  source, conversion, previous-action, clock, and closed-learning check.
  No output namespace existed before authorization.
- Consumed exactly one external mode-`0600` authorization at clean synchronized
  runtime commit `1283e1b0da405653564382d0aa47d767ce2f925b`. The token was
  deleted before conversion, and the converter was invoked exactly once.
- The converter and finalizer both exited `0`. The finalizer reopened 7,933
  rows with 65 observation features and three effective post-supervisor
  residual targets. Effective labels, requested-action audit values,
  non-history observations, case IDs, and source/execution/elapsed clocks are
  exact. Previous-action channels contain zero initially and the prior
  effective action thereafter; clipped rows remain `[0, 146, 0]`.
- Dataset SHA-256 is
  `ac138c9790eda983643ae17cc5b3dcf33cfe4634841760aada929df367acb809`.
  Final-status SHA-256 is
  `5a8662dbb883ae084c8ef8c3a174d6aa8c166b7f27bbae9122109607ba9e2a02`.
  Evidence is preserved under
  `evidence_20260724_case6_corrective_conversion_execution_cpu_v1` at commit
  `63545b6de0cc3e7b7148f6704bd69923c8cca1b3`.
- Case 6 is eligible only for a later reviewed case-merge operation. No corpus
  merge, Isaac/GPU workload, capture, BC, PPO, checkpoint, or training run was
  created. Goal completion remains `6/10`.

## Next round after Round 254

- Reopen the CPU-only corrective corpus-intake audit against converted train
  cases 6, 23, and 30. This audit must not merge a corpus or authorize BC.
- If intake passes, the next data-producing operation requires separate
  authorization: `Authorize exactly one case-2 paired canary.`

## Round 255: corpus intake binds converted cases 6, 23, and 30

- Upgraded the CPU-only intake audit to schema
  `cinebotrl_two_wheel_riser_model_based_corrective_corpus_intake_v3`.
  It now reopens the real case-6 dataset, final status, consumed authorization
  admission, closed execution contract, and conversion result in addition to
  the existing case-23 and case-30 evidence.
- Case 23 retains its exceptional path-recovery requirements: one converter
  invocation, no converter retry, byte-identical canonical dataset, and closed
  learning state. Case 6 independently requires matching dataset/source/runtime
  identities, effective post-supervisor targets, reconstructed previous
  effective actions, and closed merge/BC/PPO/training state.
- Converted train cases are `[6, 23, 30]`. The minimum four-case train tranche
  now lacks only case `2`; validation still lacks cases `8` and `16`. Case `7`
  remains an additional train candidate. The corpus manifest is not ready.
- macOS and `.98` Windows/Isaac Python produced byte-identical reports at
  SHA-256
  `d151a8c144005737eb2a86d4038fda19b9ed6da2286cc9e1f3132024bb8fdea8`.
  Evidence is preserved under
  `evidence_20260724_model_based_corrective_corpus_intake_v3`.
- Implementation commit
  `c1e45027ec14f7aeaf4c39b066cd99418d6f116b`, evidence/goal commit
  `aa7eb27f207c81e1a6a397df6aea18fbc1272217`, and downstream goal-audit fix
  `7a4eeb3e945e1c76c1fa05c37a42a4267bb4c6d7` are pushed and synchronized.
- The first full suite correctly exposed one stale expected dataset count:
  `1359 passed, 1 failed, 12 skipped, 2 warnings in 196.83 s`. After updating
  that downstream ledger assertion, focused coverage passes
  `45 passed, 2 warnings in 5.18 s` on macOS and
  `45 passed, 2 warnings in 12.56 s` on `.98`. The authoritative `.98`
  Windows-Isaac Python CPU suite passes
  `1360 passed, 12 skipped, 2 warnings in 190.65 s`.
- No corpus merge, new conversion, Isaac/GPU workload, capture, BC, PPO,
  checkpoint, or training run was created. Goal completion remains `6/10`.

## Next round after Round 255

- The next data-producing operation requires separate authorization:
  `Authorize exactly one case-2 paired canary.`
- Do not infer capture, conversion, corpus-merge, BC, PPO, validation, or
  training authorization from this intake audit.

## Round 256: case-2 paired canary is rejected

- Issued and consumed exactly one external mode-`0600` authorization for the
  case-2 natural-error paired canary. Authorization commit
  `b94d40587510cd6e7ccb123f4557c6447e88479d`, Windows test-fix/runtime commit
  `9363f2818688653c2c6db60699caba496a0c8d3a`, and evidence commit
  `a24a1fb1305dc71bcd4953025e5a290283b11a7a` are pushed.
- The token was deleted before Isaac started. Baseline ran first; candidate ran
  only after the baseline passed. Both exited `0`, passed the unchanged dynamic
  gates, had zero action saturation, and released the GPU.
- Baseline position p95/max was `0.1398296339 / 0.1535142853 m`. Candidate
  position p95/max was `0.1395659017 / 0.1530445505 m`. The improvement was
  only `0.0002637322 m` (`0.1886096%`), below the required `0.003 m` and `2%`.
- The finalizer also failed closed because the case-specific projection
  observer did not inject its expected telemetry block into the runtime JSON.
  Ordinary corrective telemetry exists, but it does not satisfy that pinned
  evidence contract.
- Final-status SHA-256 is
  `92215fe1ebd95a027768bfc0d0c360bbd89e07d6d89b0ca41d4fe999ec281db8`.
  The complete namespace is preserved under
  `evidence_20260724_case2_natural_error_pair_execution_v1`.
- Case 2 is not eligible for corrective-label capture. The three converted
  train datasets `[6, 23, 30]` remain unchanged. No capture, conversion, merge,
  BC, PPO, checkpoint, or training run was created.
- At synchronized closure commit
  `b03ae2adf77a11d1ac5d2e35bb2656f6410b1bd1`, the authoritative `.98`
  Windows-Python CPU suite passed `1362 passed, 12 skipped, 2 warnings in
  188.10s`.

## Next round after Round 256

- Repair the shared projection-observer evidence contract CPU-only so later
  canaries fail for the correct reason.
- The next data-producing operation requires a new separate authorization:
  `Authorize exactly one case-7 paired canary.`

## Round 257: pending corrective routes use one CPU preflight

- Replaced repeated operator-side CPU preparation for pending cases 7, 8, and
  16 with one committed route catalog and one fail-closed preflight command.
  The existing case-specific wrappers remain the runtime backends.
- The catalog binds 61 identities across the three routes and verifies the
  canonical 65-observation, three-action residual contract with scales
  `[0.05, 0.05, 0.02]`.
- The first Windows run exposed two portability defects rather than route
  defects: native Windows Python passed `G:\...` paths to WSL Git and launched
  wrapper checks through Git Bash without WSL `python3`. Both boundaries now
  use explicit WSL translation and have regression coverage.
- The final focused compatibility suite passes `46 passed, 2 warnings in
  6.96s` locally and `46 passed, 2 warnings in 66.63s` on `.98`. Both hosts
  produced byte-identical all-route reports at SHA-256
  `74aafa819003e41fbf8040ad4ba563915a149ae21e4fb55b98d20c5567ddbd52`.
- Evidence is preserved under
  `evidence_20260724_model_based_corrective_route_catalog_cpu_v1`.
  Implementation commit
  `b5292df1ebf704f13e7e67552502b70635c34197` is pushed and synchronized.
- No plan, controller command, dynamic gate, profile, or safety ownership
  changed. No authorization token, runtime namespace, Isaac/GPU workload,
  capture, conversion, merge, BC, PPO, checkpoint, or training run was
  created. Goal completion remains `6/10`.

## Next round after Round 257

- Repair the shared projection-observer evidence contract CPU-only so later
  canaries fail for the correct reason.
- The next data-producing operation remains separately authorized:
  `Authorize exactly one case-7 paired canary.`

## Round 258: projection evidence survives Isaac shutdown

- Replaced admission dependence on the unreliable adapter-injected telemetry
  block with schema
  `cinebotrl_two_wheel_riser_corrective_projection_evidence_v2`.
  The extractor reads atomic requested-action, effective post-supervisor
  action, projection-delta, affected-sample, completed-step, and action-scale
  fields already written by shared playback.
- The extractor is evidence-only and fail-closed. It rejects missing,
  non-finite, out-of-bound, nonzero disabled-route, or inconsistent-count
  inputs. It does not intercept or modify commands.
- Reopened the preserved case-2 pair CPU-only. Projection evidence now passes
  over `9,204` candidate steps with `65` projection-affected samples, and the
  dynamic pair is complete. Case 2 remains rejected for the correct reason:
  position p95 improvement is only `0.0002637322 m` (`0.1886096%`), below the
  unchanged `0.003 m` and `2%` gates.
- Resealed the pending case-16 validation route with 25 identities, including
  the new projection evidence engine. The consolidated pending-route catalog
  now binds 62 identities across cases 7, 8, and 16.
- The affected suite passes `68 passed, 2 warnings in 11.61s` locally and
  `68 passed, 2 warnings in 83.31s` on `.98`. Both hosts produced
  byte-identical route reports at SHA-256
  `b58ebb7725f483777b732b05ff88ae0ec6aca0477ffc1f7cfe42de2872046f5d`.
- Evidence is preserved under
  `evidence_20260724_corrective_projection_evidence_repair_cpu_v2`.
  Implementation commit
  `d51702fa25ec90ffc8c64d23bd92ebbfe1b9c620` is pushed and synchronized.
- No plan, controller, profile, dynamic threshold, action scale, or command
  path changed. No authorization token, runtime namespace, Isaac/GPU
  workload, capture, conversion, merge, BC, PPO, checkpoint, or training run
  was created. Goal completion remains `6/10`.

## Next round after Round 258

- The shared projection-evidence blocker is closed.
- The next data-producing operation requires separate authorization:
  `Authorize exactly one case-7 paired canary.`

## Round 259: case-7 paired corrective target passes

- Consumed exactly one external mode-`0600` authorization and ran the pinned
  case-7 wrapper once at clean synchronized runtime commit
  `4b1f3fe3868bf380aa4fc1cf84f6095d9bf41861`. The token was removed before
  Isaac started; baseline and candidate each ran once with no retry.
- Both rollouts used the same plan, source/execution clocks, reset seed,
  controller, safety gates, and deterministic 20-step `20 N` body-x wrench.
  Both dynamic-quality gates passed without termination or saturation.
- Candidate position p95 improved from `0.130825693 m` to `0.124908848 m`:
  `0.005916844 m` absolute and `4.5227%` relative. Position max improved from
  `0.142947680 m` to `0.141102227 m`; attitude max improved, while pitch
  remained within the allowed non-regression margin.
- The repaired projection evidence passed over `6,597` candidate steps.
  Nine samples were projection-affected; effective normalized maxima remained
  `[0.383306444, 0.201544181, 0.063141920]`. The evidence observer did not
  modify commands.
- Preserved the complete namespace and checksums under
  `evidence_20260724_case7_corrective_pair_execution_v1`. Final-status SHA-256
  is `7c2f7ac0ff1b8bc1e50d95d317dabcb097e2cef35f2c5ab817bc0b73e62179d4`;
  projection-audit SHA-256 is
  `9ca2080d71c2bfae344ed172c4882ec94bc0d82e7160f4b02df94be020a6d9c3`.
- Evidence/status commit
  `69516b4273ec9b363f328eade80835b2422f748d` preserves the complete
  hash-bound result.
- Closure commit `1f524236fbf51dc62c79a6e6e6641b3470daed3b` is synchronized
  to `.98`. Focused coverage passes `37 passed, 2 warnings in 8.03 s` on
  macOS and `37 passed, 2 warnings in 47.22 s` on `.98`; the authoritative
  `.98` Windows-Isaac Python CPU suite passes
  `1375 passed, 12 skipped, 2 warnings in 222.39 s`.
- The consumed route is closed again in source. GPU ownership is released,
  and no label capture, dataset, conversion, corpus merge, BC, PPO,
  checkpoint, or training output was created. Goal completion remains `6/10`.

## Next round after Round 259

- The next data-producing operation requires separate authorization:
  `Authorize exactly one case-7 corrective-label capture.`
- Do not reuse the consumed paired-canary authorization and do not proceed
  directly to conversion, corpus merge, BC, PPO, validation, or training.

## Round 260: case-7 corrective capture route is CPU-ready and closed

- Added a case-bound capture contract, validator, wrapper, and finalizer at
  implementation commit
  `93d9b60eea2a4fa4bdd9748e5e4864d52f456514`. The route pins the successful
  case-7 pair, its projection audit, exact plan, corrective and wrench
  profiles, active drive profile, controller/runtime code, and robot assets.
- The route uses a fresh namespace and accepts only a separate external
  mode-`0600` token whose SHA-256 is supplied out of band. No token or hash is
  committed. Running `--execute` without that token rejects before Python or
  Isaac starts.
- The tokenless `.98` preflight passed all 19 identity, pair, projection,
  profile, clean-head/upstream, namespace, and closed-state checks. The
  preflight SHA-256 is
  `7ead7a8277c853960ae89bb78a5217df26af4d145c1f0e57f061f8a19eb4490c`.
- Focused coverage passes `13 passed, 2 warnings in 0.19 s` locally and
  `13 passed, 2 warnings in 0.65 s` on `.98`. The authoritative `.98`
  Windows-Isaac Python CPU suite passes
  `1382 passed, 12 skipped, 2 warnings in 221.90 s`.
- Evidence is preserved under
  `evidence_20260724_case7_corrective_capture_route_cpu_v1`. No runtime
  namespace, authorization token, Isaac/GPU workload, label capture, dataset,
  conversion, corpus merge, BC, PPO, checkpoint, or training run was created.
  Goal completion remains `6/10`.

## Next round after Round 260

- The next data-producing operation requires separate authorization:
  `Authorize exactly one case-7 corrective-label capture.`
- A successful capture would still require a separately authorized CPU
  conversion before case 7 could join the train corpus.
