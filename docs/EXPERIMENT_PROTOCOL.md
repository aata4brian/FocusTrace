# Final Experiment Protocol

## Session structure

Each participant completes exactly one session:

1. preflight and device pairing;
2. 30 s `CALIBRATION`;
3. `TRANSITION`;
4. six 120 s experimental blocks;
5. `TRANSITION` between blocks;
6. final STOP and integrity review.

Nominal structured time is 12 min 30 s. Actual raw video can be longer because transitions have variable duration and remain continuously recorded.

## Conditions

- **A1 / DTE Text**: read on the laptop and type an answer.
- **A2 / DTE Video**: watch a local laptop video and answer.
- **B1 / TREE Paper**: task-relevant work off-screen/on paper.
- **B2 / TREE External Video**: relevant material on an external stimulus device.
- **C1 / TIE Scrolling**: standardized simulated non-academic social feed; never personal social media.
- **C2 / TIE Conversation**: structured neutral, non-sensitive, task-irrelevant conversation with the researcher/confederate approximately 30–45 degrees to the side.

Every session contains A1, A2, B1, B2, C1, and C2 exactly once. Presets counterbalance their order across participants.

## State/label rules

Raw system states include `READY`, `CALIBRATION`, `BLOCK`, `TRANSITION`, and terminal states. Experimental conceptual labels are DTE, TREE, and TIE.

Main binary mapping:

- DTE -> 0
- TREE -> 0
- TIE -> 1

Calibration, transition, unknown/invalid intervals, and unusable missing-data ranges are excluded from supervised training. Off-screen behavior is not automatically task-irrelevant: TREE is explicitly on-task despite expected gaze/head movement away from the main screen.

## Synchronization

The laptop session clock is the sole authoritative experimental clock. Video frame timestamps, feature rows, events, and block intervals use the same monotonic session timebase. Phone clocks are never ground truth.

## Recording

Raw video starts at session START and remains continuous until STOP/ABORT. Visual overlays are optional derived outputs and must never replace the clean raw recording.

## Review

Initial experimental labels remain immutable. Post-session verified corrections are stored separately with reviewer, time range, reason, initial label context, and timestamp. Dataset preparation can require verified-only windows when appropriate.

## Privacy and scientific scope

Use anonymous participant IDs. Do not implement face recognition, identity inference, personal social-media access, cloud upload, emotion inference as the main signal, or claims that the system measures internal mental focus. Follow the approved ethics/consent and data-retention process for the study.
