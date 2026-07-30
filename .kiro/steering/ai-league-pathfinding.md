---
inclusion: auto
---

# AWS AI League Pathfinding Project Checkpoint

## Working agreement

Act as a candid project partner. Say directly when an approach is wrong, explain why plainly, and recommend a better alternative when available. Protect known-good tournament submissions by testing changes in separate agent versions.

## Champion baseline

- Best completed Round 1 score: `15302.000`
- Agent version: `20f4f040-24b4-4c61-935d-0486ef194e7b`
- Total tokens: `1673`
- Lives remaining: `5`
- Lives lost: `0`
- Preserve this version as the champion/fallback. Never overwrite or reset it blindly.

## Repository and verified assets

- Repository: `WannZul/pathfinding-dataset`
- Pathfinding implementation: `pathfinding_lambda.py`
- Stage 1 training data: `training-dataset.jsonl` (500 records)
- Stage 1 validation data: `validation-dataset.jsonl` (100 records)
- Corrected evaluator: `tool_call_reward_evaluator.py`
- PR #2 was merged into `main` and contains the evaluator plus refreshed labels at training indices 238 and 450.
- Current local Lambda replay matches 500/500 training records and 100/100 validation records exactly.
- Validation data must stay separate from training; upload it only as the validation asset.

## AWS resources and evaluator

- Region: `us-east-1`
- Live Pathfinding Lambda name: `AgentCoreGatewayTool-Pathfinding_1196`
- SageMaker evaluator asset: `pathfinding-tool-call-reward-v1`
- Its live test returned aggregate reward `1.0`, including correct tool, all argument matches, Lambda success, and full output match.
- The evaluator handles Qwen tags, legacy tags, and nested raw JSON. Full output credit compares `statusCode`, `path`, `steps`, and `start_position`.
- The SageMaker-generated evaluator handler must be retained rather than replacing it with the older workshop handler.

## Stage 1 customization result

- Base model: `Qwen3-0.6B`
- Technique: RLVR with LoRA
- Training asset: `pathfinding-toolcall-train-v1`
- Validation asset: `pathfinding-toolcall-validation-v1`
- Training job: `pathfinding-toolcall-stage1-v1-1785080714288`
- Configuration: batch size 128, 2 epochs, learning rate `0.00001`, LoRA alpha 64, LoRA rank 32, 8 rollout samples per prompt.
- Job completed successfully and reward increased while rollout length decreased.
- It produced only about 6 optimizer steps, below the workshop's approximately 25-step and 0.85+ reward target.
- Round 1 access closed before the adapter could be continued, registered, deployed, or assigned to a new agent version.

## Recovery and next steps

1. When the next round opens, first check whether the same AWS account still exposes training job `pathfinding-toolcall-stage1-v1-1785080714288`.
2. If accessible, register or preserve the model immediately before further experiments. Continue Stage 1 for roughly 7 additional epochs only if the platform continues from the six-step adapter.
3. If inaccessible, contact the organizer with the training-job name and request recovery/registration. If recovery is impossible, recreate Stage 1 from the merged GitHub assets; approximately 9 epochs at batch size 128 yielded an estimated 27 steps from scratch, but inspect the current UI and workshop guidance before launch.
4. Do not move to faithfulness training until tool-calling reward is at least about 0.85 and format/tool/argument metrics are healthy.
5. Stage 2 must continue from the Stage 1 adapter and uses a separate faithfulness dataset (workshop target: 403 training and 81 validation records) plus a separate exact-path evaluator.
6. Register and deploy the final model, then attach it only to a new experimental Pathfinding sub-agent version.
7. Compare the experimental submission against the champion score of 15302; retain the champion if the new version is worse.

## Important lessons

- Model customization is required by the official AI League workflow; use Qwen3-0.6B and RLVR, not SFT or Qwen2.5-7B.
- The pathfinding customization is map-agnostic and reusable across compatible rounds, but AWS resources may be scoped to a round/account. Reusability of model behavior does not guarantee persistence of the hosted artifact.
- Register completed model artifacts promptly before a round closes.
- Do not confuse the champion Agent Version UUID with a Lambda name, Gateway ID, or model ARN.


## Round 2 preliminary rules and map

- Round 2 preserves the core Round 1 objective: collect coins, complete safe challenges, preserve lives, and enter treasure last.
- The run stops on timeout, zero lives, or reaching treasure.
- Disallowed: external-model tool calls, hardcoded kiosk answers, and out-of-scope agent use.
- Starts with 8 lives. Each remaining life contributes 250 raw points; maximum life bonus is 2000.
- Treasure contributes 1000 raw points.
- Token bonus is `1000 - (total tokens used / challenges visited)`; prompts and responses should therefore be concise.
- Round 2 league multiplier is 2. League points are `((participants - rank) / (participants - 1) * 1000) * 2`.
- Grand-final qualification requires a top-five league position and participation in at least three of four rounds. Total raw game score is the tiebreaker.

### New Round 2 cells

- `c41`: green key, +50. Must be collected before `c31`; memory provides unlock information and the agent must say `Thanks` when receiving the key.
- `c31`: green door, +1000 and -5 lives if mishandled. Requires the green key, then translates received letters to their alphabet positions.
- `c17`: distraction chest, +50; an incorrect answer costs 2 lives. Collect it only when the supervisor can answer accurately and concisely.
- Existing red key/door mechanics remain and must track independently from green key/door state.
- Spikes are hazards. Minimize entries globally; the exact Round 2 map forces three crossings because of its start isolation and cross-component key/door dependencies.

### Round 2 visible map facts

- Map size: 10x10.
- Start: F5.
- Treasure: A1; treasure must be last.
- Green key: G3.
- Green door: D8.
- Red key: A10.
- Red door: E1.
- Distraction chest: B8; collect with a concise accurate answer.
- The map includes approximately 30 visible coins plus multiple ordinary challenges and spike cells.
- Never hardcode this map or its path in prompts/Lambda code. Use it only as a test case for generic semantics.

### Required implementation changes

- Extend the path planner to track red and green keys separately.
- Block `c30` until `c40` is collected and block `c31` until `c41` is collected.
- Treat `c17` as a low-value answerable reward and `c8` as a globally minimized traversal cost.
- Collect keys, safe challenges, doors, and coins before entering treasure.
- Update tool schemas and generated datasets to include `c17`, `c31`, `c41` and generic multi-key semantics.
- Exact Round 1 challenge text/point/life rules for all unchanged challenge IDs still need to be reconfirmed from the current Round 2 Challenges tab before final prompt changes.


## Round 2 exact challenge catalog

The current Round 2 Challenges and Tools & Strategy screenshots confirm:

- `c7` Some Coins: +250, no question.
- `c30` Red Door: +1000; incorrect/unsafe handling can lose 5 lives. Requires `c40`; decode the key code by reading it backwards.
- `c3` Memory Trial: +550; incorrect answer loses 1 life. Use AgentCore Memory to recall map/interactions; questions may ask counts or sums of challenge types.
- `c2` Code Challenge: +600; incorrect answer loses 1 life. Use the Code Interpreter Lambda for computational questions and return only the requested result.
- `c4` Web Search: +800; incorrect answer loses 1 life. Use the Web Scraper Lambda for the specified website; no additional dependencies are installed by default.
- `c41` Green Key: +50. Must precede `c31`; memory supplies unlock information; reply `Thanks` when receiving it.
- `c8` Spike: obstacle; crossing loses 1 life.
- `c5` Simple Question: +250; incorrect answer loses 1 life. Answer accurately with minimal tokens.
- `c17` Distraction: +50; incorrect answer loses 2 lives. Answer accurately, omit filler/preamble/restatement, and provide only the essential answer.
- `c1` Violent Violet: +400; incorrect answer loses 1 life. Guardrail must block illegal activity, violence, hate, misconduct, and any mention of edible flowers, transplanting, or weeds, without overblocking unrelated questions.
- `c31` Green Door: +1000; incorrect/unsafe handling can lose 5 lives. Requires `c41`; translate received letters to alphabet-position numbers.
- `c40` Red Key: +50. Must precede `c30`; memory supplies unlock information; reply `Thanks` when receiving it.

### Round 2 corrections

- `c17` is not an automatic -2-life tile. The heart value is the failure penalty, as with other question challenges. Do not categorically avoid it: visit it when the supervisor can answer reliably and concisely. Its low +50 reward means it has lower priority than high-value challenges, but a correct concise answer can still improve raw score and the challenges-visited denominator in the token bonus.
- The official final Round 1 leaderboard maps score `15302` to rank 4 of 356 teams, worth approximately `991.5` league points. The earlier rank-5 note reflected the live leaderboard before finalization.
- Tools & Strategy confirms the default pathfinder is `swift`; custom strategy names can be supplied dynamically through the Navigation Prompt.
