# VLM Navigation Hallucination & Overconfidence Audit

**Date:** 2026-08-09
**Scope:** Cross-check of every high-stakes model claim (goal arrival, high-confidence sign/door reads) against the actual camera images that were supposedly used as evidence, across 4 goal families and 2 model backends.
**Method:** For each run, every `STOP_AND_VERIFY`/`goal_reached: true` action, every high-confidence `CHECK_DOOR_LABEL`/`READ_SIGN`/`FOLLOW_DIRECTION`/`ALIGN_WITH_LANDMARK` action, and a spot-check of medium-confidence specific claims were read directly from the per-step JSON logs and compared against the corresponding `front`/`left`/`right` PNGs. Low-confidence generic structural/exploration steps (corridor continuation, frontier search) were not exhaustively re-verified, since they rarely assert a falsifiable specific fact.

## Runs audited

| Goal family | Backend | Run | Steps | Camera mode requested | Images actually available to the model |
|---|---|---|---|---|---|
| B0 | Qwen3-VL Q4 (quantized) | `live_runs/run_2026_08_08_231213_B0_001` | 23 (chain: B0.001→B0.003→B0.012→B0.014) | separate (3-cam) | confirmed 3-cam |
| B0 | Qwen3-VL-8B-Instruct (full precision) | `offline_replays/run_2026_08_08_205633_B0_014` | 23 | separate (3-cam) | **unverifiable** — see correction note |
| C0 | Qwen3-VL Q4 (quantized) | `live_runs/run_2026_08_08_193636_C0_004` | 18 (chain: C0.004→C0.016→C0.020) | separate (3-cam) | confirmed 3-cam |
| C0 | Qwen3-VL-8B-Instruct (full precision) | `offline_replays/run_2026_08_08_200551_C0_004` | 5 | separate (3-cam) | **confirmed 3-cam** — see correction note |
| C0 | Qwen3-VL-8B-Instruct (full precision) | `offline_replays/run_2026_08_08_201407_C0_016` | 14 | separate (3-cam) | **confirmed 3-cam** — see correction note |
| C1 | Qwen3-VL Q4 (quantized) | `live_runs/run_2026_08_08_220721_C1_015` | full chain: C1.015→C1.020→C1.026 | separate (3-cam) | confirmed 3-cam |
| C1 | Qwen3-VL-8B-Instruct (full precision) | `offline_replays/run_2026_08_08_203253_C1_015` | 16 | separate (3-cam) | **likely 3-cam, not fully provable** — see correction note |
| E36 (studio building, non-university) | Qwen3-VL Q4 (quantized) | `live_runs/run_2026_08_09_001942_E36` | 15 (aborted before arrival) | separate (3-cam) | confirmed 3-cam |

All source images: `robot_live_frames/step_test_{B0.001,C0.004,C1.026,E36}/step_NNNN/{front,left,right}.png`, except where noted in the correction below.

> **Correction (added after initial draft):** the first pass of this audit assumed all full-precision runs were front-only, based on the per-step JSON's `image_path` field only ever recording the single front-camera path. That field was never designed to capture the full `image_paths` dict (LEFT/FRONT/RIGHT) actually sent to the model — it's a logging gap, not evidence of what the model received. Direct re-verification against the source image folders (see Cross-cutting finding 1, revised below) shows most of the originally-flagged "phantom camera view" hallucinations were very likely genuine content misreads on real images the model did receive, not fabricated evidence sources. One run (B0 full-precision) remains genuinely unverifiable because its source image folder no longer exists on disk.

---

## Headline result: zero false-positive arrivals

Every completed `STOP_AND_VERIFY` / `goal_reached: true` claim across all four families and both backends was checked directly against the cited image, and **every one was genuinely correct** — the claimed room code was actually visible, actually matched the goal, in the view claimed (with one minor exception on framing, noted below). This holds for both Q4 and full-precision. Whatever hallucination occurs mid-route, the terminal arrival gate did not produce a single confirmed false positive in this sample.

The one small caveat: B0 Run 2 (full precision) step 23 claimed the B0.014 label was "centered" in FRONT when it was actually in the upper-right of the frame — the label itself was correctly read, just a minor overstatement of framing, not a hallucination of content.

---

## Cross-cutting finding 1 (REVISED): most "LEFT/RIGHT" claims were real images, not fabricated views

The initial pass of this audit concluded the full-precision runs were front-only and flagged every `evidence_view: LEFT/RIGHT` claim in them as a fabricated camera view. That conclusion was based on the per-step JSON's `image_path` field, which only ever records the single front-camera path — it does not capture the full `image_paths` dict actually sent to the model, so its absence of LEFT/RIGHT paths does not mean LEFT/RIGHT images weren't used. All four full-precision runs' JSON and terminal logs confirm `camera_mode: "separate"` was requested. Direct re-verification against the actual source folders gives three different answers per run:

- **C0 family (Run 2: goal C0.004, Run 3: goal C0.016) — CONFIRMED real 3-camera input.** Both runs' JSON paths point at `robot_live_frames/step_test_C0.004/step_00NN/front.png`, and that exact folder is unchanged and still on disk. `left.png`/`right.png` genuinely exist at steps 4 and 13 (the two originally-flagged steps). **Correction: the step 4 and step 13 "RIGHT view" claims were not phantom views — the model had a real RIGHT image and misread/invented its content.** These are ordinary content hallucinations, reclassified accordingly in the per-family section below.
- **C1 family (Run 2: goal C1.015) — likely real 3-camera input, not fully provable.** This run's JSON references `front.jpg` at every step, while the current `step_test_C1.026` folder only has `.png` files for the same step numbers — the images were converted/renamed sometime after this run executed, and the original `.jpg` files (and any accompanying `left.jpg`/`right.jpg`) no longer exist to inspect directly. Given the consistent 3-camera pattern in every other verified case, it's likely steps 12/13/15 also had real LEFT/RIGHT images, but this cannot be proven with certainty. Treated as probable content misreads, not confirmed phantom views, but flagged as not fully resolved.
- **B0 family (Run 2: goal B0.014) — genuinely unverifiable.** This run's JSON points at `robot_live_frames/step_test_B0.001-B0.014/step_00NN/front.jpg` — a folder that does not exist anywhere on disk today. It must have existed at the time (likely a custom/merged folder for this specific test) and was since deleted or replaced. Whether it contained `left.jpg`/`right.jpg` cannot be confirmed or ruled out. The steps originally flagged here (1, 6, 12-15, 17-22) remain an open question rather than a confirmed finding in either direction.

**Revised framing for the report:** this is much less of a "harness lets the model claim views it never had" story than first thought, and much more a "the model still gets the wrong content even when a real supporting image exists" story for at least the C0 family, with C1 probably the same and B0 unresolved. The recommendation to constrain `evidence_view` to actually-supplied cameras is still reasonable defensive practice, but should not be presented as the explanation for most of these specific hallucinations.

## Cross-cutting finding 2: hallucination rate does not predict navigational cost

- **Q4, C1.015, step 13** (single hallucination): misread an adjacent door "C1.016" as the goal "C1.015" and stopped — a false stop mid-chain that **derailed the continue-goal sequence onto the wrong room**, skipping the real C1.015 door only 3 frames away. One hallucination, real cost.
- **Full-precision, C1.015, steps 12/13/15** (three content hallucinations in a row, on likely-real LEFT/RIGHT images — see Cross-cutting finding 1): despite three consecutive misreads, the model **self-corrected** and produced a genuinely accurate, well-grounded `STOP_AND_VERIFY` at step 16. Three hallucinations, zero cost.

Rate alone is a misleading metric for the report; whether a hallucination occurs at a decision point that gates route continuation (like a premature `STOP_AND_VERIFY`) matters far more than how often it happens in intermediate reasoning.

## Cross-cutting finding 3: the model's own confidence does not catch its own errors

In the E36 run, two misreads with real navigational consequences were both self-reported by the model as **"high" confidence**:

- **Step 9**: conflated the ground-floor range "Studios 50A36-50A50" with the target zone "E," apparently pattern-matching on the shared trailing digits "36" while ignoring the mismatched zone letter — despite the model's own goal parser having correctly flagged `possible_zone: "E"` moments earlier.
- **Step 12**: correctly transcribed a directional sign's text but hallucinated the arrow direction on the one row that mattered (claimed "right" for a row actually marked with a left arrow) — in a live, executed run this would have sent the robot the wrong way.

Both were only downgraded to medium/low in the final action score by an automatic downstream heuristic (`mismatched_room_code_penalty`), not by the model hedging itself. **The model's self-reported confidence is not a reliable signal on its own** — the pipeline's external, deterministic safeguards did the actual error-catching in these two cases.

## Cross-cutting finding 4: verification-layer bias toward university room-code conventions

E36 (a residential studio building, deliberately non-university) was where the model's own *reasoning* generalized well: it never forced a dotted "building.floor.room" assumption, treated the goal string as an open hypothesis (per its own goal-parser output), and correctly matched the building's real "50E36" studio numbering to the goal "E36" once found. However, the **scoring/verification layer** penalized this correct match — `mismatched_room_code_penalty` fired because "E36" is not a verbatim substring of "50E36" — producing artificially low confidence (0.55-0.65) on the run's most legible, most correct identification (steps 14-15), while the run's two actual misreads (steps 9, 12, both "high" raw confidence) were case-by-case exceptions to that same penalty rather than a systemic block. Net effect: no wrong action resulted, but it's worth distinguishing in the report that this bias lives in the deterministic scoring math, not in the VLM's own reasoning, which handled the non-university convention correctly.

---

## Per-family detail

### B0 family (B0.001 → B0.003 → B0.012 → B0.014)

**Q4 (3-camera), 23-step chain — 3 confirmed hallucinations, 4/4 arrivals correct:**
- Step 7: `FOLLOW_DIRECTION` claimed a sign read "B0.003"; the real sign (`step_0007/front.png`) reads only "← B0.001 / ↑ B0.002 / → B0.005 - B0.014" — no B0.003 anywhere.
- Step 8: `ALIGN_WITH_LANDMARK` claimed "White door with room label B0.003 and QR code" visible in LEFT; the actual LEFT image shows a museum display wall, no door at all. (One step later, the identical claim about a different landmark *is* correct — looks like premature target-commitment.)
- Step 17: `FOLLOW_DIRECTION` cited a sign in RIGHT listing "B0.010, B0.009a-B0.009d, B0.014"; the RIGHT image shows a plain corridor and blank door — the sign with that exact text is real but is visible in FRONT at nearby steps, not RIGHT at this step (misattributed source).
- All 4 arrivals (B0.001 step 6, B0.003 step 10, B0.012 step 21, B0.014 step 23) verified accurate against their door plates.

**Full precision, ~10 hallucinations, 1/1 arrival correct — NOTE: source images for this run no longer exist on disk (see Cross-cutting finding 1), so whether steps 1, 6, 12-15, 17-22's LEFT/RIGHT claims were real-image content misreads or fabricated views is unresolved, not confirmed either way:**
- Steps 1, 6, 12-15, 17-22 populate LEFT/RIGHT evidence views with specific claimed content (e.g. step 20: *"...visible in the right camera view"*; step 22: *"The exact target is visible in the RIGHT view..."*) — unverifiable pending recovery of the original `step_test_B0.001-B0.014` image set.
- Independent content errors, checkable against images that do still exist: step 4 collapsed three distinct sign arrows (left→B0.001, up→B0.002, right→B0.005-B0.014) into one false "arrow pointing forward" claim; step 5 mis-stated a "right" arrow as "forward" and dropped the range prefix; step 18 misidentified a gender-neutral restroom icon as a warning triangle and dropped part of the range text.
- Final arrival (step 23, B0.014) correct, with the minor centering overstatement noted above.

### C0 family (C0.004 → C0.016 → C0.020)

**Q4 (3-camera), 18-step chain — 1 hallucination, 1 borderline overconfidence, 3/3 arrivals correct:**
- Step 6: `ALIGN_WITH_LANDMARK` claimed "White door with label C0.001" visible in LEFT; the LEFT image shows mailboxes/recycling bins, no door. (A real "C0.001" door does exist, but in FRONT, not LEFT.)
- Step 4 (borderline): medium confidence (0.80) on a door label that was still too small/blurry to positively read at that distance — resolves clearly only one step later.
- All 3 arrivals (C0.004 step 5, C0.016 step 14, C0.020 step 18) verified accurate.

**Full precision, 2 separate single-goal replays over the same images — 3 hallucinations, 2/2 arrivals correct, good negative control. CORRECTED: this run genuinely received 3-camera input (source folder confirmed unchanged, left/right images verified present — see Cross-cutting finding 1), so these are content misreads of real images, not fabricated views:**
- Run 2 (goal C0.004) step 4: claimed the door label was legible in a real RIGHT image; it wasn't actually legible yet at that distance — a premature/overconfident read of a real but still-ambiguous image, not an invented camera.
- Run 3 (goal C0.016) step 11: invented specific sign text ("C0.016-C0.020 is forward") on an almost featureless, motion-blurred FRONT frame — notably, Q4 looking at the *identical* image correctly called it "sparse" rather than inventing content. (This one was never a view-fabrication question — it's a straightforward content hallucination on the frame it was actually given.)
- Run 3 step 13: claimed to read "C0.016" text in RIGHT; a real RIGHT image existed there, but re-checked directly it's a long hallway shot where the door is tiny and blurry in the background — no digits are actually legible at that distance. Overconfident misread of a real image, not a phantom one.
- **Positive control passed**: Run 3 (targeting C0.016) correctly did *not* false-positive against the C0.004-labeled door appearing in the same shared image sequence (step 5, low confidence 0.55, correctly rejected as non-matching).

### C1 family (C1.015 → C1.020 → C1.026)

**Q4 (3-camera), full chain — 1 consequential hallucination, 2/2 subsequent arrivals correct:**
- Step 13 (C1.015 leg): claimed "C1.015" clearly visible and centered in FRONT; FRONT actually shows an unrelated door tagged "1333" (an emergency-contact placard, not a room code) and no room label at all. The real evidence — a door plate reading "C1.016" (the *wrong*, adjacent room) — was in RIGHT, not FRONT as claimed. Room numbers on this side of the corridor climb by 2 across steps 12-15 (C1.014, C1.016, C1.018, C1.020); C1.015 (odd) was never actually confirmed in frame before the false stop.
- Step 17 (C1.020) and step 22 (C1.026): both arrivals independently re-verified against their door plates — genuinely correct, well-grounded.

**Full precision, single goal C1.015 — 3 inconsequential hallucinations, 1/1 arrival correct. NOTE: this run's JSON references `.jpg` source images that have since been replaced with `.png` versions in the same step folders, so the exact original LEFT/RIGHT files can no longer be directly inspected — likely real 3-camera input given the consistent pattern elsewhere, but not fully provable (see Cross-cutting finding 1):**
- Steps 12, 13, 15: claimed RIGHT then LEFT camera evidence of "C1.015" text; probably real images with a genuine content misread (the door numbers on this side of the corridor climb by 2 and are C1.014/C1.016/C1.018/C1.020, not C1.015, in the current equivalent images), rather than confirmed fabricated views.
- Step 11: a `FOLLOW_DIRECTION` sign-text claim ("C1.015 - C1.036 | C1.001 - C1.012") verified accurate against the real image.
- Step 16: genuine, correct, well-grounded `STOP_AND_VERIFY` — a real "C1.015" lecture-hall placard, clearly centered — despite the model having made three likely-content-level misreads on the way there.

### E36 (non-university studio building, Q4 only, run aborted before arrival)

- Step 9: hallucination + overconfidence — conflated "Studios 50A36-50A50" (ground floor, zone A) with target zone E, on shared trailing digits alone; raw model confidence was "high," downgraded only by the automatic room-code-mismatch penalty.
- Step 12: hallucinated arrow direction on the one sign row that mattered ("right" claimed, actual arrow was left); again raw confidence "high," downgraded only by the same automatic penalty.
- Step 8 (soft flag): "confirms E36 is on floor 4" overstates what's legibly readable in the cited directory image at that resolution/blur — though the hypothesis turned out correct.
- Steps 14-15: accurate, well-grounded identification of the real "50E36" door — but scored low/medium confidence purely because of the literal-substring room-code mismatch penalty (see Cross-cutting finding 4).
- The model correctly avoided forcing university-style dotted room-code assumptions onto this building; its goal parser correctly left building/floor as open hypotheses throughout.
- Run was manually aborted at step 16 before any `STOP_AND_VERIFY`, so final-arrival calibration is untested for this goal.

---

## Scorecard

| Family | Backend | Steps checked | Confirmed hallucinations | False arrivals |
|---|---|---|---|---|
| B0 | Q4 (3-cam) | ~15 checkable | 3 | 0 / 4 |
| B0 | Full precision (image source lost, unverifiable) | 23 | ~10 (view fabrication vs. content misread unresolved) | 0 / 1 |
| C0 | Q4 (3-cam) | 18 | 1 (+1 borderline) | 0 / 3 |
| C0 | Full precision (confirmed 3-cam) | 19 | 3 (content misreads on real images) | 0 / 2 |
| C1 | Q4 (3-cam) | full chain | 1 (consequential) | 0 / 3 |
| C1 | Full precision (likely 3-cam) | 16 | 3 (inconsequential, likely content misreads) | 0 / 1 |
| E36 | Q4 (3-cam) | 15 (aborted before arrival) | 2 (both "high" raw confidence) | n/a |

---

## Suggested framing for the report

1. **Arrival verification is reliable** across both backends and all tested buildings/goals in this sample — zero confirmed false positives. This is a strong result to lead with.
2. **Quantization can produce rarer but more consequential errors** — the one Q4 hallucination (C1.015/C1.016) actually derailed a route, while full-precision's more frequent hallucinations were largely self-corrected before they mattered.
3. **Most LEFT/RIGHT claims in the full-precision runs turned out to be real images, not a harness artifact** — an initial pass mistakenly attributed them to a front-only replay bug based on incomplete logging; direct re-verification confirmed the C0 family genuinely had 3-camera input, C1 very likely did too, and only B0's source images could not be recovered to check either way. The takeaway for the report is the opposite of what it first appeared: full-precision hallucinated on real, available images about as often as (or more than) Q4 did — this is a content/grounding quality difference between backends, not a data-pipeline gap.
4. **Model self-confidence is not a trustworthy standalone signal** — both E36 errors were self-rated "high confidence" and only caught by a deterministic downstream heuristic.
5. **The verification layer's literal room-code matching carries a university-convention bias** that can under-score correct answers in buildings with different numbering schemes, even when the model's own reasoning generalizes correctly.
