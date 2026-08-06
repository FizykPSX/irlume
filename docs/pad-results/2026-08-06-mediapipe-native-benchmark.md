# MediaPipe models on the native runtime: full benchmark and per-stage verdict

Date: 2026-08-06. Machine: ASUS Zenbook S 14, Intel Core Ultra 7 258V, 8 CPUs,
Fedora 44, bundled TFLite runtime v2.19.0 (`/usr/share/irlume/tflite/`).
Corpus: the 2026-08-05 stage-3 set, 512 frames (223 with an accepted YuNet
face), both cameras (Zenbook internal, NexiGo N930W). The frames live outside
the repository; their contents are pinned by
`2026-08-05-stage3-corpus.sha256`, and both new harnesses verify every frame
against that manifest and refuse on any difference, so the gates bind to this
exact evidence rather than to frame counts.

This closes the question the 0.9.0 cycle opened: irlume can now run Google's
published `.tflite` artifacts unconverted (#295), so every MediaPipe model
relevant to face authentication gets measured on that runtime and compared
with what irlume ships. Earlier measurements this doc builds on:
`2026-08-05-fullrange-threshold.md` (full-range operating point),
`2026-08-05-blaze-full-parity-*.csv` (full-range decode parity),
`2026-08-06-mesh-parity.md` (mesh conversion fidelity).

## Models covered

| Model | sha256 (first 16) | Source | Status in irlume |
|---|---|---|---|
| BlazeFace short-range `.tflite` | `b4578f35940bf5a1` | Face Detector card, Apache-2.0 | shipped as ONNX conversion (rescue slot) |
| BlazeFace full-range `.tflite` | `3698b18f063835bc` | Face Detector card, Apache-2.0 | decoder landed, stage closed (#276) |
| Face landmarks detector `.tflite` (478 pt) | `c7d54204ce044847` | face_landmarker.task `64184e229b263107` | shipped as ONNX conversion (also 478 pt) |
| Face blendshapes `.tflite` | `4f36dded049db18d` | same .task | never run before today |

Holistic stays excluded: 6 of its 7 inner files carry no model card, one is a
different 192px mesh (decision recorded on #276, 2026-08-05). The face
stylizer and the generic image embedder do no work an authentication pipeline
needs, so they were not benchmarked.

## Short-range BlazeFace: the shipped conversion is faithful

Harness: `crates/irlume-auth/examples/blaze_short_parity.rs`. Both models run
in one process on the identical input tensor (`blaze_letterbox_input`, the
production preprocessing) through the identical decode
(`decode_short_range_best`, the production decode), so the only difference
left is ONNX-converted weights on ort against original weights on TFLite.
This pair had never been compared on the native runtime; the 2026-08-05 bench
compared against the official Python runtime, which mixes in a preprocessing
gap.

Result over all 512 frames (`2026-08-06-blaze-short-parity.csv`): zero
one-sided detections, mean IoU 0.999999, minimum 0.999996, maximum score
delta 1.967e-6, and 139 of 512 frames score bit-identically across runtimes.
The instrument proves it can fail: a 2px shift injected into only the native
input drops mean IoU to 0.757 and leaves zero bit-identical scores
(`BLAZE_PARITY_SKEW_PX=2`). The harness is an enforcing gate: pinned 512/512
denominator, minimum IoU bound 0.9999, score-delta bound 5e-6.

## Face mesh: parity settled on 2026-08-06, latency measured today

`2026-08-06-mesh-parity.md` already established the shipped ONNX mesh
matches the .task's model at mean NME 6.9e-7 (worst 1.5e-6) over the same
crop, now extended to all 478 points (worst iris-only NME 1.249e-6, see
that doc's addendum). What was missing was cost. Measured on one frontal frame, 100
iterations after 10 warmup (`2026-08-06-mp-latency-zenbook.csv`):

| Stage call | Runtime | Threads | Mean ms | p50 | p95 |
|---|---|---|---|---|---|
| YuNet detect | ort | default | 13.12 | 12.10 | 23.67 |
| Short-range Blaze | ort | default | 4.23 | 3.16 | 11.89 |
| Short-range Blaze | tflite | 1 | 2.56 | 2.27 | 4.02 |
| Short-range Blaze | tflite | 2 | 2.77 | 1.98 | 4.65 |
| Short-range Blaze | tflite | 4 | 1.23 | 1.21 | 1.50 |
| Full-range Blaze | tflite | 2 | 4.29 | 4.20 | 5.23 |
| Mesh 468 (shipped) | ort | default | 10.34 | 9.90 | 15.53 |
| Mesh 478 (native) | tflite | 1 | 8.28 | 8.02 | 9.51 |
| Mesh 478 (native) | tflite | 2 | 5.79 | 5.69 | 6.96 |
| Mesh 478 (native) | tflite | 4 | 3.92 | 3.85 | 4.65 |
| Blendshapes | tflite | 1 | 1.72 | 1.71 | 1.77 |

Blendshapes timing includes per-frame input-tensor construction (#314
review; the upstream graph rebuilds that tensor every invocation). Google's
own reference for short-range Blaze is 2.94ms on a Pixel 6 CPU (Face
Detector solution page); the native runtime here lands in the same band.

One repeatability caveat, learned by running the bench twice: the ort means
moved between runs on this machine (mesh 7.62 to 10.34ms, short-range Blaze
3.13 to 4.23ms) while every tflite row held within its own p95. The safe
cross-runtime claim is therefore not a single percentage but a direction:
the native mesh at 2 threads was faster than the ONNX mesh in both runs,
with a visibly tighter p50-to-p95 band.

## Blendshapes: open-eye behavior and runtime

Harness: `crates/irlume-auth/examples/blendshapes_probe.rs`. Contract read
from `face_blendshapes_graph.cc` in the MediaPipe source: input [1, 146, 2],
146 selected landmarks as pixel coordinates, output 52 ARKit-order
coefficients. The 146-index subset ends with the 10 iris points (468-477).
An earlier revision of this doc claimed the shipped ONNX mesh lacks them;
that was wrong: the shipped `face_landmark.onnx` is the landmarker-generation
conversion and emits all 478 points (measured on a corpus frame), so it can
feed this model today. Iris parity, which the original #306 run skipped, is
now measured too: the extended mesh_parity gate compares all 478 points and
the iris-only worst NME is 1.249e-6, as faithful as the body (addendum in
2026-08-06-mesh-parity.md).

Over the 223 accepted corpus frames (`2026-08-06-blendshapes-probe.csv`),
pooled per-frame `max(eyeBlinkLeft, eyeBlinkRight)` against irlume's
production `min(EAR)` gives Pearson r = -0.938. This is an observational
correlation across changing cameras, illumination, glasses, pose, and mesh
quality; the corpus contains no controlled eyelid closures, so the number
cannot distinguish a blink cue from a shared image-condition artifact
driving both readings, and it establishes neither blink sensitivity nor
liveness value. All 52 outputs stayed finite on every frame.

What the run does establish is an open-eye nuisance baseline a future
labeled blink experiment must exceed: the blink coefficient never passed
0.37 as a segment mean, with RGB segments at 0.04 to 0.21, IR at 0.13 to
0.37, and glasses-on IR the worst case (0.32 to 0.37), the same conditions
that degrade EAR. Any authentication use needs paired open/closed captures
per camera, illumination, and glasses condition, with the threshold chosen
on frames the evaluation never saw.

## External corpus: 13,233 LFW frames on a second machine

Per the session's direction, the whole benchmark re-ran on archhost (AMD
Ryzen 7 5700G, CachyOS-adjacent rolling Arch, same bundled runtime build)
over the Kaggle LFW dataset (`jessicali9530/lfw-dataset`, deep-funneled
250px portraits), converted to PPM and content-pinned by
`2026-08-06-lfw-corpus.sha256`. The harnesses ran in their external-corpus
mode: every frame still verified against the manifest, agreement bounds
still enforced, only the stage-3 count pins replaced by the manifest total.
Per-frame evidence in the three `2026-08-06-lfw-*.csv` files.

- **Short-range parity got stronger, not weaker**: 13,233/13,233 frames,
  mean IoU 1.000000, minimum 0.999999, max score delta 5.364e-7, zero
  one-sided detections, 8,023 frames bit-identical across runtimes.
- **Mesh parity holds at the same magnitude**: mean NME 7.384e-7 (stage-3:
  6.868e-7), worst 2.090e-6. The stage-3-derived 2.0e-6 bound was pinned to
  that corpus's noise and a 26x longer tail poked past it, so the external
  mode carries its own 5e-6 bound; the run also caught that YuNet accepts a
  face on every single LFW frame. The iris tail is the loosest region:
  worst iris-only NME 1.634e-5, an order above the body but still under a
  thousandth of a pixel at LFW eye distances.
- **The blendshape/EAR relationship reproduces on a 60x larger, fully
  independent corpus**: pooled Pearson r = -0.916 (stage-3: -0.938), all
  outputs finite. Still observational; LFW is also an open-eye set.
- **Latency on the Ryzen** (`2026-08-06-lfw-mp-latency-archhost.csv`):
  the machine is far steadier than the Zenbook (p50-to-p95 bands of a few
  hundredths of a ms), and here ort wins the SHORT-RANGE row outright
  (1.01ms vs 2.02/1.22/0.82ms native at 1/2/4 threads), a reversal of the
  Zenbook ordering worth knowing before quoting either machine as "the"
  number. The mesh ordering, the one that feeds the switch decision, holds
  on both machines: native 4.58ms at 2 threads vs 7.36ms ONNX, and
  blendshapes costs 1.41ms.

| Stage call (archhost) | Runtime | Threads | Mean ms | p50 | p95 |
|---|---|---|---|---|---|
| YuNet detect | ort | default | 10.09 | 9.89 | 11.08 |
| Short-range Blaze | ort | default | 1.01 | 0.98 | 1.29 |
| Short-range Blaze | tflite | 4 | 0.82 | 0.82 | 0.83 |
| Full-range Blaze | tflite | 2 | 3.28 | 3.22 | 4.13 |
| Mesh 468 (shipped) | ort | default | 7.36 | 7.36 | 7.40 |
| Mesh 478 (native) | tflite | 2 | 4.58 | 4.58 | 4.62 |
| Mesh 478 (native) | tflite | 4 | 3.08 | 3.07 | 3.12 |
| Blendshapes | tflite | 1 | 1.41 | 1.41 | 1.42 |

## Verdict, per pipeline stage

**Primary detection: YuNet stays.** No MediaPipe detector was benchmarked as
a YuNet replacement and none is a candidate: the 2026-08-05 stage-3 bench has
YuNet at 100% on RGB everywhere, and the rescue slot exists for exactly the
frames it drops.

**Rescue detection: full-range is the measured winner, and the stage stays
closed anyway.** Full-range detects 100% of the far-IR frames short-range
misses 9 of 9 on (2026-08-05 bench) at an affordable 4.3ms against the
short-range rows' 1.2 to 2.8ms.
That ranking changes nothing today: the rescue slot is grant-capable, so
flipping `Stage::Detection` open waits on an end-to-end false-grant corpus
(prints, screens, other faces carried to the auth outcome), not on any
detection accuracy number. Recorded on #276; re-deciding this on accuracy is
the exact mistake the #299 review caught.

**Landmarks: switch to the native 478-pt .tflite.** The evidence is now
complete on all three axes the decision needed. Fidelity: the two models
agree to mean NME 6.9e-7, and both parity harnesses run as enforcing
regression gates. Latency: native at 2 threads beat the shipped ONNX mesh on
BOTH measured machines (Zenbook 5.79 vs 10.34ms; archhost Ryzen 4.58 vs
7.36ms) with tighter tails; 2 threads matches the precedent
`FullRangeBlaze` set. Supply chain: the switch
deletes a conversion step nobody can reproduce from the shipped artifact
alone and replaces it with Google's published file, byte-pinned, on the
runtime irlume already bundles and verifies. The iris points are NOT an argument
for the switch (the shipped conversion also emits them, measured; an earlier
revision claimed otherwise), and the parity gate now covers all 478 points.
Packaging cost:
`face_landmarks_detector.tflite` (2.5MB) joins the model set on every lane,
and the mesh loader grows a TFLite path behind the same pin discipline.
Recommendation: do it as its own PR with the mesh_parity gate as the
acceptance test.

**Short-range Blaze runtime: no move needed.** The conversion is proven
faithful to 2e-6, so the ONNX copy is not a correctness risk. If the
landmarks switch lands and the rescue slot later moves to full-range, the
ONNX blaze and its conversion step retire naturally; switching it alone buys
1.7ms in a slot that only runs when YuNet already failed.

**Blendshapes: not evaluated as an authentication cue.** The model runs, at
an affordable cost, and its open-eye baseline is now on record; no
authentication conclusion follows from an all-open-eye corpus. Before it can
even be evaluated as a consent-gesture or liveness signal it needs a
labeled blink corpus with controlled eyelid closures (#316); the iris
parity question is settled (worst iris NME 1.249e-6). Until the corpus
exists it is not a candidate for wiring.

## Not measured

- Mini PC (N100) latency; the Zenbook and archhost are both measured and
  disagree on the short-range ordering, so a third machine would not be
  redundant.
- Open/closed blendshape separation (no closed-eye frames in the corpus).
- Multi-thread determinism is now answered for the mesh: mesh_parity at
  the production thread count (2) reproduces the 1-thread CSV
  byte-identically on this machine. The other tflite models' recorded
  parity runs remain single-threaded.
- Full-range Blaze accuracy beyond the 2026-08-05 corpus; nothing new today.

## Future optimization candidates, unbenchmarked

- The sparse full-range Blaze variant on the Face Detector page claims a
  roughly 60% smaller file, but its graph carries a Densify op the current
  decoder path has never run; it would need its own parity gate.
- Thread counts above 4 were not swept, and the 4-thread mesh numbers
  already show diminishing returns against 2 (3.92ms vs 5.79ms mean for a
  doubled thread budget).
- Google publishes no quantized (int8) variants of these face models on the
  solution pages, so the float16 files stay the only supply-chain-clean
  option.
