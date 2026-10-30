# Content rights

Body Double's own code is Apache-2.0 (see LICENSE / NOTICE). Model weights are not redistributed; each keeps
its upstream license (MODELS.md). This file records the basis for the **demo footage and characters** used in
the challenge entry.

## Demo assets

| Asset | Used in | Basis |
|---|---|---|
| Breakdance plate (source clip) | `media/body_double_demo.mp4`: the first 30 s of Mixkit 452 "Man breakdancing", 1920×1080, 23.976 fps, no audio | [Mixkit](https://mixkit.co/free-stock-video/man-breakdancing-452/) Stock Video Free License: commercial and personal use, no attribution required, no redistribution of the clip as stock. The source clip itself is not in this repository. |
| Cast character "noir" (orange suit) | The demo and the explainer | An original fictional character, not a real person. The portrait was generated with google/nano-banana-2.1 (Replicate) from a text prompt only, with no photo input; the full-body view and the orange suit are Nano Banana edits of those images. The cigarette was removed locally with FLUX.2 Klein 4B. Made outside Body Double, which makes no outbound requests. |
| Dance plate (explainer only) | `media/body_double_explainer.mp4`: an excerpt of the GENER8ION/STORM dance video, with audio | Ownership and permission for public demonstration, including its music/audio, have not been established in this project. |

## Rules applied

- Only footage and characters we own or have explicit permission to use are eligible for the public demo and
  submission. If a basis above is wrong, stop and swap the plate/cast before publishing — do not ship an
  ambiguous entry.
- Outputs are AI-generated and must carry an AI-generated label wherever they are published (see NOTICE).
- The app itself enforces no telemetry and makes no outbound requests; content licensing is the operator's
  responsibility when they run it on their own media.

## Before making this repository public

The explainer still shows the STORM plate. Rebuild it on the breakdance take (or another cleared plate), or
confirm the STORM rights in writing, and update this table in the same commit.
