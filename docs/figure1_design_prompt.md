Design the main figure (Figure 1) for an ICRA robotics paper on **Q3C**, an
imitation-learning method. It is a schematic diagram, not a chart — no data
plotting. Output a self-contained HTML page I will export to PDF and place in a
LaTeX two-column paper.

## The one message the figure must land

Implicit Behavioural Cloning picks an action by scoring **16,384 uniformly
sampled candidates** at every control step. Q3C learns a small network that
**proposes 20 candidates**, scores those, and takes the best — **21 network
evaluations instead of 49,152**, at equal or better task success. The figure
should make that contrast immediate, before any text is read.

## What Q3C actually does

Two networks, trained together on expert demonstrations only (no rewards).

**Control-point generator** `f_φ(s)` maps a state to N = 20 candidate actions
("control points"), squashed by tanh so they always lie inside the action box.

**Q-estimator** `Q_θ(s,a)` scores a single (state, action) pair.

*Training*, per batch:
1. The generator emits its 20 control points for the state.
2. An MSE term pulls only the **nearest** control point toward the expert
   action — not the mean, which would collapse the cloud onto the average of
   multimodal demonstrations. A separation term pushes the remaining points
   apart so they spread over the action space.
3. The top-k control points that the critic currently scores highest are taken
   as **hard negatives** (with a stop-gradient), alongside uniform random
   actions, Langevin-refined actions, and noisy copies of the expert action.
4. The critic is trained by InfoNCE: the expert action is the positive, all of
   the above are negatives.
5. The generator receives that **same InfoNCE term with the opposite sign**,
   with the critic frozen. It is therefore adversarial — it is pushed to
   propose candidates that compete with the expert, while the MSE term stops
   the cloud from drifting away. The equilibrium surrounds the expert action
   with the hardest negatives the generator can find.

*Inference*: emit the 20 control points, score them, take the argmax. Optionally
refine with a few iterations of derivative-free optimisation, though 0–5 is
enough and more hurts on real hardware.

## Layout

Two bands, sharing one horizontal flow left to right.

- **Top band — training.** state → generator → the control-point cloud, with the
  expert action marked distinctly. Show the two forces on the cloud (one point
  pulled to the expert, the others pushed apart), the negatives feeding into the
  critic, and the sign-flipped arrow returning from the critic to the generator.
  That opposing arrow pair is the conceptual heart; make it legible.
- **Bottom band — inference.** the same cloud, scored, argmax, action to the
  robot. Beside it, a small visual comparison against IBC's uniform cloud, with
  the two evaluation counts (49,152 vs 21) as the payoff.

Render the action space as a small 2-D box in both bands so the reader sees the
same object move through the pipeline: dense grey dots for IBC's uniform
sampling, ~20 clearly separated marks for Q3C's proposal.

## Constraints

- Aspect ratio about **3.4 : 1** (it spans both columns: ~7 in wide, ~2 in tall).
  Design at 1400 × 410 px or a multiple.
- **Must survive greyscale printing.** Never encode meaning in hue alone — use
  shape, fill, dash pattern and position too. Assume a reader with a black and
  white printer.
- Legible when the whole figure is 7 inches wide on paper. Smallest text no
  finer than ~7 pt at that size; nothing thinner than about 1 pt.
- Restrained palette: one accent (#2a78d6, matching the paper's other figures),
  greys for everything else, white background. No gradients, no shadows, no
  3-D, no icons-as-decoration.
- Maths set as maths: `s`, `a*`, `Q_θ(s,a)`, `f_φ(s)`, `N = 20`. Inline SVG or
  styled HTML, no external fonts or CDNs.
- Label every arrow with what flows along it. Two or three short callouts
  maximum; the caption carries the detail, not the figure.
- Self-contained HTML: inline CSS and SVG only, no scripts, no network requests.

## What to avoid

A generic ML block diagram — boxes labelled "encoder / decoder / loss" joined by
identical arrows. The interesting content is specifically: the cloud of
candidates, the two opposing forces acting on it, and the order-of-magnitude
drop in the number of scored candidates. If those three are not immediately
visible, the figure has failed.
