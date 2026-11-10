# Phase H composition review

This source-free review records findings from the private overlays under ignored `work/sas-1968/overlays/`. No source imagery is copied here. The topology and transforms remain provisional, and the overlays are not approval to promote them.

## Underturn

The latest fit improved its training aggregate from `0.613130` to `0.548245`. The `i` member improved from `0.729239` to `0.497801`; `u` worsened from `0.497021` to `0.598690`. Its two component-local regions show similar but imperfect path alignment. The `w` hold-out remains diagnostic-only, with mean objective `0.828928`, a training-to-hold-out gap of `0.280683`, and a visibly poor local match. Keep the two-part `w` construction, but do not treat the shared primitive or its transforms as validated.

## Hump

After shifting the second `m` hump transform to align with its local crop, both fit members improved. The aggregate moved from `0.638877` to `0.572258`; `n` moved from `0.567215` to `0.511573`, and `m` from `0.710538` to `0.632944`. The `h` hold-out local score is `0.675105`, giving a training-to-hold-out gap of `0.102847`. The crop-local overlays are improved but still provisional.

## Ascender loop

The training aggregate improved from `0.531270` to `0.501043`; `l` moved from `0.492803` to `0.437086`, and `b` from `0.569737` to `0.564999`. Hold-outs remain poor: `k` scores `0.755977` and `h` scores `0.791303`, for a mean hold-out score of `0.773640` and a gap of `0.272597`. The private overlays show substantial mismatch in these hold-outs. Do not promote the shared loop based on its training aggregate; reassess the family split and per-glyph transforms before release use.

## Diagnostic interpretation

Reports include each member's objective terms, topology transform, residual map, anchor and endpoint terms, split summaries, and generalization gap. Current residual maps are empty; endpoint and anchor penalties are zero for these component-only fits. These diagnostics do not establish connector correctness. Entry/exit connectors and `b`/`k` continuations remain provisional, and hold-outs did not influence selected parameters.
