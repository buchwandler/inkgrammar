# Image import

## Why layout-aware labelling?

Connected-component detection can find ink, but a historical scan does not intrinsically say
"this crop is uppercase A". Generic OCR is also a poor source of authority for unusual school
scripts.

For alphabet plates, the reliable workflow is:

1. declare the known row sequence;
2. segment each row automatically;
3. assign the declared labels;
4. inspect `review.png`;
5. correct the layout or detections before accepting them as observations.

Example:

```bash
inkgrammar page extract scan.png \
  --layout examples/layouts/sas-1968-alphabet.yaml \
  --output work/sas1968-alphabet
```

Output:

```text
work/sas1968-alphabet/
  detections.json
  review.png
  uppercase-1-00-U+0041.png
  ...
  lowercase-1-00-U+0061.png
  ...
```

`detections.json` records that the **shape** was detected from the image while the **identity**
came from the layout sequence.

Rows may optionally contain reviewed normalized `centers`. These are useful when wide historical
forms overlap horizontally or contain large internal white spaces (for example `M`/`N`), where a
pure whitespace splitter can choose the wrong boundary. The extractor still tightens each crop
to detected ink; the centers only provide a stable ownership region for each declared label.

The supplied SAS-1968 layout includes reviewed centers for this reason.

## OCR can be added later

An OCR classifier can later propose labels for arbitrary pages, but its output should be stored
as a hypothesis with confidence and human review. It should not silently become canonical data.


## Dataset extraction strategies

Page extraction is split into reusable pieces (`threshold_source`,
`crop_region`, `tighten_to_ink`, `segment_row`, `render_review`,
`write_detection_manifest`) with three strategy extractors:

- `labeled-rows-v1`: declared label rows (`PageLayout` v1) segmented into
  cells; identity always comes from the declared sequence.
- `region-layout-v1`: declared region records (`RegionLayout`, normalized
  coordinates) extracted intact, never cut into glyphs.
- `annotated-region-v1`: like `region-layout-v1` but keeps declared `ink`
  subregions separate from printed annotation marks (numbers, arrows), so
  those marks never enter the target ink.

## Isolation stage

Raw crops are candidates, not fitting truth. For each labeled cell the
materializer proposes a target-ink isolation: the connected component nearest
the reviewed layout center, rejecting components that sit closer to a
neighboring label center (cursive letters touch). A cell-ownership fence at
the midpoint to adjacent label centers keeps crossing ink out of the target
mask. Every rejected component is recorded with its reason in an isolation
proposal and a review overlay; the result stays `isolation_status: pending`
until human review accepts it.
