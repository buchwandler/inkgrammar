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
scriptstrokes page extract scan.png \
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
