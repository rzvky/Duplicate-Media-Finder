# Model folder

The AI model (`clip_vitb32_visual_w8.onnx`, 96,359,160 bytes) is not stored in git because of its size.
It is attached to every release as 4 parts instead.

1. Download `DuplicateMediaFinder_AI_part1of4.bin` to `DuplicateMediaFinder_AI_part4of4.bin` from the
   [latest release](https://github.com/rzvky/Duplicate-Media-Finder/releases/latest).
2. Leave them in your Downloads folder, or put them in the tool folder (or this folder).
3. Start the tool. It joins the parts into the model file here, checks the SHA-256 and removes the parts.

SHA-256 of the joined model: `c2580cf93f5fe7425cdde1c42b1d3c4221cb7449decf0a0b619399423af9a257`
