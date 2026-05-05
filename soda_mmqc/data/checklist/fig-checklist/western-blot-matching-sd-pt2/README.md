# western-blot-matching-sd-pt2

This check verifies that provided source data files for western blots (or similar gel-based data) match what is shown in the figure.

## Providing source data

Place source data **image** files under the figure's `content/source_data/` folder. Two layouts are supported.

### Option 1: Panel subfolders (recommended)

Use one subfolder per figure panel; the folder name is the panel label (e.g. A, B, 1G). The model receives images grouped by panel.

```
content/
  caption.txt
  <figure image>.(png|jpg|jpeg|tiff|webp)
  source_data/
    A/
      uncropped_blot.png
      scan.jpg
    B/
      raw_lanes.tif
```

### Option 2: Flat files

Put all images directly in `source_data/`. They are sent as a single group labelled "Source data files:".

```
content/
  source_data/
    uncropped_blot.png
    panel_A_scan.jpg
```

Supported image formats: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`.  
If `source_data/` is missing or has no images, the check runs using only the figure and caption.
