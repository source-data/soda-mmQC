---
title: "Figure checklist prompts (truncated)"
subtitle: "Task-focused summaries; JSON schema / boilerplate removed"
---

# 1. Panel matches caption


## Panel-image-matches-caption

## Summary 
Your task is to analyze a scientific figure to check whether each panel image matches what is described in the corresponding figure caption. This check ensures that the visual content of each panel accurately corresponds to the textual description provided in the caption.

## 1. Read the caption description for each panel
For each panel in the figure:
- Locate the section of the figure caption that describes that specific panel (typically indicated by the panel label, e.g., "(A)", "Panel A", etc.).
- Carefully read what the caption states about the panel's content, including:
  - Type of data shown (plot, micrograph, scheme, etc.)
  - Experimental conditions or treatments
  - Key features or elements that should be visible
  - Any specific labels, symbols, or annotations mentioned

## 2. Examine the panel image content
For each panel in the figure:
- Carefully examine the actual visual content of the panel image.
- Identify what is actually shown in the image:
  - Type of visualization (plot type, image type, etc.)
  - Visible elements, labels, symbols, annotations
  - Experimental conditions or treatments that can be inferred from the image
  - Any notable features or characteristics

## 3. Compare image with caption description
For each panel, systematically compare:
- Does the type of visualization match? (e.g., if caption says "bar chart", is it actually a bar chart?)
- Do the experimental conditions or treatments match what is described?
- Are the key features mentioned in the caption actually visible in the image?
- Do labels, symbols, or annotations mentioned in the caption appear in the image?
- Are there elements in the image that contradict the caption description?
- Are there elements mentioned in the caption that are missing from the image?

## 4. Determine match status
For each panel decide:
- "pass" if the panel image content accurately matches the caption description with no significant discrepancies.
- "fail" if there are clear discrepancies, contradictions, or missing elements between the image and caption.
- Set "image_matches_caption" to "not_applicable" if the panel cannot be meaningfully compared (e.g., the caption provides no specific description for that panel, or the panel type makes comparison impractical).

## 5. Document discrepancies
If "fail":
- provide a clear, concise description of the specific discrepancies found.
- Be specific about what differs between the image and caption (e.g., "Caption describes three conditions but image shows four", "Caption mentions error bars but none are visible", "Image shows a line plot but caption describes a bar chart").

\newpage

# 2. Method is mentioned


## Method-is-mentioned
## Summary
Your task is to examine a scientific figure and identify whether each panel includes clear statements or labels describing the experimental assays or methods used to generate the results.


## 1. Check the figure caption for an experimental method
For each panel determine whether the figure caption explicitly mentions the experimental assay or method used for that panel.
Examples include (but are not limited to):
“Western blot”, “immunohistochemistry”, “electron micrograph”, “PCR”, “RNA-seq”, “luciferase assay”, “flow cytometry”.

## 2. Check the panel image itself if the caption is unclear
If the caption does not explicitly mention a method, examine the panel image for clues such as:
- Axis labels (e.g., “relative luciferase activity”)
- Plot titles
- Embedded text within the panel image
- Also consider whether the panel clearly belongs to a previous panel and uses the same experimental method.

## 3. Decide whether the method is identifiable

If the experimental method or assay can be identified with reasonable confidence, set it to "pass".
If the method cannot be confidently identified, set it to "fail". Only return "pass" if you are very confident it is explicitly mentioned or clearly inferred from the panel image.
If the panel shows a schematic describing the experiment, please simply put "n/a".

**If you cannot identify the experimental method or assay type, do not invent or guess methods.**


\newpage

# 3. Error bars defined


## error-bars-defined

## Summary 
Your task is to analyze a scientific figure and its caption to check for the presence of error bars and whether they are properly defined.

## 1. Determine if the panel contains error bars
Your job is to pay attention to any plots that have error bars (typically bar charts, line plots). This is easy when the plot is itself an individual panel image. Pay attention also to more difficult cases when a plot is only part of a composite panel image. For each panel in the figure:
- Determine if the plot in the panel image contains error bars (lines extending from data points indicating variability). These are typically on bar charts, line charts, sometimes scatter plots. Note: the whiskers on a box plot are NOT considered as error bars.

## 2. Check figure caption for explanation
- If error bars are present in the figure panel image, check if the text of the caption explains what they represent (e.g., standard deviation, standard error, confidence interval). If yes-> "pass". If no-> "fail"
- If error bars are *not* present on the figure, then, obviously, no explanation in the caption is needed. In this case indicate that error bars are not needed in the caption.


\newpage

# 4. Statistical test


## Stat-test
## Summary

Your task is to determine if the statistical test that was used to determine the significance levels, is mentioned in the figure caption. 

## 1. Determine if a panel contains plot of quatitative data and significance statements
Determine whether the panel is a plot of quantitative data (plots with XY or XYZ axes, bar charts, line charts, any scatter plots, including FACS analyses, histograms, pie charts, box plots, heatmaps, area charts, bubble charts, violin plots, radar or spider charts, treemaps, waterfall charts, funnel charts, dot plots, Sankey diagrams, contour plots, density plots, candlestick charts, polar charts, gauge charts, etc.). If it is not a plot of quantitative data, then there is no need to mention a statistical test and no need to justify anything on your side. If it is a plot of quantitative data but there is no mention of statistical significance, then there is no need to mention a statistical test and no need to justify anything on your side. The mere presence of error bars does NOT imply statistical analysis and does NOT require a statistical test.

## 2. Decide whether a statistical test must be mentioned
If, and only if, it is a quantitative plot, use the image of the panel and the text of the figure caption to understand whether a statistical test needs to be mentioned. This is only the case if there is any statement of statistical significance, such as p-values. It is then mandatory to state which statistical test was used.

## 3. Verify whether the test is named in the caption
If, and only if, statistical significance is indicated, verify whether the statistical test is explicitly mentioned in the caption. E.g. "Unpaired two-tailed Student’s t test", "Ordinary one-way ANOVA test", "Mann-Whitney test", "one-way ANOVA with Dunnett’s multiple comparisons test". Of course, if there is no need of a statistical test, then there is no need to verify or extract anything from the caption.

## 4. Justify a missing test when one is required
If the statistical test is missing when it should have been provided, explain in one sentence why you think it should be provided. Again, if there are no claims of statistical significance in the caption or on the plot, then there is no need to mention a statistical test.

## 5. Set pass / fail / n/a
If the statistical test is mentioned, set this check to "pass". If it is not mentioned, "fail"; and "n/a" if it is not applicable. 


\newpage

# 5. Statistical significance level


## Stat-significance-level

## Summary
Your task is to analyze a scientific figure to check for the presence of adequate statement about the level of statistical significance of the results displayed.

## 1. Determine whether the panel is a plot of quantitative data.
For each panel in the figure:
- Determine whether the panel contains a plot of numerical/quantitative data (e.g., bar charts, line plots, scatter plots with quantitative axes).
- Note: This check applies only to numerical plots. Micrographs, schemes, and other non-numerical visualizations are not relevant for this check.

## 2. Check for indication of statistical significance in panel image
For all panels identified in step 1. check for symbols on the plot that could indicate a level of statistical significance. These are typically, but not always, stars or asterisks aligned with the respective experimental groups or group comparisons. Sometimes the level of significance is indicated directly on the image, for example "p<0.001" or "p=0.05" written next to the symbol.

For each symbol found:
- Determine if the significance level is defined directly on the image (e.g., p-value shown next to the symbol, or explicit text like "p<0.001" on the plot).

## 3. Check for indication of statistical significance in figure caption
For symbols not defined on the panel image, check the text of the figure caption to see if they are clearly defined there.

## 4. Decide if information is complete
For each panel decide if all symbols of significance are defined. If yes-> "pass", if no-> "fail". If there are no significance symbols -> "n/a" 

\newpage

# 6. n larger than two


## N-larger-two

## Summary 
Your task is to analyze numerical plots that show averaged or aggregated data from multiple experimental repetitions to check whether the number of replicates (n) is larger than 2.

## 1. Determine if the panel contains a numerical data plot
For each panel in the figure:
- Determine whether the panel contains a plot of numerical/quantitative data (e.g., bar charts, line plots, scatter plots with quantitative axes).
- Note: This check applies only to numerical plots. Micrographs, schemes, and other non-numerical visualizations are not relevant for this check-> "n/a".

## 2. Check if the plot shows averaged/aggregated data
For each panel that is a numerical plot:
- Examine the plot to determine if it shows averaged or aggregated data from multiple experimental repetitions. Look for:
  - Error bars (lines extending from data points indicating variability, such as standard deviation, standard error, or confidence intervals)
  - Boxplots (showing quartiles, median, and outliers)
  - Bar charts with averages (bars representing mean values)
  - Violin plots (showing distribution shapes)
  - Any other visualization that represents aggregated data from multiple replicates
- If the plot shows such averaged/aggregated data go to step 3.
- If the plot shows individual data points only (e.g., scatter plots with all individual measurements visible), set "n/a".
- If the panel is not a numerical plot, set the check to "n/a".

## 3. Identify the number of replicates (n)
For each numerical plot that shows averaged/aggregated data:
- Search the figure caption for information about the number of replicates (n).
- Look for explicit statements such as "n = 2", "n = 3", "three independent experiments", "n = 4-5", etc.
- Also check the panel image itself for any embedded text indicating n values.
- If a specific n value is mentioned, extract it and provide it as a string (e.g., "2", "3", "5"). If a range is given (e.g., "n = 3-4"), use the lower bound for conservative assessment.
- If n is not mentioned anywhere, set the check to "fail".

## 4. Assess if n is larger than two
For each numerical plot:
- If the plot shows averaged/aggregated data AND n is mentioned (n_value is provided as a string like "2", "3", "5", etc.):
  - Compare the numeric value: If n > 2, set the check to "pass" (this is appropriate - averaged data requires n > 2 to be statistically meaningful).
  - If n ≤ 2 (n = 1 or n = 2), set the check to "fail" (this is not appropriate - averaged data should not be shown with n ≤ 2).


## 5. Important rules
- Only flag as "fail" (n not larger than two) when you are confident that n ≤ 2 AND averaged/aggregated data is shown.
- Be conservative: if n is unclear or ambiguous, use "fail" rather than guessing.
- Focus on the specific n value for the data shown with averaged representation. If different groups have different n values, assess each case separately if possible, or use the minimum n value.
- Remember: plots showing individual data points (all replicates visible) do not require n > 2, but plots showing averaged/aggregated data do.

\newpage

# 7. Replicates defined


## Replicates-defined

## Summary 
Your task is to analyze a scientific figure to check whether the type of replicates is explicitly defined in the caption.

## Background
The task here is to check whether the number and nature of replicates are clearly stated in the figure caption.
Generally speaking, replicates are used to estimate the source of variability in experimental results. This can be obtained by analyzing independent samples, running independent experiments, analyzing individual samples multiple times, etc... The nature of the replicates should be specified in the figure caption. For each panel, locate the corresponding description in the figure caption.

## 1. Check for mentions of number and nature of replicates 
For each panel in the figure:
- Check in the figure panel and also the corresponding figure caption to understand whether the experiment involves replicates. 

## 2. Decide if the information about the replicates is sufficient
- If the experiment replicates are defined set the check to "pass". If you cannot find any information about the type or number of replicates, set the check to "fail".

\newpage

# 8. Individual data points


## indiviual-data-points

## Summary
Your task is to analyze a scientific figure and its caption to verify that plot charts that display average values also display the actual individual data points.

## 1. Determine if the panel is a plot
Your job is to pay attention to any plots that have error bars (typically bar charts, line plots) or that show *mean values*. For each and every labeled panel in the figure decide:
- If the panel image is a plot, determine if it displays average values.
- If the panel image displays average values, check if the individual values are also displayed. If so set the check to "pass". If individual data points are missing set the check to "fail".
- If the panel image does not display average values or is not a plot, then, obviously, there is no need to check for individual data points ->"n/a".

## 2. Exceptions and special cases
- Include all panels visible in the figure, even if they don't contain error bars.
- for line-graphs, for example something is meassured over time, it is okay to not plot the individual data points as this wold make the line graph illegible and its commonly accepted to not plot he individual data points. In these cases put set the check to "n/a".  
- Similarly, if the n is very large (>200) it is ok to ommit the individual data points also in bar graphs and you can put "n/a". 
- Violin plots do not need individual data points as their shape indicates the measurement distribution, so please put "n/a"


\newpage

# 9. Plot axis units


## Plot-axis-units

## Summary
Your task is to analyze a scientific figure to check whether plot axes have defined units.

## 1. Determine if the panel is a quantitative plot
For each panel in the figure:
- Determine whether the panel is a plot of quantitative data with axes (XY or XYZ axes).
- If, and only if, it is a quantitative plot with axes, proceed to check for units. Otherwise, set "n/a".

## 2. Understand when units are needed
A unit is a standardized quantity used to express a measurement. Units should be clearly defined for any continuous variable that represents a physical or biological measurement.

**Units ARE needed for:**
- Physical measurements (e.g., meters, seconds, grams, moles, degrees Celsius)
- Biological measurements (e.g., cells/mL, copies/µL, concentration units)
- Arbitrary units (AU) - these MUST be explicitly labeled as "arbitrary units" or "AU"

**Units are NOT needed for:**
- Categorical variables (e.g., treatment groups, time points as categories, sample names)
- Unitless quantities such as:
  - Ratios and percentages (e.g., 0.5, 50%)
  - Fold changes (e.g., 2-fold increase)
  - Log ratios (e.g., log2 fold change)
  - Normalized values (e.g., normalized to control)
  - Z-scores or other standardized scores
  - Fluorescence intensity 

## 3. Check for units on each axis
For each quantitative plot, examine each axis (x-axis, y-axis, z-axis if present):
- First, check the panel image itself. Units are typically displayed next to the axis label (e.g., "Time (hours)", "Concentration (µM)").
- If units are not visible on the image, check the figure caption for unit definitions.
- For each axis, determine:
  - "pass": Units are provided (either on the image or in the caption)
  - "fail": Units are missing but needed (continuous variable without units)
  - "n/a": Units are not required (categorical or unitless variable)

## 4. Justify missing units
If units are missing (answer is "fail"):
- Provide a clear justification explaining why units are needed but missing.
- Be specific about what measurement is shown and why units are important (e.g., "protein abundance is plotted but units are not specified").

*IMPORTANT*: When extracting text from the caption, ONLY include the text that is specific to axis units. Do NOT include general descriptions of the figure or panel content.

\newpage

# 10. Plot gap labeling


## Plot-gap-labeling (Enhanced v5)

## Summary
Your task is to inspect scientific figure images and determine whether any axis breaks/gaps/discontinuities are present and whether they are clearly indicated with standard visual markers.

## 1. Is this a quantitative plot?
For each panel: decide whether the panel contains a quantitative plot (axes with numeric tick labels, histograms, scatter/line plots, bar charts with numeric axes).

## 2. Determine axis scale type
For each numeric axis, try to infer `scale_type` from tick labels and spacing:
- `linear` — evenly increasing labels (e.g., 0, 10, 20)
- `log` — multiplicative labels (e.g., 1, 10, 100) or labels that increase by powers of 10
- `categorical` — non-numeric or category labels
- `ambiguous` — cannot determine
If `scale_type == 'log'`, do NOT treat non-uniform tick spacing as an axis gap.

## 3. Detect labeled breaks (visual templates)
A labeled axis break should match one or more of these visual templates near the axis line. 
Templates (match any):
- Two short, parallel oblique/diagonal slashes crossing the axis (// or \\) placed close together and crossing the axis line.
- Zigzag/squiggle mark drawn across the axis (sawtooth or jagged break mark) that interrupts the axis line.
- Short double-tick markers or thick slash marks placed directly on the axis, often with a small gap in the axis line.
- A printed text label such as "break" or an obvious visual whitespace gap in the axis accompanied by a marker.
- For any of the above cases set the check to "pass"


## 4. Detect unlabeled breaks (heuristics)
If no labeled break is visible, apply the following measurable heuristics. 
Heuristics (flag when true):
- Numeric jump factor: consecutive tick labels show a multiplicative jump >= 4× (e.g., 10 → 40 or 5 → 25) without an axis-scale change or explanatory label.
- Tick-spacing ratio: physical distance between two adjacent tick marks differs from the previous interval by factor >= 3× (measure pixel distance along axis).
- Suspicious whitespace: a contiguous empty region along the axis occupying >= 20% of axis length with no ticks while ticks exist on both sides.
- Abrupt plotted-gap: plotted line/points show a discontinuous jump with no visual break marker at the axis (for example, a line that disappears and resumes far away on same axis scale).
- Numerical label anomaly: tick labels skip intermediate numeric decades without clear log-scale formatting or annotation (e.g., 10, 20, 30, 150).
- If you detect an unlabelled axis break set the check to "fail"


## 5. Edge cases and exceptions
- Log scales: non-uniform tick spacing is expected; set `scale_type: 'log'` and default `answer: 'not needed'` unless a visual labeled break is present.
- Dual/secondary axes: evaluate left and right y-axes independently; record separately.
- Decorative or caption-explained breaks: if the caption or visible text explicitly documents an axis discontinuity (e.g., "x-axis break between 50–100 omitted"), accept and mark `answer: 'yes'` for labeled break.
- Scientific notation: unequal spacing caused by notation (e.g., 1e-3 to 1e1) should be treated per `scale_type` rules.
- Legends, gridlines, or data-inset panels should not be mistaken for axis breaks. Ignore similarity of decorative patterns unless they cross the axis at the expected location.

\newpage

# 11. Micrograph scale bar defined


## Micrograph-scale-bar

## Summary
Your task is to analyze a scientific figure to check for the presence of a scale bar on micrographs and make sure they are properly defined either in the image itsself or in the caption.
## 1. Identify microscopy images and check for scale bars
For each panel in the figure:
- Determine whether the panel image is a micrograph, a picture of a microscopic sample, microscopy images. 
- If and only if it is a micrograph or microscopic image, check whether there is a scale bar in the image. A scale bar is a visual reference element added to scientific micrographs (microscopic images) that indicates the actual size of the objects being depicted. It typically appears as a line or bar of defined length. 

## 2. Identify if scale bar is defined in the image itsself or in the figure caption
In some cases the defined length of the scale bar is written in the image itsself and displayed as label such as "10 μm" or "500 nm" or it is defined only in the figure caption. For each microscopy image identified in step 1 check:
- If the scale bar is defined in the image itsself, set the check to "pass" . This would be indicated with a number and a unit next to the scale bar.  
- If the scale bar is defined in the figure caption, set the check to "pass". 
- If there is not definition or no scale bar at all set the check to "fail".



\newpage

# 12. Micrograph symbols defined


## Micrograph-symbol-defined

## Summary
Your task is to analyze a scientific figure to check for the presence of symbols such as arrows, arrowheads, and other annotations, on micrographs, stereomicrographs, and low-magnification images or other images derived from imaging techniques and make sure that these symbols are properly defined in the caption.
## 1. Identify panels of interest  
Identify all panel images that are potentially of interest because they are derived from imaging experiments. For each panel in the figure:
- Determine whether the panel image is a micrograph, a picture of a microscopic sample, microscopy images, low-magnification of a sample, or whole mount image. Apply the following steps *ONLY* to panels of interest and ommit any numerical data plots. 

## 2. Check for symbols in the image
For each panel image identified in step 1 check:
- If there are symbols such as arrows, arrowheads, circles, stars, dashed lines or other markings highlighting some specific structures or features on the image.

Detection rules (apply these strictly):
- Treat as a "symbol" any graphical overlay intended to call out image features: `arrow`, `arrowhead`, `caret`, `circle`, `ellipse`, `box`, `star`, `dashed line`, `solid line`, `cross`, `dot`, `triangle`, or similar shapes that are not part of the sample itself.
- *Ignore the scale bar* as these are not stricly annotations highlighting image coneten features. 
- For each symbol instance, record: shape (use the standardized names above), color (perceived: `white`, `black`, `red`, `green`, `blue`, `yellow`, or `other`)
- If there are multiple instances of the same shape but differing by color or location, report them as separate symbol entries.
- If color is ambiguous (e.g., due to low contrast or grayscale image), use `other`. 

- Identify the type of symbols AND specify their color (white, red, green, etc.). There can be multiple different types of symbols with different shapes and colors on a single image.
- *Please note:* Symbols on plots (for example stars to indicate statistical significance) can be ignored in this task. 

## 3. Identify symbol description in the figure caption
If such symbols are present, check whether the text of the caption explains what they represent or what features they are highlighting. If all symbols are described in the figure legend set the check to "pass". If one or more symbol is not defined set the check to "fail". If there are no symbols, then such explanations are not needed -> "n/a".


\newpage

# 13. Single-channel for overlay


## Single-channel-for-overlay

## Summary 
Your task is to evaluate each panel for the presence of overlays and corresponding single-channel images.

## Background

The goal of this check is to assess whether, when a multicolored (merged) microscopy image is shown, the corresponding single-channel images are also provided.

Single-channel images may be shown as:

grayscale images, or

colorized single-channel images

Step-by-step instructions (follow strictly)

For each panel:

## 1. Determine whether a multicolored overlay micrograph is present. 
Check if the panel contains a microscopy image with two or more distinct channels merged into one image.
If the panel does not contain a merged microscopy image (e.g., plots, schematics, brightfield images), answer "n/a".

## 2. Count the number of channels in the overlay
In case of multicolored overlay micrograph, count the number of distinct channels merged. Count only channels that are visually distinguishable and relevant.
If uncertain, choose the most conservative lower number. 

## 3. Count the number of single-channel images shown
Count how many separate single-channel images corresponding to the overlay are provided in the same panel. Do not count quantification plots or unrelated images. 

## 4. Determine completeness
If a multicolored overlay is present:Set this check to "pass" if the number of single-channel images is equal to or greater than the number of overlay channels. Otherwise set it to 'fail'.

## 5. Important rules

- If you are unsure whether an image corresponds to the overlay, do not count it. Be conservative: when in doubt, under-count rather than over-count.

