# README for western_blot_matching_SD_pt2 benchmarking

## Datasets used for benchmarking:
WBcheck_native_10.1038_s44319-026-00694-8: 
    - 5 figures; each has at least one Western Blot panel or more. 
    - All source data files are provided in the source_data folder 
    - the content of the source_data folder matches the figure.

WBcheck_missingSD_10.1038_s44319-026-00694-8:
    - 5 figures each has at least one Western Blot panel.
    - Some source data files are missing in the source_data folder. 
        Figure 2: Source data figure 2E is missing (where 2BDEFGH should be present)
        Figure 5: Source data figure 5F is missing (where 5C and 5F should be present)
        Figure 7: Source Data figure 7C is missing (where only fig 7C is a western blot.)
    - the provided content matches the figure. 

WBcheck_contentFailMissingSD_10.1038_s44319-026-00694-8:
    - 5 figures each has at least one Western Blot panel
    - All western blots have source data files in the source_data folder
    - Within the source data files the source data for one band is missing while other bands are present
        Figure 1C: source data for the flag-IP raw blots are missing. Two parts of the blot are not accounted for. 
        Figure 2G: source data for the actin part is missing. 
        Figure 5F: source data for actin is missing.
        Figure 6A: source data for CNTM4 is missing.
        Figure 6F: source data for lysate flag is missing.

WBcheck_contentFail_label_switch_10.1038_s44319-026-00694-8:
    - 5 figures each has at least one Western Blot panel
    - All western blots have source data files in the source_data folder
    - Within the source data files the labels of the raw blots are switched:
        Figure 2B:source data; the labels of CMTM6 and CMtM4 are switched
        Figure 5F: source data; the labels of cleaved caspase 8 and caspase 8 are switched
        Figure 6H: source data; the labels of flag and actin are switched
        Figure 7C: source data; the labels of lysate and alpha ip are switched

WBcheck_missingLabels_10.1038_s44319-026-00694-8: 
    - 5 figures each has at least one Western Blot panel
    - All western blots have source data files in the source_data folder
    - Within the source data files some of the labels are missing
        Figure 2H: in the source data files all the labels of the protein bands are missing
        Figure 6G: in the source data file the label of CMTM6 is missing
        Figure 7C: in the source data files the labels of the bands and the lanes are missing
