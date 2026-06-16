
from soda_mmqc.scripts.visualize import *

MODEL = "gpt-5-mini-2025-08-07" # "claude-sonnet-4-20250514" #"gpt-4o-2024-08-06"  #o3-mini-2025-01-31  # o4-mini-2025-04-16

# Figure checklist tests dict
fig_similarity_dict = {
  "error-bars-defined": {
    "main-check": "longest_common_subsequence",
    "subchecks": {
      "panel_label": "longest_common_subsequence",
      "error_bar_on_figure": "longest_common_subsequence",
      "error_bar_defined_in_caption": "longest_common_subsequence",
      "error_bar_definition": "longest_common_subsequence"
    }
  },

  "individual-data-points": {
    "main-check": "longest_common_subsequence",
    "subchecks": {
      "panel_label": "longest_common_subsequence",
      "plot": "longest_common_subsequence",
      "average_values": "longest_common_subsequence",
      "individual_values": "longest_common_subsequence"
    }
  },

  "method-is-mentioned": {
    "main-check": "longest_common_subsequence",
    "subchecks": {
      "panel_label": "longest_common_subsequence",
      "method-identifiable": "longest_common_subsequence",
      "method_source": "longest_common_subsequence",
      "from_the_caption": "longest_common_subsequence"
    }
  },

  "micrograph-scale-bar": {
    "main-check": "longest_common_subsequence",
    "subchecks": {
      "panel_label": "longest_common_subsequence",
      "micrograph": "longest_common_subsequence",
      "scale_bar_on_image": "longest_common_subsequence",
      "scale_bar_defined_in_caption": "longest_common_subsequence",
      "from_the_caption": "longest_common_subsequence",
      "scale_bar_defined_in_image": "longest_common_subsequence",
      "from_the_image": "longest_common_subsequence"
    }
  },

  "micrograph-symbols-defined": {
    "main-check": "longest_common_subsequence",
    "subchecks": {
      "panel_label": "longest_common_subsequence",
      "micrograph": "longest_common_subsequence",
      "symbols": "longest_common_subsequence",
      "symbols_defined_in_caption": "longest_common_subsequence",
      "from_the_caption": "longest_common_subsequence"
    }
  },

  "n-larger-two": {
    "main-check": "longest_common_subsequence",
    "subchecks": {
      "panel_label": "longest_common_subsequence",
      "is_numerical_plot": "longest_common_subsequence",
      "shows_averaged_data": "longest_common_subsequence",
      "n_value": "longest_common_subsequence",
      "n_larger_than_two": "longest_common_subsequence"
    }
  },

  "panel-image-matches-caption": {
    "main-check": "semantic_similarity",
    "subchecks": {
      "panel_label": "longest_common_subsequence",
      "image_matches_caption": "semantic_similarity",
      "discrepancies": "semantic_similarity"
    }
  },

  "panelisation-and-classification": {
    "main-check": "longest_common_subsequence",
    "subchecks": {
      "panel_label": "longest_common_subsequence",
      "is_a_micrograph": "longest_common_subsequence",
      "is_a_nummerical_data_plot": "longest_common_subsequence",
      "is_a_scheme": "longest_common_subsequence",
      "is_mixed": "longest_common_subsequence"
    }
  },

  "plot-axis-units": {
    "main-check": "longest_common_subsequence",
    "subchecks": {
      "panel_label": "longest_common_subsequence",
      "is_a_plot": "longest_common_subsequence",
      "units_provided": "longest_common_subsequence",
      "justify_why_units_are_missing": "semantic_similarity",
      "unit_definition_as_provided": "longest_common_subsequence"
    }
  },

  "plot-gap-labeling": {
    "main-check": "longest_common_subsequence",
    "subchecks": {
      "panel_label": "longest_common_subsequence",
      "is_a_plot": "longest_common_subsequence",
      "gaps_defined": "longest_common_subsequence",
      "gap_description": "semantic_similarity",
      "justify_why_gaps_are_missing": "semantic_similarity"
    }
  },

  "replicates-defined": {
    "main-check": "longest_common_subsequence",
    "subchecks": {
      "panel_label": "longest_common_subsequence",
      "involves_replicates": "longest_common_subsequence",
      "number_of_replicates": "longest_common_subsequence",
      "type_of_replicates": "longest_common_subsequence"
    }
  },

  "single-channel-for-overlay": {
    "main-check": "longest_common_subsequence",
    "subchecks": {
      "panel_label": "longest_common_subsequence",
      "multicolored_overlay_micrograph": "longest_common_subsequence",
      "number_of_channels_in_overlay": "longest_common_subsequence",
      "number_of_single_channel_images_shown": "longest_common_subsequence",
      "all_channels_shown_separately": "longest_common_subsequence"
    }
  },

  "stat-significance-level": {
    "main-check": "longest_common_subsequence",
    "subchecks": {
      "panel_label": "longest_common_subsequence",
      "is_a_plot": "longest_common_subsequence",
      "significance_level_symbols_on_image": "longest_common_subsequence",
      "significance_level_symbols_defined_in_image": "longest_common_subsequence",
      "significance_level_symbols_defined_in_caption": "longest_common_subsequence",
      "from_the_caption": "longest_common_subsequence"
    }
  },

  "stat-test": {
    "main-check": "longest_common_subsequence",
    "subchecks": {
      "panel_label": "longest_common_subsequence",
      "is_a_plot": "longest_common_subsequence",
      "statistical_test_needed": "longest_common_subsequence",
      "statistica_test_mentioned": "longest_common_subsequence",
      "justify_why_test_is_missing": "semantic_similarity",
      "from_the_caption": "longest_common_subsequence"
    }
  }
  ,
  "panel-data-replication-validation": {
    "main-check": "semantic_similarity",
    "subchecks": {
      "panel_label": "longest_common_subsequence",
      "plot_type": "longest_common_subsequence",
      "image_matches_caption": "semantic_similarity",
      "replicable_from_data": "longest_common_subsequence",
      "image_matches_data": "semantic_similarity",
      "discrepancies": "semantic_similarity"
    }
  },

  "panel-label-detection": {
    "main-check": "longest_common_subsequence",
    "subchecks": {
      "panel_label": "longest_common_subsequence"
    }
  },

  "SD-mapping": {
    "main-check": "longest_common_subsequence",
    "subchecks": {
      "panel_label": "longest_common_subsequence",
      "source_data_filenames": "longest_common_subsequence"
    }
  },

  "western-blot-matching-SD": {
    "main-check": "longest_common_subsequence",
    "subchecks": {
      "panel_label": "longest_common_subsequence",
      "is_western_blot": "longest_common_subsequence",
      "source_data_matches_figure": "longest_common_subsequence",
      "discrepancies": "semantic_similarity"
    }
  }
}

fig, df = checklist_visualization(
    "fig-checklist",
    metric="semantic_similarity",  # "semantic_similarity", "perfect_match" longest_common_subsequence
    model=MODEL,
    aggregation_level=0,
    theme="light"
)
fig.show()