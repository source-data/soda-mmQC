"""Scripts that checks a checklist for consistency. Performs the following checks:
- consistency between the prompt and the schema
- correct layout of the directories and files
- consistent naming of the files and "name" attributes.
- schema is valid for the OpenAPI schema format."""

import os
import json
from pathlib import Path

