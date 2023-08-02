import os
import sys

# Get the root directory of the project
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

# Add the project root directory to the Python path
sys.path.insert(0, PROJECT_ROOT)
