import os
import sys
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(BASE_DIR, os.pardir))
import sys
from deep_insight_face import cli
import sys

if __name__ == "__main__":
    sys.exit(cli.main())