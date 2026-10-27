import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.plots import generate_all_figures

if __name__ == "__main__":
    generate_all_figures()
