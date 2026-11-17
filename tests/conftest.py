# Put the project root on sys.path so tests can import extraction.branch_b.*
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
