import os, sys

from pylml.agentic import HackuityAgent, RAGAgent

# This is a test file. Run it with pyyel as an installed package.

for cls in [
    RAGAgent,
    HackuityAgent,
]:
    try:
        cls()
    except Exception as e:
        print(e)
