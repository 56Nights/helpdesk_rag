"""Thin wrapper so `python cli.py "..."` still works from a checkout.

The real implementation lives in rag/cli.py, which is what the installed
`helpdesk` command calls.
"""
from rag.cli import main

if __name__ == "__main__":
    main()
