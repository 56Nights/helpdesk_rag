"""Answer a helpdesk question from the terminal.

    python cli.py "my laptop won't connect to wifi"
    python cli.py --retriever tfidf --k 3 "printer not responding"
"""
from __future__ import annotations

import argparse

from rag.config import Settings
from rag.pipeline import HelpdeskRAG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+", help="the problem to look up")
    parser.add_argument("--retriever", default=None, choices=["hybrid", "embedding", "tfidf"])
    parser.add_argument("--k", type=int, default=None, help="ticket groups to retrieve")
    parser.add_argument("--csv", default=None, help="path to a ticket export")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.retriever:
        settings.retriever = args.retriever
    if args.csv:
        settings.data_csv = args.csv

    bot = HelpdeskRAG(settings).load()
    print(f"{bot.kb.source} · {len(bot.kb)} problems · {bot.model or 'no model'}\n")
    if bot.note:
        print(f"note: {bot.note}\n")

    answer = bot.answer(" ".join(args.question), top_k=args.k)
    print(answer.text)

    if answer.sources:
        c = answer.confidence
        print(f"\n-- evidence ({c.level}: {c.reason})")
        for src in answer.sources:
            print(f"   {src['score']:.2f}  {src['issue']}  ({src['n_tickets']} tickets)")
            for sol in src["solutions"][:3]:
                print(f"         {sol['share']:>4.0%}  {sol['text']}")


if __name__ == "__main__":
    main()
