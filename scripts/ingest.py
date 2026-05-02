"""CLI script for document ingestion into the vector store.

Usage:
    python -m scripts.ingest --source data/raw/ --reset
"""

import click
from rich.console import Console

console = Console()


@click.command()
@click.option(
    "--source", "-s",
    default="./data/raw",
    type=click.Path(exists=True),
    help="Source directory with parsed KB articles.",
)
@click.option("--reset", is_flag=True, help="Reset vector store before ingestion.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def main(source: str, reset: bool, verbose: bool) -> None:
    """Ingest KB articles into the vector store (Phase 3 - not yet implemented)."""
    console.print("[yellow]Ingestion pipeline not yet implemented (Phase 3).[/yellow]")
    console.print(f"  Source: {source}")
    console.print(f"  Reset: {reset}")


if __name__ == "__main__":
    main()
