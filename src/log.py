"""Shared rich-based pretty logging for the pipeline."""
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()


def section(label: str) -> None:
    """Top-level section header — used by run_all between major stages."""
    console.print()
    console.print(Panel.fit(f"[bold white]{label}[/]", border_style="bold cyan"))


def subsection(label: str) -> None:
    """Sub-stage header inside a module's main()."""
    console.print()
    console.print(Rule(f"[bold magenta]{label}[/]", style="magenta", align="left"))


def step(label: str) -> None:
    """Discrete step within a subsection (e.g. 'Generating script 2/5')."""
    console.print(f"[bold blue]▸[/] {label}")


def info(msg: str) -> None:
    console.print(f"  [dim]· {msg}[/]")


def success(msg: str) -> None:
    console.print(f"  [bold green]✓[/] {msg}")


def warn(msg: str) -> None:
    console.print(f"  [bold yellow]![/] {msg}")


def error(msg: str) -> None:
    console.print(f"  [bold red]✗[/] {msg}")


def status(label: str):
    """Context manager: animated spinner with label. Use during long ops."""
    return console.status(f"[cyan]{label}[/]", spinner="dots")
