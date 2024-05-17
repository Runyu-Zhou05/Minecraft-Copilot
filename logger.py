import rich
import sys

def info(*args, **kwargs):
    rich.print(f'[cyan]{sys.argv[0]}[/cyan]:', *args, **kwargs)

def question(*args, **kwargs):
    rich.print(f'[pink]{sys.argv[0]}[/pink]:', *args, **kwargs)

def error(*args, **kwargs):
    rich.print(f'[red]{sys.argv[0]}[/red]:', *args, **kwargs)

def error_exit(*args, **kwargs):
    error(*args, **kwargs)
    exit(1)