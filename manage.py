#!/usr/bin/env python
"""Gerenciador padrão de projetos Vela (gerado por vela startproject)."""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from vela.cli.commands import CommandRunner


def main():
    CommandRunner().execute(sys.argv[1:])


if __name__ == "__main__":
    main()
