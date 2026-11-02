#!/usr/bin/env python
"""
Vela Framework - CLI Manager
Ponto de entrada principal do framework.

Uso:
    python manage.py runapp          # Inicia o app desktop
    python manage.py shell           # Abre o shell interativo
    python manage.py startapp <nome> # Cria um novo app
    python manage.py logs            # Exibe os logs do sistema
    python manage.py version         # Exibe a versão do framework
"""

import sys
import os

# Garante que o root do projeto esteja no sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from vela.cli.commands import CommandRunner


def main():
    runner = CommandRunner()
    runner.execute(sys.argv[1:])


if __name__ == "__main__":
    main()
