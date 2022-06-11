#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR
source ~/.zshrc
exec python3 uci.py