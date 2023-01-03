#!/bin/bash
cd ../docs
sphinx-build -b html . _build/html
sphinx-build -b latex . _build/latex

