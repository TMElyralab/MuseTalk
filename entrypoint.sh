#!/bin/bash

echo "entrypoint.sh"
whoami
which python
source /opt/conda/etc/profile.d/conda.sh
conda activate musev
which python
python app.py
