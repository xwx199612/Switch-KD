#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
This helper has been deprecated.

The project no longer stores offline teacher logits or switch logits.
Use:

  vlm-distill teacher-precompute --config <config>

to write teacher labels only, then run:

  python -m vlm_distill.train_online_align_dbild --config <config> --max-steps 1

Online DBiLD computes teacher/student logits during training and does not save
them to disk.
EOF

exit 1
