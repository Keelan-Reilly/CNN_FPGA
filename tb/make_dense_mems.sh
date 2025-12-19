#!/usr/bin/env bash
set -euo pipefail

# CASE0
cat > tb_fc_w_case0.mem <<'EOF'
0001
0000
0000
0000
0000
0000
0000
0000
0000
0001
0001
0000
0000
0000
0000
0000
0000
0000
0000
0002
0000
0000
0000
0000
0001
0001
0001
0001
0000
0000
0000
0000
ffff
0000
0000
0000
0000
0000
0000
0000
EOF

cat > tb_fc_b_case0.mem <<'EOF'
0002
ffff
0003
0000
0005
EOF

# CASE1 (+SAT)
python3 - <<'PY'
OUT_DIM=5; IN_DIM=8
with open("tb_fc_w_case1.mem","w") as f:
  for _ in range(OUT_DIM*IN_DIM): f.write("7fff\n")
with open("tb_fc_b_case1.mem","w") as f:
  for _ in range(OUT_DIM): f.write("7fff\n")
PY

# CASE2 (-SAT)
python3 - <<'PY'
OUT_DIM=5; IN_DIM=8
with open("tb_fc_w_case2.mem","w") as f:
  for _ in range(OUT_DIM*IN_DIM): f.write("8000\n")
with open("tb_fc_b_case2.mem","w") as f:
  for _ in range(OUT_DIM): f.write("8000\n")
PY

echo "Wrote dense mem files:"
ls -1 tb_fc_w_case*.mem tb_fc_b_case*.mem
