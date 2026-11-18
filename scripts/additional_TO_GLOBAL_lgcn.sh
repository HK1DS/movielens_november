#!/usr/bin/env bash
# Additional request 2026-11-18 (see ADDITIONAL.md): TO_GLOBAL IKGR_lgcn_dyn seeds 2021, 2022 on the rented RTX 3090.
# Seed 2020 runs on Jinwook's PC, so do not add it here. Finished seeds are skipped on rerun.
#   bash scripts/additional_TO_GLOBAL_lgcn.sh            # PY=python3 SEEDS=2021 to change
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
SEEDS=${SEEDS:-2021,2022}

# Inputs: shared .inter (TO bundle) and the TO_GLOBAL intent KG (TO_GLOBAL bundle); both come with git pull.
[ -f data/kc_k30/ikgr-custom/ikgr-custom.inter ] || "$PY" scripts/unpack_data_bundle.py --split TO
[ -f run_k30_TO_GLOBAL/kg_pack.pt ] || "$PY" scripts/unpack_data_bundle.py --split TO_GLOBAL
"$PY" scripts/check_meta_kg.py

"$PY" -u run_pipeline.py --k 30 --split TO_GLOBAL --steps E --seeds "$SEEDS" --specs IKGR_lgcn_dyn

# Result zip for Jinwook (scripts/merge_results.py reads it as is).
"$PY" - <<'EOF'
import glob
import zipfile

out = "results_additional_TO_GLOBAL.zip"
files = ["run_k30_TO_GLOBAL/slice_eval_TO_GLOBAL_result.json"]
files += sorted(glob.glob("run_k30_TO_GLOBAL/per_user_TO_GLOBAL/IKGR_lgcn_dyn*"))
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f)
print(f"[done] {out}: {len(files)} files -> send this zip to Jinwook")
EOF
