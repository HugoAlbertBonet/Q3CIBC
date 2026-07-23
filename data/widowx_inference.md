## Terminal 1 - Launch Docker Container

cd ~/bridge_data_robot
USB_CONNECTOR_CHART=$(pwd)/usb_connector_chart.yml docker compose up --build robonet


## Terminal 2 - Start Server

cd ~/bridge_data_robot
docker compose exec robonet bash -lic "widowx_env_service --server"


## Terminal 3 - Client Side

#### Activate env

conda activate q3c_deploy

#### Align the camera

cd ~/Q3CIBC
python scripts/align_pusht_camera.py

#### Run inference

##### Dry-run

python scripts/deploy_pusht_real.py \
    --seed-dir checkpoints/pusht_real_combinedv2/seed_0011 \
    --dry-run --device cpu

##### Run policy
python scripts/deploy_pusht_real.py \
    --seed-dir checkpoints/pusht_real_combinedv2/seed_0011 \
    --device cpu --steps XX


## Diagnostic 

cd ~/Q3CIBC

CKPT_ROOT=checkpoints/pusht_real_combinedv2_v2   # <-- point at the run you're testing
STAMP=$(date +%m%d_%H%M)

for d in "$CKPT_ROOT"/*/; do
    tag=$(basename "$d")
    echo "===== $tag ====="

    # 1. No-motion sanity: dumps deploy_dryrun/fed_*.png + raw_*.npy, prints actions
    python scripts/deploy_pusht_real.py --seed-dir "$d" --device cpu \
        --dry-run --dry-run-steps 20 \
        --dump-dir "results/dry_${tag}_${STAMP}"

    # 2. Capped logged rollout (arm moves — keep a hand on the E-stop)
    python scripts/deploy_pusht_real.py --seed-dir "$d" --device cpu \
        --steps 120 \
        --log-dir "results/roll_${tag}_${STAMP}"
done

