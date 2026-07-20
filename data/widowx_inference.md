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

