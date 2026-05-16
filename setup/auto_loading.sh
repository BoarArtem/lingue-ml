cd

sudo apt update && sudo apt upgrade

sudo apt install python3

git clone https://github.com/BoarArtem/linguo-ml.git

cd linguo-ml

python3 -m venv env --system-site-packages

source env/bin/activate

pip3 install -r requirements.txt

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

export PYTHONPATH=$PYTHONPATH:~/linguo-ml