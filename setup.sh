#First install tf and keras
conda create --name rcps python=3.7
conda install tensorflow-gpu=1.14
conda install keras=2.2.4

# Then add sklearn and avanti's own modules
pip install scikit-learn
pip install keras-genomics
pip install seqdataloader
pip install momma_dragonn

