apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
git clone https://github.com/el3ment/gym
cd gym
pip install -e '.[all]'
pip install tqdm, pyqtgraph