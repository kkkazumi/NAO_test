# PYTHONPATH

- before starting,

  - run choregraphe
  - check port number by pushing button of network
  - change the port number of test.py

  - cd ~/prog/emopy_test/
  - run `python3 fermodel_example_webcam.py`

- at first, `source ./.bashrc`
- then, `python2.7 test.py`

export PYTHONPATH=${PYTHONPATH}:/usr/local/src/pynaoqi-python2.7-2.8.6.23-linux64
export LD_LIBRARY_PATH="/home/user_name/Software/pynaoqi-python2.7-2.8.6.23-linux64:$LD_LIBRARY_PATH"

なんとなくできてきた感じがする。
もう少し、表情の時系列を取るようにしたらどうかな。
それができたら次は、ロボットの動作（ループ）を何回やってから学習するか。
ループN回やって学習するならば、N*input次元が入力ベクトルの次元になる。
