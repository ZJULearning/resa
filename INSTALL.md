
# Install

1. Clone the RESA repository
    ```
    git clone https://github.com/zjulearning/resa.git
    ```
    We call this directory as `$RESA_ROOT`

2. Create a conda virtual environment and activate it (conda is optional)

    ```Shell
    conda create -n resa python=3.8 -y
    conda activate resa
    ```

3. Install dependencies

    ```Shell
    # Install pytorch firstly, the cudatoolkit version should be same in your system. (you can also use pip to install pytorch and torchvision)
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

    # Or you can install via pip
    pip install torch torchvision

    # Install python packages
    pip install -r requirements.txt
    ```

4. Data preparation

    Download [CULane](https://xingangpan.github.io/projects/CULane.html) and [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). Then extract them to `$CULANEROOT` and `$TUSIMPLEROOT`. Create link to `data` directory.
    
    ```Shell
    cd $RESA_ROOT
    ln -s $CULANEROOT data/CULane
    ln -s $TUSIMPLEROOT data/tusimple
    ```

    For Tusimple, the segmentation annotation is not provided, hence we need to generate segmentation from the json annotation. 

    ```Shell
    python scripts/convert_tusimple.py --root $TUSIMPLEROOT
    # this will generate segmentations and two list files: train_gt.txt and test.txt
    ```

    For CULane, you should have structure like this:
    ```
    $RESA_ROOT/data/CULane/driver_xx_xxframe    # data folders x6
    $RESA_ROOT/data/CULane/laneseg_label_w16    # lane segmentation labels
    $RESA_ROOT/data/CULane/list                 # data lists
    ```

    For Tusimple, you should have structure like this:
    ```
    $RESA_ROOT/data/tusimple/clips # data folders
    $RESA_ROOT/data/tusimple/lable_data_xxxx.json # label json file x4
    $RESA_ROOT/data/tusimple/test_tasks_0627.json # test tasks json file
    $RESA_ROOT/data/tusimple/test_label.json # test label json file
    ```

5. Install CULane evaluation tools. 

    This tools requires OpenCV C++. Please follow [here](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html) to install OpenCV C++.  Or just install opencv with command `sudo apt-get install libopencv-dev`

    
    Then compile the evaluation tool of CULane.
    ```Shell
    cd $RESA_ROOT/runner/evaluator/culane/lane_evaluation
    make
    cd -
    ```
    
    Note that, the default `opencv` version is 3. If you use opencv2, please modify the `OPENCV_VERSION := 3` to `OPENCV_VERSION := 2` in the `Makefile`.