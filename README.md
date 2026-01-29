# how to run it:
## Structure Guidance
```bash
workspace/
└── src/
    └── llm/                    # Metapackage/Main package
        ├── CMakeLists.txt      # Main package's CMakeLists
        ├── package.xml         # Main package's package.xml
        ├── src/                # Main package's source code
        │   └── ...             # Main package's functions/code
        ├── speech_interface/   # Sub-package 1
        │   ├── setup.py
        │   └── package.xml
        ├── servo_service/      # Sub-package 2
        │   ├── CMakeLists.txt
        │   └── package.xml
        └── llm_msgs/           # Sub-package 3
            ├── CMakeLists.txt
            └── package.xml
```

## Env settings
```bash
    # order matters!
	python3 -m venv groupE_venv --symlinks
    touch groupE_venv/COLCON_IGNORE
	source groupE_venv/bin/activate

    python -m pip install "scipy==1.16.3"
	pip install mediapipe piper-tts faster-whisper sounddevice soundfile webrtcvad "numpy<2"
    pip install catkin_pkg empy lark pyyaml opencv-contrib-python==4.9.0.80
    python -m pip install "numpy==1.26.4"

    # fallback, not neccessary
    python -m pip install -r requirements.txt

    source /opt/ros/jazzy/setup.bash
	source groupE_venv/bin/activate
	
	colcon build --symlink-install
	source install/setup.bash

    export PYTHONPATH="$PWD/groupE_venv/lib/python3.12/site-packages:${PYTHONPATH}"
```


## Launch the nodes

	ros2 launch bringup bringup.launch.py




# some notes:

Your can check all TODOs in the following packages

1. (llm_interface): If you want to use llm api calling, you should get into the .bashrc with the command: nano ~/.bashrc and write the following command in it: export OPENAI_API_KEY= "YOUR_CHATGPT_APIKEYS"
  



2. (speech_interface)
tts_node:adapt the model path to your own one;
asr_node:adapt the microphone id to your own one, if needed


3. (ainex_vision):
face_detection_node:adapt the topic name to match your camera setup



PS: If anyone want to use it, feel free to text me, i will send you the API KEY.
