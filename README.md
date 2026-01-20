# Notation from Yuan Zhao. 20. Jan

The README.md for vision part please go to src/vision_part/vision



# how to run it:
## Structure
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

	python3 -m venv groupE_venv --system-site-packages --symlinks
	source groupE_venv/bin/activate

	pip install mediapipe
	pip install piper-tts
	pip install faster-whisper sounddevice 
	pip install face_recognition
	soundfile numpy
	pip install webrtcvad
    pip uninstall numpy 
    pip install "numpy<2"

	source groupE_venv/bin/activate
	source /opt/ros/jazzy/setup.bash
	colcon build --symlink-install
	source install/setup.bash


	export PYTHONPATH=/groupE_venv/lib/python3.12/site-packages:$PYTHONPATH


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