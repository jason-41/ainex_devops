# how to run it:

## Env settings



	python3 -m venv groupE_venv --system-site-packages --symlinks
	source groupE_venv/bin/activate

	pip install mediapipe
	pip install piper-tts
	pip install faster-whisper sounddevice 
	soundfile numpy
	pip install webrtcvad

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