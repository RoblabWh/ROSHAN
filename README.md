# ROSHAN
![](ROSHAN_uebersicht.png)
ROSHAN (Rescue Oriented Simulation: Handling and Navigating Fires), is a wildfire simulation tool. ROSHAN integrates the principles of cellular automata with reinforcement learning to simulate wildfire dynamics and automatic handling. entral to this simulation tool is its rendering and simulation of fire spread, achieved by integrating data from the CORINE database. The interactive graphical interface of ROSHAN facilitates real-time monitoring and manipulation of fire scenarios. A key component in ROSHAN is the incorporation of a Reinforcement Learning agent, embodied as a drone, which learns to detect and mitigate fires.

You can read everything about the development [here](paper.pdf).

<div align="center">
  <video src="agent.mp4" width="400" />
</div>

# Installation
## Dependencies

These modules go under externals:

###### DearImgui 

https://github.com/ocornut/imgui

https://github.com/aiekick/ImGuiFileDialog

###### JSON
https://github.com/nlohmann/json

###### Pybind11
`sudo apt install libpython3.9-dev`

https://github.com/pybind/pybind11

#### NodeJS

`cd openstreetmap`

`npm install express body-parser`

`npm install --save-dev nodemon`

#### CORINE CLC+ 

Download Corine CLC+ Backbone - 10 meter (Year 2018)

https://land.copernicus.eu/pan-european/clc-plus/clc-backbone/clc-backbone?tab=download

Move *CLMS_CLCplus_RASTER_2018_010m_eu_03035_V1_1.tif to `/CORINE/dataset/`

Install GDAL and GDAL C++ headers

`sudo apt install libgdal-dev gdal-bin libsdl2-image-dev`

#### SDL2 - min. 2.0.17 

Install SDL2 according to:

https://github.com/libsdl-org/SDL/releases/tag/release-2.26.5
https://github.com/libsdl-org/SDL_image/releases/tag/release-2.6.3

##### Anaconda & PyTorch

`conda create --name roshan python=3.9 libffi==3.3`

`conda activate roshan`

`pip install torch torchvision`

`conda install tensorboard`

`conda install packaging`

##### LLM Support

`pip install transformers[torch] onnxruntime bitsandbytes optimum onnx`

#### Compile

`cd \ROSHAN`

`mkdir build && cd build`

`cmake .. && make -j&(nproc)`

# Usage

## Just the ROSHAN Simulation

`cd build`

`./ROSHAN`

## ROSHAN + Reinforcement Learning

`cd drone_agent/FireSimAgent`

`python main.py`