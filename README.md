# TX2-dnn-power-easurements
This project measures power of the TX2 board processing DNN models(take two stream cnn as an example). NVIDIA is providing a command line tool 'nvpmodel' which takes out a lot of the guess work in configuring the CPU and GPU settings to maximize performance and energy usage under different scenarios

## Usage
Nvpmodel introduces five different “modes” on the Jetson TX2. To summarize:
![](https://developer.ridgerun.com/wiki/index.php?title=File:NVPmodes.png)

To call nvpmodel:

$ sudo nvpmodel -m [mode]
    
where mode is the number of the mode that you want to use. For example:

$ sudo nvpmodel -m 1

You can query which mode is currently being used:

$ sudo nvpmodel -q --verbose

The file /etc/nvpmodel.conf holds the different models. Developers can add their own models to add different modes suitable to their application.

We can develop some code to read out these sensors to perform power measurements on the TX2 board. It is very convenient in general, and even allows to automatize the process for different benchmark programs, etc.
