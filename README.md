# TX2-dnn-power-easurements
This project measures power of the TX2 board processing DNN models(take two stream cnn as an example). NVIDIA is providing a command line tool 'nvpmodel' which takes out a lot of the guess work in configuring the CPU and GPU settings to maximize performance and energy usage under different scenarios

## Nvpmodel
Nvpmodel introduces five different “modes” on the Jetson TX2. To summarize:

![alt text](https://developer.ridgerun.com/wiki/images/c/cd/NVPmodes.png)

To call nvpmodel:

$ sudo nvpmodel -m [mode]
    
where mode is the number of the mode that you want to use. For example:

$ sudo nvpmodel -m 1

You can query which mode is currently being used:

$ sudo nvpmodel -q --verbose

The file /etc/nvpmodel.conf holds the different models. Developers can add their own models to add different modes suitable to their application.

We can develop some code to read out these sensors to perform power measurements on the TX2 board. It is very convenient in general, and even allows to automatize the process for different benchmark programs, etc.

## QuickStart

Remember that the Jetson TX2 consists of a GPU along with a CPU cluster.Go to the *tx2-power-measurements* directory,the *tx2_predict.py* script with two stream cnn model executes 50 inferences. Then it reports the power measurements.Use the commands below to excute CPU and GPU versions:

*GPU version

python tx2_predict.py

*CPU version

CUDA_VISIBLE_DEVICE="" python tx2_predict.py

Make sure all dependencies is installed before running.

## Reference
[1] "Convenient Power Measurements on the Jetson TX2/Tegra X2 Board" url:https://embeddeddl.wordpress.com/2018/04/25/convenient-power-measurements-on-the-jetson-tx2-tegra-x2-board/

[2] "NVPModel – NVIDIA Jetson TX2 Development Kit" url:https://www.jetsonhacks.com/2017/03/25/nvpmodel-nvidia-jetson-tx2-development-kit/

