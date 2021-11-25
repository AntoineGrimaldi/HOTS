# neuronhots

A bio-plausible version of [HOTS: A Hierarchy of Event-Based Time-Surfaces for Pattern Recognition](https://ieeexplore.ieee.org/abstract/document/7508476)

## Requirements :
Running the HOTS library requires the following packages :

* numpy
* pandas
* matplotlib

This can be performed using the `requirements.txt` file :
```
pip install -r requirements.txt
```


## Installation


To Install the HOTS library, move to the root folder (where you find the `setup.py` file) and type :
```
pip install -e .
```

## TODO:
- find an entropy measure for the (not histograms) kernels -> then method to get hom_param (+ measure of the balance between strengh of the gain and homeostasis)
- find a way to tune the hyper-parameters as a function of the signal
- make a good recordings of the network characterisitics during learning
- think about how to save/output data (tonic format -> easy to use and exchange)
- have some easy to implement methods for:
    - training a network
        - unsupervised (done)
        - supervised
    - running a network (ouptut is classification result or stream of events, make a flag for the last layer)
    - learning rule for the classification layer (spiking mechanism==same as in HOTS)
    - get results of the paper:
        - online classification results
        - histogram classification 
        - adding jitter
- try the averaged time surface to separate movement and object
- try to add a threshold
- try with alpha instead of exponential decay
- what happens with sigma
- try to make a stride within the network
