# NETSIM Installation and Usage Tutorial

The tutorial below explains how to install and compile NETSIM, run demo simulations, analyze outputs using MATLAB, and leverage a Python interface for additional functionality.

## 1. Unzipping NETSIM 

To begin working with NETSIM, start by unzipping the package in your working directory—a process that typically takes less than 10 seconds.

## 2. Navigating to the NETSIM Folder

After unzipping, open your terminal and change the working directory to the NETSIM folder.

## 3. Compiling NETSIM

Since prior builds might conflict with new compilations, it is advisable to run the following command to remove any remnants of previous compilations.

*(Note: The following code cell contains a shell command. To run it in a Python-based Jupyter notebook, you might need to prefix it with `!` or use the `%%bash` cell magic.)*


```python
make clean
```

Once the directory is cleared, compile the source code with performance settings by running:

*(Note: The following code cell contains a shell command. To run it in a Python-based Jupyter notebook, you might need to prefix it with `!` or use the `%%bash` cell magic.)*


```python
make COMPILE_TYPE=performance
```

This entire procedure ordinarily finishes in under 30 seconds.

## 4. Running a Demo Simulation

After installation, you can run a demo simulation by executing NETSIM with a sample parameter file. For example, entering the command below instructs NETSIM to read the simulation parameters from the file “test1.parameters” and save the output in the current directory.

*(Note: The following code cell contains a shell command. To run it in a Python-based Jupyter notebook, you might need to prefix it with `!` or use the `%%bash` cell magic. Ensure paths are correct relative to your notebook's execution directory or use absolute paths.)*


```python
./netsim -f ./tests/test1.parameters -o .
```

Once the simulation runs—which usually lasts around 60 seconds—the terminal should display that a total of **343298 spikes** were generated, and several binary files (such as `00000002individual.bin`, `00000002gi.bin`, `00000002ge.bin`, among others) will be created.

## 5. Analyzing Output with MATLAB

For researchers who wish to analyze the simulation output, MATLAB provides an efficient approach. 

*(Note: The following code cell contains MATLAB code.*


```python
[ids, times] = load_netsim_spikes('../00000002spk.bin');

% Once the data is loaded, create a raster plot by issuing the command
plot(times, ids, '.k')
xlabel('id')
ylabel('time (s)')
```

## 6. Using the Python Interface (pyNetsim)

For those who prefer working with Python, NETSIM is accessible through the `pyNetsim` package, which offers an object-oriented interface.

First, install the package (if you haven't already):

*(Note: The following code cell contains a shell command for pip install. To run it in a Python-based Jupyter notebook, you might need to prefix it with `!`.)*


```python
pip install git+https://github.com/mullerlab/NETSIM-dev.git
```

Once installed, integrate NETSIM into your Python workflow by importing the package:


```python
import pyNetsim.netsim as netsim
```

You can then create a NETSIM object by specifying the parameter file (replace `<path-to-parameter-file>` with the actual path):


```python
net = netsim.netsim(params='<path-to-parameter-file>')
```

And run the simulation via:


```python
net.run()
```

After the simulation completes, retrieve the results using:


```python
res = net.get_results(reshape=True)
print(res)
```

If it is necessary to modify parameters and rerun the simulation, simply adjust the parameters by executing:


```python
net.set_params(ge=1e-8)
# # And run the simulation again
net.run()
```

## 7. Advanced Analysis

In addition to running simulations, NETSIM supports detailed analysis and optimization. On the analysis side, you can compute important metrics such as pairwise spike correlation, coefficients of variation, and firing rate. This is achieved by executing a command like the following within your simulation environment (assuming `simulation` is your `netsim` object, e.g., `net` from the previous steps):


```python
# Assuming 'net' is your netsim object from the Python interface section
# results = net.analyze(['pairwise_spike_correlation',
#                          'coefficients_of_variation',
#                          'firing_rate'],
#                         dt=1e-4, bin_size=500)
# print(results)
```

## 8. Parameter Optimization with neverSim

NETSIM also integrates with the `neverSim` package to facilitate optimization of simulation parameters. This functionality uses iterative approaches guided by a user-defined cost function. An illustrative example of the optimization workflow is provided below. You can paste the following code directly into your Python editor (or a Jupyter cell).

First, ensure `neverSim` is installed (e.g., `!pip install neverSim` if not already). You would also need your NETSIM setup accessible to this Python environment.


```python
import neverSim
import json

# Define parameters
params = {
    "param1": [1.0, 0.5, 2.0],
    "param2": [10, 5, 15]
}

# Convert parameters to JSON string as expected by the main function
# params_json = json.dumps(params) # Not directly used by optimize_NETSIM, but good to know

# Define a simple user cost function
# This function would typically interact with NETSIM or analyze its output
# For this example, it's kept simple.
def user_cost_example():
    # In a real scenario, this function would:
    # 1. Run a NETSIM simulation with parameters set by neverSim.
    # 2. Analyze the output.
    # 3. Return a cost value (float).
    # For now, it just returns a dummy value.
    print("User cost function called (example - returns dummy value)")
    return 100.0

# Call the optimize_NETSIM function
# Note: You'll need to ensure NETSIM is correctly configured for neverSim to call it.
# The 'params' argument to optimize_NETSIM should be the dictionary, not the JSON string here.
# The 'user_cost' function provided to optimize_NETSIM will be called by neverSim.

# This is an illustrative call. Actual usage may require NETSIM to be callable
# and the user_cost function to be implemented to interact with NETSIM runs.
try:
    print("Attempting to call neverSim.optimize_NETSIM (this is illustrative).")
    print("Ensure NETSIM and its dependencies are correctly set up for neverSim.")
    # neverSim.optimize_NETSIM(
    #     opt='OnePlusOne',
    #     params=params,  # The dictionary of parameters to tune
    #     out_dir='./output_neversim_example', # Ensure this directory can be written to
    #     user_cost=user_cost_example, # Your actual cost function
    #     itrs=5, # Reduced for quick example; original was 40
    #     workers=1 # Reduced for quick example; original was 4
    #     # You might need to pass netsim_path or other specific args depending on neverSim setup
    # )
    print("Call to neverSim.optimize_NETSIM commented out to prevent errors in a generic setup.")
    print("Refer to neverSim documentation for actual usage with NETSIM.")
except Exception as e:
    print(f"Error during neverSim example setup (as expected if not fully configured): {e}")
```

In this example, the optimization function `neverSim.optimize_NETSIM()` takes the parameters to be tuned, defines the output directory (“./output”), uses a user cost function that simply returns 100, and runs the optimization for 40 iterations across 4 worker processes. This process automates parameter tuning, thereby enhancing simulation performance and ensuring that NETSIM operates at peak efficiency.
