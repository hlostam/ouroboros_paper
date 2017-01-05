Ouroboros experiments running
=============================

The experiments for the Ouroboros are written in Python3. For easy usage and reproducing the 
experiments  Vagrant file was created. Using vagrant, new virtual machine (VM)  will be installed 
with all the necessary source codes and libraries. This VM serves as the platform for the experiments.

Installation - Preparing of the experiment platform
----------------------------------------------------
Dependencies
-  VirtualBox
-  Vagrant

1.  Download and install virtualbox -- the VM image was created using virtualbox.
    
    https://www.virtualbox.org/wiki/Downloads

2.  Download and install vagrant:
    
    https://www.vagrantup.com/downloads.html

3.  Locate the Vagrant file and run
    
    vagrant up

    This operation might take some time as the virtual machine image is downloaded together with
    python libraries necessary for running the experiments.

Running the experiments
-----------------------

1.  Now open your browser on your operating system and enter url:
    http://localhost:8888/

2.  Jupyter notebook will show up with the list of notebooks that contain the experiments.
    -  ouroboros_experiments.ipynb -- experiments from the evaluation part of the paper
    -  ouroboros_stats.ipynb -- various statistic/summary information that were used for the introduction and motivation of the paper

3.  Open ouroboros_experiment.ipynb and click on Cell->Run All in the top menu 

4.  Now all the experiments are run and the graphs are generated.
