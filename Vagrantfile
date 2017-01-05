VAGRANTFILE_API_VERSION = "2"

$ouroboros_installation = <<SCRIPT
apt-get update
apt-get install -y bzip2
apt-get install -y python-pip python-dev python3-pip
apt-get install -y curl
apt-get install -y vim
apt-get install -y xvfb
apt-get install -y git
apt-get install -y libhdf5-serial-dev

python3 -m pip install --upgrade pip
pip3 install virtualenv
pip3 install -U -r /var/share/selflearner/selflearner/requirements.txt

git clone --recursive https://github.com/dmlc/xgboost.git
cd xgboost
./build.sh
cd python-package; python3 setup.py install

export PYTHON_PATH=/var/share/selflearner

pip3 install jupyter

SCRIPT

$ouroboros_run_services = <<SCRIPT

cd /var/share/selflearner/notebooks
jupyter notebook --ip=0.0.0.0 &

SCRIPT

Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|
  # Every Vagrant virtual environment requires a box to build off of.
  config.vm.box = "ubuntu/trusty64"
  config.vm.synced_folder "./", "/var/share/selflearner", create: true

  config.vm.network "forwarded_port", guest: 8888, host: 8888
 
  config.vm.provider "virtualbox" do |v|
    v.memory = 4096
  end
  
  config.vm.provision "shell", inline: $ouroboros_installation
  config.vm.provision "shell", run: "always", inline: $ouroboros_run_services, privileged: false
end
