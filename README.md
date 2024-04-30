# Dev repo for Thesis David Haasnoot

Code for Thesis on Windows Subsystem for Linux which runs the required eWaterCycle Mamba system. 
Runs models in eWaterCycle environment, applies DA to said models

```bash
wsl --shutdown
netsh winsock reset
netsh int ip reset all
netsh winhttp reset proxy
ipconfig /flushdns
netsh winsock reset
shutdown /r
```

This Repo contains data from the CAMELS USA data set which can be found [here](https://ral.ucar.edu/solutions/products/camels). 
Current hosting of this data is merely to keep it organised for myself and own use. Please download the data for your use case from the original owner. 
On completion of this thesis all data will be removed from github. 
