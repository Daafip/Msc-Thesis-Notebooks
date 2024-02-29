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
