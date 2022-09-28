# DogBreedIdentification
Dog breed "fusion" code to generate a new breed 

## Instructions:

### Steps to install requirements of the code:

First, create an virtualenv and activate it with the commands:
```
python3 -m venv env
source env/bin/activate
```
Then run the below code to install the src as root folder and install dependencies.
```
pip3 install -e .
```

After the installation of dependencies, please, download the processed data with the below link, and paste it in the data folder:
```
https://drive.google.com/file/d/1V9rTCyw9LWE5rYW0QRpLEcRwI__Slv3-/view?usp=sharing
```

## To run the code:
First, go to src/scripts/ then, run the code:

```
python3 run_vae.py
```

The dataset already converted to tensors will be loaded and used in the code.

Then, the model will be loaded and the two selected images will be shown for 5 seconds.

After that, the model will be used to generate some images from the two images.