# signs-challenge
Apply deep learning to detects, reads and positions parking signs on a map.

# Versions
Python version is 3.5. All the versions of the third-party libraries are given in `requirements.txt`.
To install all the third-party libraries do the following:
```shell
pip install -r requirements.txt
```

# Use
## Necessary folders
You need to create several folders at the root of the project: `data`, `logs`, and `models`.
```shell
mkdir data logs models
```

## Data preparation
You need to copy the data folders in the `data` folder.
Then, to generate the tfrecords files, you need to run the notebook called `tfrecords_generation.ipynb`.
Finally, to separate the training data into train and test dataset, you need to run the notebook called `data_separation.ipynb`.

## Model training
To run the model you can either run the `model_training.ipynb` notebook (be careful to set the parameters to the desired value).
You can also run the python script `model_training.py`.
```shell
python model_training.py
```
The checkpoints will be saved in the `logs` folder and the weights in the `models` folder.
You can check the evolution of the training with tensorboard using:
```shell
tensorboard --logdir=logs
```

## Result
The results of the exercise can be computed in the notebook called `inference.ipynb`.
