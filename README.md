**This is a practice repo for MLOps.**

The goal is to implement a containerized model for processing the dataset:
corrupted mnist

**Usage**

This project utilizes docker to train and test the model. Here are the steps to run the project as intended:

1. pull this repository
	1. `git clone <my repository>`
2. fetch data from the dvc located in the data folder
	1. `dvc pull`
3. build the trainer.dockerfile image
	1. `docker build -f trainer.dockerfile . -t trainer:latest`
4. run the trainer.dockerfile image
	1. `docker run --name exp1 trainer:latest`
5. copy the generated model to a docker_folder
	1. `docker cp exp1:app/model /<path to local gh repository>`
6. build the test.dockerfile image
	1. `docker build -f tester.dockerfile . -t tester:latest`
7. run the test.dockerfile image to get the stats
	1. `docker run --name test1 tester:latest`
8. et viola.
