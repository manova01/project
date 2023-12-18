# Dataset
Context
According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.
This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.

## Attribute Information
1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12) stroke: 1 if the patient had a stroke or 0 if not
*Note: "Unknown" in smoking_status means that the information is unavailable for this patient

### Objective
Early prediction and medication.
## How It Works 

Everything here runs locally. If you want to try out the service, follow the steps below:

Before you proceed, create a virtual environment.

I used python version 3.10. To create an environment with that version of python using Conda:

conda create -n <env-name> python=3.10

Just replace <env-name> with any title you want. Next:

 conda activate <env-name>

to activate the environment.


## Dockerfile
Now run:

 pip install -r requirements.txt

to install all necessary external dependencies.

Next, Run:

docker build -t <service-name>:v1 .

Replace <service-name> with whatever name you wish to give to the body fat percent estimator service, to build the image.

To run this service:

docker run -it --rm -p 9696:9696 <service-name>:latest

NOTE: I am running this on Windows hence Waitress. If your local machine requires Gunicorn, I think the Dockerfile should be edited with something like this:


RUN pip install -U pip

WORKDIR /app

COPY [ "online_webservice_flask/predict.py", "models/pipeline.bin", "requirements.txt", "./" ]

RUN pip install -r requirements.txt

EXPOSE 9696 
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]

If the container is up and running, open up a new terminal. Reactivate the Conda environment. Run:

python predict.py

NOTE: predict.py is an example of data you can send to the ENTRYPOINT to interact with the service. Edit it as much as you desire and try out some predictions.

