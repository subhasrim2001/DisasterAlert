# DisasterAlert
Google Girl Hackathon 2023 ideathon submission

## Setting up the project

### 1. Create a virtual environment
```
python -m venv venv
venv\Scripts\activate
```
### 2. Install requirements
```
pip install -r requirements.txt
```
Note: some of the requirements may not be required depending on your IDE (say Spyder) and you can manually install whatever is required and skip the rest

## Running the project

The project consists of many components, each of which can be tested separately as well as together in the flask UI.

### 1. Train and save the model 

Run `train.py` 
The model will be saved in `model/` folder

Note: the model has already been trained and the .model and .pickle file have been pushed and can be used directly

### 2. Predict video: 
`predict_video.py`

This is a function which predicts the disaster occurring in the video in every frame and displays the output. It can also save the output file, if we enable the writer. If you do not need it, because of security reasons, you may comment it off.

Run the code and call the function in your terminal

### 3. Predict real time video: 
`predict_video_realtime.py`

Uses opencv to capture real time video using camera and processes it to predict the disaster

### 4. Use telegram to send alerts and notifications

Telegram was used because it has a free Telebot which can be used to share messages and chat.
This component can be accessed in the flask application
It uses the `telebot\` folder
`credentials.py` : has the required credentials to access the bot
`mastermind.py` : has the alert code. We can add special kinds of alerts such as auditory alerts or notification alerts in the future as extension. We can also add our AI component here.

### 5. Run the flask application

Run `app.py` and open `http://127.0.0.1:8000/` : runs in port 8000
It consists of 2 routes
1) Upload a video to application and render disaster o/p
2) Use a telegram token to send alerts


## Some outputs and folders

1. Precision-Recall.png : File which shows Precision-Recall

2. Confusion.png : File which shows normalized confusion matrix

3. uploads: Folder which stores the uploaded video in the flask UI

4. templates: Folder which stores simple website template

5. output: Folder to save the output video here (if necessary)

6. example_clips: Folder which has sample i/p links which can be used for testing

7. data: Folder which has folders of 41 sample images of Cyclone, Earthquake, Flood and Wildfire - can increase dataset size from given drive link in pdf to increase accuracy
