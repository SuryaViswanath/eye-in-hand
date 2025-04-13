This project is about eye-in-hand. In this our aim is to understand the persons action based on his gaze, hand trajectory, and hand shape. Based on these parameters we aim to predict what is the object that the person could potentially pick up

Using pre-trained models to identify what's happening in the frame with regards to the person's actions

_approach inspired from the NVIDIA Cosmos-reason1 paper_: https://d1qx31qr3h6wln.cloudfront.net/publications/Cosmos_Reason1_Paper.pdf

Stages:

        system starts -> 
        
            check eye gaze direction -> does object exist -> set value
            
            check hand movement exists -> plot hand movement trajectory -> set value
            
            check handshape -> which object uses the handshape -> set value
            
            -> use the 3 values to predict the next action -> this is done till the action is done
            
            -> check if action is performed -> if not, repeat the process

How to run this:

Step 1 Clone the repository;

`git clone https://github.com/SuryaViswanath/eye-in-hand.git`

Step 2 Install the dependencies:

`pip install -r requirements.txt`

Step 3 Download and setup Ollama:

`https://ollama.com/download`

after installing ollama, run the following command:

`ollama run deepseek-r1:1.5b`

Start the LLM inference for reasoning capabilities:

`ollama serve`

Step 4 Run the system:
`python main.py`


System Design:
![image](https://github.com/user-attachments/assets/25d96f08-5a13-4726-a666-35657248d204)

Example Outputs:
![Screenshot 2025-04-13 at 5 49 43 PM](https://github.com/user-attachments/assets/a631bd6e-d230-4b38-a368-df7a2be35819)
![Screenshot 2025-04-13 at 5 50 14 PM](https://github.com/user-attachments/assets/37baa5aa-235e-401d-b15c-4f10cf8de748)

To Do:
- Add object detection
- Add live action anticipation
