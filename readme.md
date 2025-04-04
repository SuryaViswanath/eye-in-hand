This project is about eye-in-hand. In this our aim is to understand the persons action based on his gaze, hand trajectory, and hand shape. Based on these parameters we aim to predict what is the object that the person could potentially pick up

What shall we do, how can we approach this:
1. Using vision transformers, identifying their attention and seeing how the action is carried based on the attention heads

or

2. Train gaze tracker, train hand trajectory identifier model, train shape vs object model. Run these models individually to predict the final action that they could do


Let's go for the second option:
    Stages:
        system starts -> check eye gaze direction -> does object exist -> set value
                         check hand movement exists -> plot hand movement trajectory -> set value
                         check handshape -> which object uses the handshape -> set value
                    -> use the 3 values to predict the next action -> this is done till the action is done
                    -> check if action is performed -> if not, repeat the process


keep in mind, this should be like a reinforcement learning process. Also for future todo, may be we can do some object detection to understand what are the products available

// gaze estimation was identifies, keep the object highlighted a little may be with outline of may be amber color depicting unsure
// the more the parameters emphasise towards the object on left or right, the closer the color gets to green and locking