# DC-DRL

**Distributed and Collective Deep Reinforcement Learning for Computation Offloading: A Practical Perspective**


Python code for our work, it includes:
* simulation code for mutli-user computation offloading
* testbed code to run experiment on Raspberry pis

# Require Packages

* numpy
* pytorch
* gym
* face-recognition 
* redis

# How the code works

this code has two part: simuation code and testbed code

## Simulation code

To run simulation, you should:

**step 1**: deploy the master agent and client agents (PS: you can also run the distributed code on single machine by changing the host to `127.0.0.1`)

**step 2**: start redis database, we use it to control the experiences and models

**step 3**: run the following python file: 

* parameter_server.py: deploy on the master side, which is responsible to:
    * handle model request from clients
    * reveive model and experience sent from clients
    * generate the next model
* master.py: deploy on the master side, which is responsible to training with the experience it collected and received from client, generally, it can provide a more compresensive than client agent as it has richer experiences
* client.py: deploy on the client side, which is responsible to interacting with env. and collect experience, multiple clients can provide higher diversity

## testbed code

To validate the practicality and applicability of the DC-DRL algorithm, we build a real multi-user computation offloading testbed with Raspberry Pi 3B and PC.

The Raspberry Pi 3B should install face_recognition and opencv, you can find an useful instruction in:
* [Install dlib and face_recognition on a Raspberry Pi](https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65)
* [Install OpenCV 4 on Raspberry Pi](https://www.learnopencv.com/install-opencv-4-on-raspberry-pi/)
 
Available image for our experiment to save your effort in configuring environment:

We also provide some files to test if your envs. can work:
* facerec_from_video_file.py
* facerec_from_camera.py


After you have setup the Raspbery Pi, you can run the experiment by:
* deploy DC-DRL on Raspberry Pi, run client.py
* deploy DC-DRL on PC, run server.py


 


    
    
    
    
      
