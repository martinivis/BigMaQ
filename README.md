# BigMaQ


## Installation
- Install pytorch3d from source
- then install torch etc.


## ToDOS
- implement the smoothing for procurstes, joint angle limits, alignment of votes 




- Joint angle limits:
  - for over time optimization anyway quite low factor, but maybe for cases where there are flips etc. that may make sense
  - --> implement optimization over the entire dataset that we put to the side instead of this exemplary action.


- test some of these implementations on an exemplary video:
  - /media/lucas/V-2/Session8/Actions/Interaction/Displacement_G_T_10
  - Tonic 73, 74 frame , would be good to check for angle limitations
- Find a single animal video where the procrustes does make a flicker in the video
- Pose limit already works quite good. for simple example


- After it all runs, clean the code for unnecessary things, dead code


- for a given action:
  - 1/7 GB for masks of entire video size
  - a little more for yolo crops
  - ---> try to get the images out of the videos