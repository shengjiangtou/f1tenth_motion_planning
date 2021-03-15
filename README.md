# The F1TENTH Motion Planner

This is the repository is based on the F1TENTH Gym environment and is providing different
Motion Planning algorithms and path tracking algorithms in the field of autonomous driving.
In addition this repository is using different maps from famous racetracks all over the world.

You can find the [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.

This repository implemented some common motion planners used on autonomous vehicles, including
* 

Also, this repository provides some controllers for path tracking, including
* [Pure Pursuit + PID](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf)
* [Front-Wheel Feedback / Stanley + PID](http://robots.stanford.edu/papers/thrun.stanley05.pdf)


## Install
You can install the environment by running:

```bash
$ git clone https://github.com/f1tenth/f1tenth_gym.git
$ cd f1tenth_gym
$ git checkout exp_py
$ pip3 install --user -e gym/
```

## Run a Motion Planner

Then you can run different motion planners example by:
```bash
cd examples
python3 waypoint_follow.py
```

## Known issues
- On MacOS Big Sur and above, when rendering is turned on, you might encounter the error:
```
ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework.
```
You can fix the error by installing a newer version of pyglet:
```bash
$ pip3 install pyglet==1.5.11
```
And you might see an error similar to
```
gym 0.17.3 requires pyglet<=1.5.0,>=1.4.0, but you'll have pyglet 1.5.11 which is incompatible.
```
which could be ignored. The environment should still work without error.
