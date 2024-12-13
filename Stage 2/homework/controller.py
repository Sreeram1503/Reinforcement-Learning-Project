import pystk
from utils import PyTux

def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=25):
    import numpy as np
    action = pystk.Action()

    action.steer = steer_gain * aim_point[0]
    action.acceleration = 1.0 if current_vel < target_vel else 0.7
    action.drift = abs(aim_point[0]) > skid_thresh
    action.nitro = current_vel < target_vel * 0.7

    return action

def test_controller(pytux, track, verbose=False):
    import numpy as np

    track = [track] if isinstance(track, str) else track

    for t in track:
        steps, how_far = pytux.rollout(t, control, max_frames=50, verbose=verbose)
        print(steps, how_far)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')

    pytux = PyTux()
    test_controller(pytux, **vars(parser.parse_args()))
    pytux.close()
