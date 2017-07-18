from src.planning.behavioral.behavioral_state import BehavioralState
'''
Static methods for computing complex features, e.g., ACDA speed.
'''

def compute_acda_speed(behavioral_state: BehavioralState):
    pass

def get_preferred_lane(behavioral_state: BehavioralState):
    '''
    Navigation/traffic laws based function to determine the optimal lane for our current route. For example, default
     should be rightmost lane, but when nearing a left turn, should return the left lane.
    :param behavioral_state:
    :return: Integer representing the lane index 0 is right most lane.
    '''
    pass
