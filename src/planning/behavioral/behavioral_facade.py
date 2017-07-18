class BehavioralFacade:

    def __init__(self, policy, behavioral_state):
        self._policy = policy
        self._behavioral_state = behavioral_state
        self._navigation_plan = None

    def update_navigation_plan(self, navigation_plan):
        self._navigation_plan = navigation_plan

    def update_state_and_plan(self, state):
        self._behavioral_state.update(state)
        self._policy.plan(self._behavioral_state)





