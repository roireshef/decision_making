class NavigationPlan:
    """
    This class hold the navigation plan.
    """
    def __init__(self, road_ids: list[int]):
        """
        Initialization of the navigation plan. This is an initial implementation which contains only a list o road ids.
        :param road_ids: list of road ids corresponding to the map.
        """
        self._road_ids = road_ids

