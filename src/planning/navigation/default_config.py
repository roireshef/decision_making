from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
import numpy as np

NAVIGATION_PLAN = NavigationPlanMsg(np.array([3537, 76406, 3646, 46577, 46613, 87759, 8766, 76838, 228030,
                                              51360, 228028, 87622, 228007, 87660, 87744, 9893,
                                              9894, 87740, 77398, 87741, 25969, 10068, 87211, 10320,
                                              10322, 228029, 87739, 40953, 10073, 10066, 87732, 43516,
                                              87770, 228034, 87996, 228037, 10536, 88088, 228039, 88192,
                                              10519, 10432, 3537]))

NAVIGATION_PLAN_PG = NavigationPlanMsg(np.array(range(20, 30)))  # 20 for Ayalon PG