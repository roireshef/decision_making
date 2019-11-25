class PredictionInterface(object):

    def load_data(self, *argv):
        """
        :param argv: input raw data (path or some python object/s)
        :return:a pandas ddta-frame with following columns:
            time_stamp, target_id, rel_x, rel_y, rel_vx, rel_vy, rel_lane, dist_from_lane_sep, blinker, rel_ax, rel_ay,
            heading
        """
        pass

    def process(self, *argv):
        """
        :param argv: input raw data (path or some python object/s)
        :return: pandas data-frame, indexed by target_id, with the following columns:
        :return: pandas data-frame, indexed by target_id, with the following columns:
             - expected_lane (will target be in left / same / right lane relative to its current position),
             - expected_time (expected time_stamp when target's center of mass enters the aforementioned change_lane)
             - expected_rel_y (longitudinal distance in front of host at the aforementioned change_time)
             - trajectory
        """
