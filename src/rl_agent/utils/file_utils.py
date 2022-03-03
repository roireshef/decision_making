from xml.etree.ElementTree import Element

from lxml import etree
import os
from datetime import datetime

from decision_making.paths import Paths


class FileUtils:
    @staticmethod
    def backup_code_into_experiment_dir(experiment_dir: str) -> None:
        """
        Backs-up the current snapshot of the source code for debug purposes into the experiment dir
        :param experiment_dir: the results directory of the experiment
        :return: nothing
        """
        source_dir = Paths.get_repo_absolute_dir_path("src")
        os.system("mkdir -p %s" % experiment_dir)
        os.system("tar czf %s/code.tar.gz %s" % (experiment_dir, source_dir))
        print("source code backed up into %s/code.tar.gz" % experiment_dir)

    @staticmethod
    def get_ray_results_dir() -> str:
        return os.path.expanduser("~") + "/ray_results"

    @staticmethod
    def get_datetime_dir() -> str:
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @staticmethod
    def get_xml_root_from_file(filename: str) -> Element:
        return etree.parse(filename).getroot()
