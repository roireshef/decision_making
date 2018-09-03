from sys import argv
import json

"""
  Use this to convert the JSON log holding the MetricLogger metrics into a valid json file (and merged)
  Currently supports only a pair of components.
  
        Usage: 
            first arg: <Json log location> 
            second optional flag [-m] (for merging))
  
"""

component_merges = {'TP': 'TP_bp_time', 'BP': 'BP_ts'}


class ConvertJsonsScript():

    @staticmethod
    def simple_strip():
        """
        :returns List of JSON records
        """
        ret = []
        with open(log_filename, "r") as log_file:
            for l in log_file:
                s = l.split('data:')[1].replace('\'', '"').rstrip()
                if len(s):
                    ret.append(s[:-1])
        return ret


    @staticmethod
    def extract_component(j):
        c_set = set(k.split('_')[0] for k in j if '_' in k)
        if len(c_set) == 0:
            return None
        return c_set.pop()


    @staticmethod
    def do_dump_jsons():
        jsons = {c: [] for c in component_merges}
        for j in [json.loads(l) for l in ConvertJsonsScript.simple_strip()]:
            ec = ConvertJsonsScript.extract_component(j)
            if ec is not None:
                jsons[ec].append(j)
        c, cf = list(component_merges.items())[0]
        c1, cf1 = list(component_merges.items())[1]
        for j_c in jsons[c]:
            merge_record = [j_c1 for j_c1 in jsons[c1] if j_c1[cf1] == j_c[cf]][0]
            for k, v in merge_record.items():
                j_c[k] = v
            print(json.dumps(j_c))


if __name__ == '__main__':
    try:
        log_filename = argv[1]
        if len(argv) > 2 and argv[2] == '-m':
            ConvertJsonsScript.do_dump_jsons()
        else:
            for j in ConvertJsonsScript.simple_strip():
                print(j)
    except:
        "Usage: {0} <Json log location> [-m]" % argv[0]
