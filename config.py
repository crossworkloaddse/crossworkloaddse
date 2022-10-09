# spec06-int * 2, fp *2
# case_name = "403.2-ref-1"
# case_name = "437.1-ref-10"
# case_name = "471.1-ref-1"
# case_name = "453.1-ref-16"

# spec17-int * 9
# case_name = "500.1-refrate-1"
# case_name = "502.2-refrate-1"
# case_name = "505.1-refrate-1"
# case_name = "523.1-refrate-1"
# case_name = "525.1-refrate-1"
# case_name = "531.1-refrate-1"
# case_name = "541.1-refrate-1"
# case_name = "548.1-refrate-1"
# case_name = "557.1-refrate-1"
# spec17-fp * 13
# case_name = "503.1-refrate-1"
# case_name = "507.1-refrate-1"
# case_name = "508.1-refrate-1"
# case_name = "510.1-refrate-1"
# case_name = "511.1-refrate-1"
# case_name = "519.1-refrate-1"
# case_name = "521.1-refrate-1"
# case_name = "526.1-refrate-1"
# case_name = "527.1-refrate-1"
# case_name = "538.1-refrate-1"
# case_name = "544.1-refrate-1"
# case_name = "549.1-refrate-1"
# case_name = "554.1-refrate-1"
import copy
import random

import numpy as np

np.random.seed(0)

case_names = [
    "500.1-refrate-1",
    "502.2-refrate-1",
    "503.1-refrate-1",
    "505.1-refrate-1",
    "507.1-refrate-1",  # not coverage, last_predict has high effect
    "508.1-refrate-1",
    "510.1-refrate-1",
    "511.1-refrate-1",
    "519.1-refrate-1",  # pareto is straight strange
    "520.1-refrate-1",
    "521.1-refrate-1",  # pareto is straight strange
    "523.1-refrate-1",  # pareto non-uniform interval is easy to stuck
    "525.1-refrate-1",
    "526.1-refrate-1",
    "527.1-refrate-1",
    "531.1-refrate-1",  # pareto is straight strange
    "538.1-refrate-1",
    "541.1-refrate-1",
    "544.1-refrate-1",
    "548.1-refrate-1",
    "549.1-refrate-1",
    "554.1-refrate-1",
    "557.1-refrate-1",
]

case_names_str = [
    'perlbench',  # "500.1-refrate-1",
    'gcc',  # '# "502.2-refrate-1"
    'bwaves',  # "503.1-refrate-1",
    'mcf',  # "505.1-refrate-1",
    'cactuBSSN',  # "507.1-refrate-1",
    'namd',  # "508.1-refrate-1",
    'parest',  # '"510.1-refrate-1",
    'povray',  # "511.1-refrate-1",
    'lbm',  # "519.1-refrate-1",
    'omnetpp',  # "520.1-refrate-1",
    'wrf',  # "521.1-refrate-1",
    'xalancbmk',  # '"523.1-refrate-1",
    'x264',  # '"525.1-refrate-1",
    'blender',  # "526.1-refrate-1",
    'cam4',  # "527.1-refrate-1",
    'deepsjeng',  # '"531.1-refrate-1",
    'imagick',  # "538.1-refrate-1",
    'leela',  # '"541.1-refrate-1",
    'nab',  # "544.1-refrate-1",
    'exchange2',  # '"548.1-refrate-1",
    'fotonik3d',  # "549.1-refrate-1",
    'roms',  # "554.1-refrate-1",
    'xz',  # '"557.1-refrate-1",
]


def get_domain_id(case_name):
    for domain_id, case_name_iter in enumerate(case_names):
        if case_name_iter == case_name:
            return domain_id
    print(f"get_domain_id: no {case_name}")
    exit(1)
    return -1


import socket
import sys

if 1 < len(sys.argv):
    if '502' == sys.argv[1]:
        case_name_config = sys.argv[1] + ".2-refrate-1"
    else:
        case_name_config = sys.argv[1] + ".1-refrate-1"
else:
    #case_name_config = "502.2-refrate-1"
    case_name_config = "523.1-refrate-1"
    #case_name_config = "519.1-refrate-2"
    #case_name_config = "557.1-refrate-1"

smoke_test = False

mape_line_analysis = True
hostname = socket.getfqdn(socket.gethostname())
#print(f"hostname={hostname}")

if 0:
    if 3 < len(sys.argv):
        exp_id = int(sys.argv[3])
        mape_line_analysis = True
    else:
        exp_id = None
else:
    if 3 < len(sys.argv):
        case_range = int(sys.argv[3])
    else:
        case_range = None

N_SAMPLES_ALL = 4 if smoke_test else 2304
N_SAMPLES_INIT = 2 if smoke_test else 25#100

#N_SRC_DOMAIN_TRAIN = N_SAMPLES_ALL
N_SRC_DOMAIN_TRAIN = 800 #500 #200
N_SRC_DOMAIN = 1#len(SRC_DOMAIN_LIST)


def get_SRC_DOMAIN_ENCODE_LIST_STR(src_domain_list):
    SRC_DOMAIN_ENCODE_LIST_STR = ''
    for domain_iter, case_name_iter in enumerate(src_domain_list):
        domain_id = get_domain_id(case_name_iter)
        SRC_DOMAIN_ENCODE_LIST_STR += "%02d" % (domain_id)
    return SRC_DOMAIN_ENCODE_LIST_STR


def get_domain_encode_map(case_name, src_domain_list):
    domain_encode_map = np.ones(len(case_names), dtype=int) * len(case_names)  # init
    for domain_iter, case_name_iter in enumerate(src_domain_list):
        domain_id = get_domain_id(case_name_iter)
        #print(f"{case_name_iter} -> {domain_id}")
        domain_encode_map[domain_id] = domain_iter
    domain_encode_map[get_domain_id(case_name)] = N_SRC_DOMAIN
    return domain_encode_map


def get_domain_encode_list(src_domain_list):
    SRC_DOMAIN_ENCODE_LIST = []
    for case_name_iter in src_domain_list:
        domain_id = get_domain_id(case_name_iter)
        SRC_DOMAIN_ENCODE_LIST.append(domain_id)
    return SRC_DOMAIN_ENCODE_LIST


def shuffle(target):
    for change in range(len(target) - 1, 0, -1):
        lower = random.randint(0, change)
        target[lower], target[change] = target[change], target[lower]


def get_src_domain_list(case_name, random_seed):
    src_domain_list = copy.deepcopy(case_names)
    src_domain_list.remove(case_name)
    #np.random.seed(random_seed)
    random.seed(random_seed)
    #shuffle(src_domain_list)
    random.shuffle(src_domain_list)
    src_domain_list = src_domain_list[:N_SRC_DOMAIN]
    return src_domain_list

'''
if smoke_test:
    SRC_DOMAIN_LIST = ["507.1-refrate-1"]
else:
    SRC_DOMAIN_LIST = get_src_domain_list(case_name)
'''

DOMAIN_ENCODE_ID = 0

#metric_name = "CPI"
metric_name = "Power"