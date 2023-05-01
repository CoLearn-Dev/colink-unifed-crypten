import os
import json
import sys
import subprocess
import tempfile
from typing import List

import colink as CL

from unifed.frameworks.crypten.util import store_error, store_return, GetTempFileName, get_local_ip

pop = CL.ProtocolOperator(__name__)
UNIFED_TASK_DIR = "unifed:task"

def load_config_from_param_and_check(param: bytes):
    unifed_config = json.loads(param.decode())
    framework = unifed_config["framework"]
    assert framework == "crypten"
    deployment = unifed_config["deployment"]
    if deployment["mode"] != "colink":
        raise ValueError("Deployment mode must be colink")
    return unifed_config


def run_crypten(cl: CL.CoLink, participants, participant_id, role: str, server_ip: str, unifed_config):
    communicator_args = {
        'WORLD_SIZE': str(len(participants)),
        'RANK': str(participant_id),
        'RENDEZVOUS': 'env://',
        'MASTER_ADDR': server_ip,
        'MASTER_PORT': '50000',
        'BACKEND': 'gloo'
    }
    # envs={}
    # for key, val in communicator_args.items():
    #     envs[key] = str(val)
    temp_conf_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    temp_conf_file.write(json.dumps(unifed_config))
    temp_conf_file.close()
    # start training procedure
    process = subprocess.Popen(
        [
            sys.executable,
            "run_crypten.py",
            temp_conf_file.name,
        ],
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        env={**os.environ, **communicator_args}
    )
    # gather result
    stdout, stderr = process.communicate()
    returncode = process.returncode
    # with open(temp_output_filename, "rb") as f:
    #     output = f.read()
    # cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:output", output)
    # with open(temp_log_filename, "rb") as f:
    #     log = f.read()
    # cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:log", log)
    return json.dumps({
        "server_ip": server_ip,
        "stdout": stdout.decode(),
        "stderr": stderr.decode(),
        "returncode": returncode,
    })


@pop.handle("unifed.crypten:client")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_client(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    unifed_config = load_config_from_param_and_check(param)
    # get the ip of the server
    if cl.get_participant_index(participants)==0:
        server_ip = get_local_ip()
        cl.send_variable("server_ip", server_ip, participants)
    else:
        server_ip = cl.recv_variable("server_ip", participants[0]).decode()
    # run crypten
    participant_id = [i for i, p in enumerate(participants) if p.user_id == cl.get_user_id()][0]
    return run_crypten(cl, participants, participant_id, "client", server_ip,unifed_config)
