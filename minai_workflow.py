import os
import sys
import uuid
import kubernetes_models
import yaml
from kubernetes_models.models.io.k8s.api.core.v1 import Container
import pdb
import minai.contrib
from minai import generate_argo_workflow
from minai.model import Task, Workflow
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--option")
    parser.add_argument("--gpu", type=int, default=8)
    parser.add_argument("--vram24", action='store_true')
    args = parser.parse_args()
    
    workflow_name = f"ulas-{uuid.uuid4().hex[:8]}"
    working_dir = f"/mnt/vol21/masomatics/ULASclone"
    local_lib = '/home/masomatics/.local/lib/python3.9/site-packages'

    user_command = f"""
pip install -r requirements.txt
python -u run.py {args.option}
"""

    task = Task(
        name="train-ulas",
        container=Container(
            name="main",
            image="asia-northeast1-docker.pkg.dev/pfn-artifactregistry/all-in-one/stable:latest",
            command=["sh", "-c"],
            args=[user_command],
            working_dir=working_dir,
        ),
    )

    
    gpu_vram = "24gb" if args.vram24 else None
    minai.contrib.mnj.setup(
        task=task,
        activity_code=3000,
        cpu=25,
        gpu=args.gpu,
        gpu_vram=gpu_vram,
        memory=(300, "Gi"),
        #ephemeral_storage=(200, "Me"),
        pfs="pfs",
        retry_limit=5,
    )

    wf = Workflow(name=workflow_name, tasks=[task])
    argo_wf = generate_argo_workflow(wf)
    manifest = kubernetes_models.asdict(argo_wf)

    yaml.dump_all([manifest], sys.stdout)


if __name__ == "__main__":
    main()