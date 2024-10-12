# Deploying jobs on Kubernetes 

Set appropriate variables in `scripts/.env` file:

```bash

export JOB_NAME="uniq_id_example-gpu-job"               # required: your job name starting with unique prefix (e.g. your initials, name+surname etc.)
export CONTAINER_NAME="gpu-experiment-container"        # required: your container name
export JOB_PRIORITY="low"                               # required: none, low, normal
export GIT_BRANCH="master"                              # required: git branch of our repo you wish to copy
export GPU_MODEL="rtx3090"                              # required: if running a GPU job, one of: gtx1080ti, rtx3090, rtx2080ti
export PYTHON_SCRIPT_PATH="path/to/script.py"           # required: if running job: path to python script file relative to project root
export WANDB_API_KEY="your-api-key"                     # optional: your wandb API key 
export WANDB_PROJECT_NAME="my-wandb-project"            # optional: your wandb project name
export WANDB_NOTES="my-wandb-project-notes"             # optional: your wandb run notes


```

*Specify the resources* and run:

First remember to connect to the cluster using SSH.

``` bash
source scripts/.env
envsubst < jobs/<job-template>.yml | kubectl apply -f -
```

When the job is finished remember to delete it if you no longer need it:

```bash
kubectl delete job $JOB_NAME
```

Remember that when you want to run a job that already exists you will get an error, so remember to delete jobs and specify unique prefix in your job name e.g. your initials, name+surname etc.

If you are running a job you can specify the resources in the container tab. You can request the following resources:

- `ephemeral-storage`: how much storage to request for the container's temporary filesystem, this storage will be deleted when the job completes, a minimum amout of "1Gi" is needed in order to install packages,   for example to requests 2GB of storage: "2Gi"  
- `cpu`: how many CPUs to request, for example: "2" 
- `memory`: how much RAM to request, for example: "2Gi"
- `nvidia.com/gpu`: how many GPUs to request, for example: 2

The values in the `requests` tab define the minimal amount of resources that will be provided by 
the cluster. The values in `limits` tab specify the upper limits of resources, going over 
the values provided in `limits` will terminate the job. <br>

Example request spec:

```json
resources:
  requests:
    ephemeral-storage: "2Gi"
    cpu: "1"
    memory: "2Gi"
    nvidia.com/gpu: 1
  limits:
    ephemeral-storage: "4Gi"
    cpu: "1"
    memory: "4Gi"
    nvidia.com/gpu: 1
```

## Useful commands
- to run a job: `envsubst < jobs/job-template.yml | kubectl apply -f -`
- to delete a job: `kubectl delete job $JOB_NAME`
- to check job's status `kubectl describe job $JOB_NAME`
- to check stdout of your script: `kubectl logs -f jobs.batch/$JOB_NAME -c $CONTAINER_NAME`

## Container setup
The container is set up by cloning the desired branch of our repo inside `/repo/awesome-information-retrieval`. Next it `cd`'s into this directory and 
 runs `pip install -r requirements.txt` file from inside our repo. Finally it runs the script `python3 $PYTHON_SCRIPT_PATH`.


## Note on directories
If your job needs to create/download temporary files you can freely create them ONLY in `/repo/awesome-information-retrieval` or inside 
`/tmp`directory, the container does not have access to write to other directories. If your job fails because of file permission issues 
you should specify these directories. These files will be deleted after the job ends. If you want to store files permanently use `/nas` directory - this should be used
for big files (GBs) that will be reused often.

## Jobs vs pods
Jobs are meant to run for long periods of time until completion of your python script. You also need to specify the resources for a job. 
If you wish for an interactive/debug experience you can run a pod. In order to do that do not change the `echo 'started pod'; while :; do sleep 3600; done` 
line in pod command args. You can enter the pod by running `kubectl exec -it $JOB_NAME /usr/bin/bash` **AND THEN RUN** `export PYTHONUSERBASE=/tmp/packages`
`export PYTHONPATH=$PYTHONPATH:/repo/awesome-information-retrieval `inside the shell. Without this step all the packages installed via `requirements.txt` will not be visible to python 
and relative imports from inside our project will not be resolved.

## Custom commands to run on container start-up
You will need to specify custom commands if you want to export additional 
environment variables for weights and biases logging (see https://docs.wandb.ai/guides/track/environment-variables).
All the commands that are run in the container are specified in the `args` tab, you can change/add commands there by chaining `&&`
with previous commands:

```json
      command: 
        - "/bin/bash"
      args: 
        - -c  
        - | 
           export WANDB_API_KEY=${WANDB_API_KEY} &&
           export WANDB_NAME=${WANDB_JOB_NAME} &&
           export WANDB_NOTES=${WANDB_NOTES} &&
           export PYTHONUSERBASE=/tmp/packages && 
           export PYTHONPATH=$PYTHONPATH:/repo/awesome-information-retrieval &&
           cd /repo/awesome-information-retrieval && 
           pip install --user -r requirements.txt && 
           python3 ${PYTHON_SCRIPT_PATH} &&
           your command here
    

```
