#!/bin/bash -e

REPO_FOLDER=$(dirname $(dirname $(readlink -f $0)))
JOB_NAME=$(sed 's/ /,/g' <<< "$@")
NOTIFICATION_EMAIL=

CONDA_ENV_NAME=QNetPy8
CONDA_ENV_PATH=~/QNetPy8

NODE_NAME=ct160
NODE_COUNT=4
NCUPS_PER_NODE=40
RUN_ID=$(date +%s%N)

LOG_FILE=~/.pbs_history/log/run.$(date +%s%N).log

PROJECT_ID=MST111245

mkdir -p ~/.pbs_history/script
mkdir -p ~/.pbs_history/log

ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --num_nodes)
      NODE_COUNT="$2"
      shift 2 # past argument
      ;;
    --node_name)
      NODE_NAME="$2"
      shift 2 # past argument
      ;;
    --notification_email)
      NOTIFICATION_EMAIL="$2"
      shift 2 # past argument
      ;;
    --pbs_log_file)
      LOG_FILE="$2"
      shift 2 # past argument
      ;;
    *)
      ARGS+=("$1")
      shift # past argument
      ;;
  esac
done

set -- "${ARGS[@]}" # restore parameters

echo '#!/bin/bash'"""
echo \"On node: \$(hostname)\"
module load anaconda3/5.1.10
# Set environments for CPU
export KMP_AFFINITY=\"granularity=fine,verbose,compact,1,0\"
export KMP_SETTINGS=1
export OMP_NUM_THREADS=$NCUPS_PER_NODE
export KMP_BLOCKTIME=30
#run python script with inputs from this .sh script
$CONDA_ENV_PATH/bin/python $REPO_FOLDER/train.py \$@ $@
""" > ~/.pbs_history/script/run.$RUN_ID.sh

EMAIL_REGEX="^[a-zA-Z0-9\._-]+\@[a-zA-Z0-9._-]+\.[a-zA-Z]+\$"
echo '#!/bin/bash'"""
#PBS -l select=$NODE_COUNT:ncpus=$NCUPS_PER_NODE
#PBS -N $JOB_NAME
#PBS -q $NODE_NAME
#PBS -P $PROJECT_ID
#PBS -j oe
#PBS -m abe
$(if [[ $NOTIFICATION_EMAIL =~ $EMAIL_REGEX ]] ; then echo "#PBS -M $NOTIFICATION_EMAIL"; fi)

echo \"The base node is: \$(hostname)\" >> $LOG_FILE 2>&1
# get node information from nodefile
NODES=(\$( cat \$PBS_NODEFILE | uniq ))
NUM_OF_NODES=\${#NODES[@]}
echo \"All nodes: \${NODES[*]}\" >> $LOG_FILE 2>&1
# worker number for other nodes
C=1
# for each node that is not the current node
for node in \${NODES[@]}
do
  if [[ \$node != \$(eval hostname) ]]
  then
    # ssh into each node and run the .sh script with node info
    # run in background
    ssh \$node \"~/.pbs_history/script/run.$RUN_ID.sh --distributed_node_index \$C --distributed_nodes \${NODES[*]} $@\" &
    C=\$((\$C+1))
    sleep 2
  fi
done
~/.pbs_history/script/run.$RUN_ID.sh --distributed_node_index 0 --distributed_nodes \${NODES[*]} $@ >> $LOG_FILE 2>&1
echo \"Done with PBS\"
""" > ~/.pbs_history/script/pbs.$RUN_ID.sh

module load anaconda3/5.1.10

if ! { conda env list | grep $CONDA_ENV_NAME; } >/dev/null 2>&1; then
  conda create -yp $CONDA_ENV_PATH
  conda install -yp $CONDA_ENV_PATH python==3.8
fi

echo "IMPORTANT! you should install pacakge before running:"
echo "$CONDA_ENV_PATH/bin/pip install -r $REPO_FOLDER/requirements.txt >/dev/null 2>&1"

chmod u+x ~/.pbs_history/script/run.$RUN_ID.sh ~/.pbs_history/script/pbs.$RUN_ID.sh
echo "[Job ID: $(qsub ~/.pbs_history/script/pbs.$RUN_ID.sh)] Start running..." > $LOG_FILE 2>&1
echo "tail -f $LOG_FILE"