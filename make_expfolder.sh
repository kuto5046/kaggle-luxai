# $1=exp-name
cd `dirname $0`
mkdir ~/work/exp/$1
cp ~/work/exp/$2/agent.py ~/work/exp/$1
cp ~/work/exp/$2/train.py ~/work/exp/$1
