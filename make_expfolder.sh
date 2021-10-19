# $1=exp-name
# $2= flag rl or None
cd `dirname $0`
mkdir ~/work/exp/$1
cp ~/work/exp/$2/agent.py ~/work/exp/$1
cp ~/work/exp/$2/train.py ~/work/exp/$1

if [ $2 =="rl" ]; then
    cp ~/work/LuxPythonEnvGym/lixai2021 ~/work/exp/$1
fi
