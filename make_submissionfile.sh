# 引数に対象のpathを指定
# ex)$1=./exp/exp004/
cd `dirname $0`
cd $1
cp -r ~/work/input/lux-ai-2021/lux .
cp -r ~/work/input/lux-ai-2021/main.py .
tar --exclude='*.ipynb' --exclude="*.tar.gz" --exclude="*.log" --exclude="wandb" -czf submission.tar.gz *
cd `dirname $0`