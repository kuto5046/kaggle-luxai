# $1=今日の日付
cd `dirname $0`
mkdir ./input/lux_ai_top_episodes_$1
cd ./input/lux_ai_top_episodes_$1
kaggle datasets download -d kuto0633/lux-ai-top-episodes
unzip lux-ai-top-episodes.zip
rm -f lux-ai-top-episodes.zip
cd `dirname $0`

