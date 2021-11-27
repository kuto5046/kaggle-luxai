# Lux AI

## Log


## Result

## Usage

### Download episodes dataset
```sh
sh load_dataset.sh <TODAY_DATE>
```

### Make the new exp folder
```sh
sh make_expfolder.sh <NEW_EXP_NAME> <COPY_EXP_NAME>
```

### Make submission file
```sh
sh submission.sh <AGENT_PATH> 
```

### Submit file to kaggle
```sh
kaggle competitions submit -c lux-ai-2021 -f submission.tar.gz -m "Message"
```

### Replay the game
./notebook/replay.ipynbを参照

### CLIで対戦
指定したagent同士で対戦
```
sudo lux-ai-2021 --python=python3 <main.py of path> <main.py of path>
```

replayデータとして出力されるjsonファイルを以下のサイトにアップロードする
https://2021vis.lux-ai.org/

LBのagentと対戦
```
sudo lux-ai-2021 --python=python3 --rankSystem="trueskill" --tournament <path/to/agent1> <path/to/agent2>...
```