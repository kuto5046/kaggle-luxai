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
kaggle competitions submit -c lux-ai-2021 -f submission.py -m "Message"
```

### Replay the game
./notebook/replay.ipynbを参照
