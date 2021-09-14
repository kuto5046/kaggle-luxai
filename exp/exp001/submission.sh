cp -r ../lux/ .
rm -f snapshots/*.pkl
tar --exclude='*.ipynb' --exclude="*.pyc" --exclude="*.pkl" --exclude="*.tar.gz" --exclude="*.sh" -czf submission.tar.gz *