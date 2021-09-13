# rm snapshots/*.pkl
tar --exclude='*.ipynb' --exclude="*.pyc" --exclude="*.pkl" --exclude="*.sh" -czf submission.tar.gz *