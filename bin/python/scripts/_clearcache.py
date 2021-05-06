import glob,shutil

cache_dirs = [f for f in glob.glob("**/__pycache__", recursive=True)]

for dir in cache_dirs:
    shutil.rmtree(dir)