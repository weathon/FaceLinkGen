"""Upload and download the checkpoints and artifacts that Git does not track.

    python tools/hf_artifacts.py upload checkpoints
    python tools/hf_artifacts.py upload artifacts
    python tools/hf_artifacts.py download checkpoints

HF_TOKEN comes from the repository-root .env. The Hub layout mirrors the working tree, so
downloads land in the paths the scripts already read.
"""

import argparse
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("action", choices=["upload", "download"])
ap.add_argument("group", choices=["checkpoints", "artifacts"])
ap.add_argument("--repo", default="weathon/FaceLinkGen-artifacts")
args = ap.parse_args()

token = os.environ["HF_TOKEN"]
local = os.path.join(ROOT, args.group)

if args.action == "upload":
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    print("uploading %s -> %s:%s" % (local, args.repo, args.group))
    api.upload_folder(folder_path=local, path_in_repo=args.group,
                      repo_id=args.repo, repo_type="dataset")
else:
    print("downloading %s/* from %s" % (args.group, args.repo))
    snapshot_download(repo_id=args.repo, repo_type="dataset",
                      allow_patterns=[args.group + "/*"], local_dir=ROOT,
                      local_dir_use_symlinks=False, token=token)
print("done")
