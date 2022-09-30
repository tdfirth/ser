#file to match git commit with model run
from pathlib import Path
from git import Repo

def get_commit(path_root):
    #repo = Repo(path_root)
    #assert not repo.bare
    #commit = repo.commit.hexsha
    git_folder = Path(path_root,'.git')
    head_name = Path(git_folder, 'HEAD').read_text().split('\n')[0].split(' ')[-1]
    head_ref = Path(git_folder,head_name)
    commit = head_ref.read_text().replace('\n','')
    print('commit no : ', commit)
    return commit
