import git

def git_hash():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return(sha)
