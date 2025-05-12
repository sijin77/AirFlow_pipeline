# clearml_init.py

import os
from clearml import Task


def init_clearml_from_env(task_name="MyClassifier"):

    project_name = os.getenv("CLEARML_PROJECT_NAME", "Digits Classification")
    # task_name = os.getenv("CLEARML_TASK_NAME", "MyClassifier")
    task_type = os.getenv("CLEARML_TASK_TYPE", "training")

    auto_connect_frameworks = (
        os.getenv("CLEARML_AUTO_CONNECT", "true").lower() == "true"
    )

    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type=task_type,
        auto_connect_frameworks=auto_connect_frameworks,
        reuse_last_task_id=False,
    )

    # Дополнительная настройка репозитория (если нужно)
    # repo_url = os.getenv("GIT_REPO_URL")
    # repo_branch = os.getenv("GIT_BRANCH")
    # repo_commit = os.getenv("GIT_COMMIT")

    # if repo_url:
    #     task.set_repo(repo=repo_url, branch=repo_branch, commit=repo_commit)

    print(f"[ClearML] Задача инициализирована: {task.id}")
    return task
