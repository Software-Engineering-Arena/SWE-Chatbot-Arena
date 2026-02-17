# References for model evaluation metrics:
# - Chatbot Arena: https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH
# - Evalica: https://github.com/dustalov/evalica/blob/master/Chatbot-Arena.ipynb

import dotenv
import evalica
import gitlab
import io
import json
import os
import random
import threading
import warnings

import gradio as gr
import pandas as pd

from datetime import datetime
from github import Auth, Github
from urllib.parse import urlparse
from gradio_leaderboard import Leaderboard, ColumnFilter
from huggingface_hub import upload_file, hf_hub_download, HfApi
from openai import OpenAI

# Load environment variables
dotenv.load_dotenv(override=True)

# Initialize OpenAI Client
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = "https://openrouter.ai/api/v1"
openai_client = OpenAI(api_key=api_key, base_url=base_url)

# Hugging Face repository names for data storage
LEADERBOARD_REPO = "SWE-Arena/leaderboard_data"
VOTE_REPO = "SWE-Arena/vote_data"
CONVERSATION_REPO = "SWE-Arena/conversation_data"
LEADERBOARD_FILE = "chatbot_arena"
MODEL_REPO = "SWE-Arena/model_data"

# Timeout in seconds for model responses
TIMEOUT = 90

# Leaderboard update time frame in days
LEADERBOARD_UPDATE_TIME_FRAME_DAYS = 365

# Hint string constant
SHOW_HINT_STRING = True  # Set to False to hide the hint string altogether
HINT_STRING = "Once signed in, your votes will be recorded securely."

# Load model metadata from Hugging Face
model_context_window = {}
model_name_to_id = {}
model_organization = {}
available_models = []

_api = HfApi()
for _file in _api.list_repo_files(repo_id=MODEL_REPO, repo_type="dataset"):
    if not _file.endswith(".json"):
        continue
    _local_path = hf_hub_download(repo_id=MODEL_REPO, filename=_file, repo_type="dataset")
    with open(_local_path, "r") as f:
        _record = json.load(f)
    # model_name is derived from the filename (without .json extension)
    _model_name = _file.rsplit("/", 1)[-1].replace(".json", "")
    available_models.append(_model_name)
    model_context_window[_model_name] = _record["context_window"]
    model_name_to_id[_model_name] = _record["id"]
    model_organization[_model_name] = _model_name.split(": ")[0]


# ---------------------------------------------------------------------------
# URL parsing helpers
# ---------------------------------------------------------------------------

def _parse_url_path(url):
    """Parse a URL and return (hostname, path_segments).

    Returns:
        tuple: (hostname: str, segments: list[str]) where segments
               are the non-empty parts of the URL path.
        Returns (None, []) if URL cannot be parsed.
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        segments = [s for s in parsed.path.split("/") if s]
        return hostname, segments
    except Exception:
        return None, []


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------

def _classify_github_url(segments):
    """Classify a GitHub URL from its path segments into resource type + params."""
    if len(segments) < 2:
        return None

    owner, repo = segments[0], segments[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    base = {"owner": owner, "repo": repo}

    if len(segments) == 2:
        return {**base, "resource": None}

    res = segments[2]

    if res == "issues" and len(segments) >= 4:
        return {**base, "resource": "issues", "id": segments[3]}
    elif res == "pull" and len(segments) >= 4:
        return {**base, "resource": "pull", "id": segments[3]}
    elif res == "commit" and len(segments) >= 4:
        return {**base, "resource": "commit", "sha": segments[3]}
    elif res == "blob" and len(segments) >= 4:
        return {**base, "resource": "blob", "branch": segments[3],
                "path": "/".join(segments[4:]) if len(segments) > 4 else ""}
    elif res == "tree" and len(segments) >= 4:
        return {**base, "resource": "tree", "branch": segments[3],
                "path": "/".join(segments[4:]) if len(segments) > 4 else ""}
    elif res == "discussions" and len(segments) >= 4:
        return {**base, "resource": "discussions", "id": segments[3]}
    elif res == "releases" and len(segments) >= 5 and segments[3] == "tag":
        return {**base, "resource": "releases", "tag": segments[4]}
    elif res == "compare" and len(segments) >= 4:
        return {**base, "resource": "compare", "spec": segments[3]}
    elif res == "actions" and len(segments) >= 5 and segments[3] == "runs":
        return {**base, "resource": "actions", "run_id": segments[4]}
    elif res == "wiki":
        page = segments[3] if len(segments) >= 4 else None
        return {**base, "resource": "wiki", "page": page}
    else:
        return {**base, "resource": "unknown"}


# -- GitHub formatters -------------------------------------------------------

def _fmt_github_repo(repo):
    parts = [f"Repository: {repo.full_name}"]
    if repo.description:
        parts.append(f"Description: {repo.description}")
    try:
        readme = repo.get_readme()
        content = readme.decoded_content.decode("utf-8", errors="replace")
        parts.append(f"README (first 2000 chars):\n{content[:2000]}")
    except Exception:
        pass
    return "\n\n".join(parts)


def _fmt_github_issue(repo, issue_id):
    issue = repo.get_issue(issue_id)
    parts = [
        f"Issue #{issue.number}: {issue.title}",
        f"State: {issue.state}",
        f"Body:\n{issue.body or '(empty)'}",
    ]
    comments = issue.get_comments()
    comment_texts = []
    for i, c in enumerate(comments):
        if i >= 10:
            break
        comment_texts.append(f"  Comment by {c.user.login}:\n  {c.body}")
    if comment_texts:
        parts.append("Comments (first 10):\n" + "\n---\n".join(comment_texts))
    return "\n\n".join(parts)


def _fmt_github_pr(repo, pr_id):
    pr = repo.get_pull(pr_id)
    parts = [
        f"Pull Request #{pr.number}: {pr.title}",
        f"State: {pr.state}  Merged: {pr.merged}",
        f"Body:\n{pr.body or '(empty)'}",
    ]
    diff_parts = []
    for f in pr.get_files():
        header = f"--- {f.filename} ({f.status}, +{f.additions}/-{f.deletions})"
        patch = f.patch or "(binary or too large)"
        diff_parts.append(f"{header}\n{patch}")
    if diff_parts:
        diff_text = "\n\n".join(diff_parts)
        if len(diff_text) > 5000:
            diff_text = diff_text[:5000] + "\n... (diff truncated)"
        parts.append(f"Diff:\n{diff_text}")
    return "\n\n".join(parts)


def _fmt_github_commit(repo, sha):
    commit = repo.get_commit(sha)
    parts = [
        f"Commit: {commit.sha}",
        f"Message: {commit.commit.message}",
        f"Author: {commit.commit.author.name}",
        f"Stats: +{commit.stats.additions}/-{commit.stats.deletions}",
    ]
    file_patches = []
    for f in commit.files:
        file_patches.append(f"  {f.filename} ({f.status}): {f.patch or '(binary)'}")
    if file_patches:
        patch_text = "\n".join(file_patches)
        if len(patch_text) > 5000:
            patch_text = patch_text[:5000] + "\n... (patch truncated)"
        parts.append(f"Files changed:\n{patch_text}")
    return "\n\n".join(parts)


def _fmt_github_blob(repo, branch, path):
    contents = repo.get_contents(path, ref=branch)
    if isinstance(contents, list):
        listing = "\n".join(f"  {c.path} ({c.type})" for c in contents)
        return f"Directory listing at {branch}/{path}:\n{listing}"
    content = contents.decoded_content.decode("utf-8", errors="replace")
    if len(content) > 5000:
        content = content[:5000] + "\n... (content truncated)"
    return f"File: {path} (branch: {branch})\n\n{content}"


def _fmt_github_tree(repo, branch, path):
    if path:
        contents = repo.get_contents(path, ref=branch)
        if not isinstance(contents, list):
            contents = [contents]
    else:
        contents = repo.get_contents("", ref=branch)
    listing = "\n".join(f"  {c.path} ({c.type}, {c.size} bytes)" for c in contents)
    return f"Tree at {branch}/{path or '(root)'}:\n{listing}"


_DISCUSSION_GRAPHQL_SCHEMA = """
    title
    body
    number
    author { login }
    comments(first: 10) {
        nodes {
            body
            author { login }
        }
    }
"""


def _fmt_github_discussion(repo, discussion_id):
    try:
        discussion = repo.get_discussion(discussion_id, _DISCUSSION_GRAPHQL_SCHEMA)
        parts = [
            f"Discussion #{discussion.number}: {discussion.title}",
            f"Body:\n{discussion.body or '(empty)'}",
        ]
        if hasattr(discussion, "comments") and discussion.comments:
            comment_texts = []
            for c in discussion.comments:
                author = c.author.login if hasattr(c, "author") and c.author else "unknown"
                comment_texts.append(f"  Comment by {author}: {c.body}")
            if comment_texts:
                parts.append("Comments:\n" + "\n---\n".join(comment_texts))
        return "\n\n".join(parts)
    except Exception as e:
        print(f"Discussion fetch failed (GraphQL): {e}")
        return None


def _fmt_github_release(repo, tag):
    release = repo.get_release(tag)
    parts = [
        f"Release: {release.title or release.tag_name}",
        f"Tag: {release.tag_name}",
        f"Body:\n{release.body or '(empty)'}",
    ]
    return "\n\n".join(parts)


def _fmt_github_compare(repo, spec):
    if "..." in spec:
        base, head = spec.split("...", 1)
    elif ".." in spec:
        base, head = spec.split("..", 1)
    else:
        return None
    comparison = repo.compare(base, head)
    parts = [
        f"Comparison: {base}...{head}",
        f"Status: {comparison.status}",
        f"Ahead by: {comparison.ahead_by}, Behind by: {comparison.behind_by}",
        f"Total commits: {comparison.total_commits}",
    ]
    commit_summaries = []
    for c in comparison.commits[:20]:
        commit_summaries.append(f"  {c.sha[:8]}: {c.commit.message.splitlines()[0]}")
    if commit_summaries:
        parts.append("Commits:\n" + "\n".join(commit_summaries))
    file_summaries = []
    for f in comparison.files[:30]:
        file_summaries.append(f"  {f.filename} ({f.status}, +{f.additions}/-{f.deletions})")
    if file_summaries:
        parts.append("Files changed:\n" + "\n".join(file_summaries))
    return "\n\n".join(parts)


def _fmt_github_actions(repo, run_id):
    run = repo.get_workflow_run(run_id)
    parts = [
        f"Workflow Run: {run.name} #{run.run_number}",
        f"Status: {run.status}  Conclusion: {run.conclusion}",
        f"SHA: {run.head_sha}",
    ]
    try:
        jobs = run.jobs()
        for job in jobs:
            if job.conclusion == "failure":
                parts.append(f"Failed job: {job.name}")
                for step in job.steps:
                    if step.conclusion == "failure":
                        parts.append(f"  Failed step: {step.name}")
    except Exception:
        pass
    return "\n\n".join(parts)


def _fmt_github_wiki(owner, repo_name, page):
    if page:
        return f"Wiki page: {page} (from {owner}/{repo_name}/wiki)\nNote: Wiki content cannot be fetched via API."
    return f"Wiki: {owner}/{repo_name}/wiki\nNote: Wiki content cannot be fetched via API."


def fetch_github_content(url):
    """Fetch detailed content from a GitHub URL using PyGithub."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not set.")
        return None

    g = Github(auth=Auth.Token(token))
    hostname, segments = _parse_url_path(url)

    if not hostname or "github.com" not in hostname:
        return None

    info = _classify_github_url(segments)
    if not info:
        return None

    try:
        repo = g.get_repo(f"{info['owner']}/{info['repo']}")
        resource = info["resource"]

        if resource is None:
            return _fmt_github_repo(repo)
        elif resource == "issues":
            return _fmt_github_issue(repo, int(info["id"]))
        elif resource == "pull":
            return _fmt_github_pr(repo, int(info["id"]))
        elif resource == "commit":
            return _fmt_github_commit(repo, info["sha"])
        elif resource == "blob":
            return _fmt_github_blob(repo, info["branch"], info["path"])
        elif resource == "tree":
            return _fmt_github_tree(repo, info["branch"], info.get("path", ""))
        elif resource == "discussions":
            return _fmt_github_discussion(repo, int(info["id"]))
        elif resource == "releases":
            return _fmt_github_release(repo, info["tag"])
        elif resource == "compare":
            return _fmt_github_compare(repo, info["spec"])
        elif resource == "actions":
            return _fmt_github_actions(repo, int(info["run_id"]))
        elif resource == "wiki":
            return _fmt_github_wiki(info["owner"], info["repo"], info.get("page"))
        else:
            return None
    except Exception as e:
        print(f"GitHub API error: {e}")
        return None


# ---------------------------------------------------------------------------
# GitLab
# ---------------------------------------------------------------------------

def _classify_gitlab_url(segments):
    """Classify a GitLab URL from its path segments.

    GitLab uses /-/ as separator between project path and resource.
    Project paths can be nested: group/subgroup/project.
    """
    try:
        dash_idx = segments.index("-")
    except ValueError:
        # No /-/ separator -- treat all segments as the project path
        if len(segments) >= 2:
            return {"project_path": "/".join(segments), "resource": None}
        return None

    project_path = "/".join(segments[:dash_idx])
    res_segments = segments[dash_idx + 1:]

    if not project_path or not res_segments:
        return {"project_path": project_path, "resource": None}

    res = res_segments[0]

    if res == "issues" and len(res_segments) >= 2:
        return {"project_path": project_path, "resource": "issues", "id": res_segments[1]}
    elif res == "merge_requests" and len(res_segments) >= 2:
        return {"project_path": project_path, "resource": "merge_requests", "id": res_segments[1]}
    elif res in ("commit", "commits") and len(res_segments) >= 2:
        return {"project_path": project_path, "resource": "commit", "sha": res_segments[1]}
    elif res == "blob" and len(res_segments) >= 2:
        branch = res_segments[1]
        file_path = "/".join(res_segments[2:]) if len(res_segments) > 2 else ""
        return {"project_path": project_path, "resource": "blob", "branch": branch, "path": file_path}
    elif res == "tree" and len(res_segments) >= 2:
        branch = res_segments[1]
        tree_path = "/".join(res_segments[2:]) if len(res_segments) > 2 else ""
        return {"project_path": project_path, "resource": "tree", "branch": branch, "path": tree_path}
    elif res == "releases" and len(res_segments) >= 2:
        return {"project_path": project_path, "resource": "releases", "tag": res_segments[1]}
    elif res == "compare" and len(res_segments) >= 2:
        return {"project_path": project_path, "resource": "compare", "spec": res_segments[1]}
    elif res == "pipelines" and len(res_segments) >= 2:
        return {"project_path": project_path, "resource": "pipelines", "id": res_segments[1]}
    elif res == "wikis":
        page = res_segments[1] if len(res_segments) >= 2 else None
        return {"project_path": project_path, "resource": "wikis", "page": page}
    else:
        return {"project_path": project_path, "resource": "unknown"}


# -- GitLab formatters -------------------------------------------------------

def _fmt_gitlab_repo(project):
    parts = [f"Repository: {project.path_with_namespace}"]
    if project.description:
        parts.append(f"Description: {project.description}")
    try:
        readme = project.files.get(file_path="README.md", ref=project.default_branch)
        content = readme.decode().decode("utf-8", errors="replace")
        parts.append(f"README (first 2000 chars):\n{content[:2000]}")
    except Exception:
        pass
    return "\n\n".join(parts)


def _fmt_gitlab_issue(project, issue_id):
    issue = project.issues.get(issue_id)
    parts = [
        f"Issue #{issue.iid}: {issue.title}",
        f"State: {issue.state}",
        f"Body:\n{issue.description or '(empty)'}",
    ]
    notes = issue.notes.list(get_all=False, per_page=10)
    note_texts = [f"  Comment by {n.author['username']}: {n.body}" for n in notes]
    if note_texts:
        parts.append("Comments (first 10):\n" + "\n---\n".join(note_texts))
    return "\n\n".join(parts)


def _fmt_gitlab_mr(project, mr_id):
    mr = project.mergerequests.get(mr_id)
    parts = [
        f"Merge Request !{mr.iid}: {mr.title}",
        f"State: {mr.state}",
        f"Body:\n{mr.description or '(empty)'}",
    ]
    try:
        changes = mr.changes()
        if isinstance(changes, dict) and "changes" in changes:
            diff_parts = []
            for change in changes["changes"][:30]:
                diff_parts.append(f"  {change.get('new_path', '?')}: {change.get('diff', '')[:500]}")
            if diff_parts:
                diff_text = "\n".join(diff_parts)
                if len(diff_text) > 5000:
                    diff_text = diff_text[:5000] + "\n... (diff truncated)"
                parts.append(f"Changes:\n{diff_text}")
    except Exception:
        pass
    return "\n\n".join(parts)


def _fmt_gitlab_commit(project, sha):
    commit = project.commits.get(sha)
    parts = [
        f"Commit: {commit.id}",
        f"Title: {commit.title}",
        f"Message: {commit.message}",
        f"Author: {commit.author_name}",
    ]
    try:
        diffs = commit.diff()
        diff_parts = []
        for d in diffs[:30]:
            diff_parts.append(f"  {d.get('new_path', '?')}: {d.get('diff', '')[:500]}")
        if diff_parts:
            diff_text = "\n".join(diff_parts)
            if len(diff_text) > 5000:
                diff_text = diff_text[:5000] + "\n... (diff truncated)"
            parts.append(f"Diff:\n{diff_text}")
    except Exception:
        pass
    return "\n\n".join(parts)


def _fmt_gitlab_blob(project, branch, path):
    f = project.files.get(file_path=path, ref=branch)
    content = f.decode().decode("utf-8", errors="replace")
    if len(content) > 5000:
        content = content[:5000] + "\n... (content truncated)"
    return f"File: {path} (branch: {branch})\n\n{content}"


def _fmt_gitlab_tree(project, branch, path):
    items = project.repository_tree(path=path or "", ref=branch, get_all=False, per_page=100)
    listing = "\n".join(f"  {item['path']} ({item['type']})" for item in items)
    return f"Tree at {branch}/{path or '(root)'}:\n{listing}"


def _fmt_gitlab_release(project, tag):
    release = project.releases.get(tag)
    parts = [
        f"Release: {release.name or release.tag_name}",
        f"Tag: {release.tag_name}",
        f"Description:\n{release.description or '(empty)'}",
    ]
    return "\n\n".join(parts)


def _fmt_gitlab_compare(project, spec):
    if "..." in spec:
        base, head = spec.split("...", 1)
    elif ".." in spec:
        base, head = spec.split("..", 1)
    else:
        return None
    result = project.repository_compare(base, head)
    parts = [f"Comparison: {base}...{head}"]
    if isinstance(result, dict):
        commits = result.get("commits", [])
        commit_summaries = []
        for c in commits[:20]:
            commit_summaries.append(f"  {c.get('short_id', '?')}: {c.get('title', '')}")
        if commit_summaries:
            parts.append("Commits:\n" + "\n".join(commit_summaries))
        diffs = result.get("diffs", [])
        diff_parts = []
        for d in diffs[:30]:
            diff_parts.append(f"  {d.get('new_path', '?')}: {d.get('diff', '')[:500]}")
        if diff_parts:
            diff_text = "\n".join(diff_parts)
            if len(diff_text) > 5000:
                diff_text = diff_text[:5000] + "\n... (diff truncated)"
            parts.append(f"Diffs:\n{diff_text}")
    return "\n\n".join(parts)


def _fmt_gitlab_pipeline(project, pipeline_id):
    pipeline = project.pipelines.get(pipeline_id)
    parts = [
        f"Pipeline #{pipeline.id}",
        f"Status: {pipeline.status}",
        f"Ref: {pipeline.ref}",
        f"SHA: {pipeline.sha}",
    ]
    try:
        jobs = pipeline.jobs.list(get_all=False, per_page=20)
        failed_jobs = [j for j in jobs if j.status == "failed"]
        if failed_jobs:
            parts.append("Failed jobs:")
            for j in failed_jobs:
                parts.append(f"  {j.name}: {j.status} (stage: {j.stage})")
    except Exception:
        pass
    return "\n\n".join(parts)


def _fmt_gitlab_wiki(project, page):
    if page:
        try:
            wiki_page = project.wikis.get(page)
            return f"Wiki page: {wiki_page.title}\n\n{wiki_page.content}"
        except Exception:
            return f"Wiki page: {page}\nNote: Could not fetch wiki page content."
    try:
        pages = project.wikis.list(get_all=False, per_page=20)
        listing = "\n".join(f"  {p.slug}: {p.title}" for p in pages)
        return f"Wiki pages:\n{listing}"
    except Exception:
        return "Wiki: Could not fetch wiki pages."


def fetch_gitlab_content(url):
    """Fetch content from GitLab URL using python-gitlab."""
    token = os.getenv("GITLAB_TOKEN")
    if not token:
        print("GITLAB_TOKEN not set.")
        return None

    gl = gitlab.Gitlab("https://gitlab.com", private_token=token)
    hostname, segments = _parse_url_path(url)

    if not hostname or "gitlab.com" not in hostname:
        return None

    info = _classify_gitlab_url(segments)
    if not info:
        return None

    try:
        project = gl.projects.get(info["project_path"])
        resource = info["resource"]

        if resource is None:
            return _fmt_gitlab_repo(project)
        elif resource == "issues":
            return _fmt_gitlab_issue(project, int(info["id"]))
        elif resource == "merge_requests":
            return _fmt_gitlab_mr(project, int(info["id"]))
        elif resource == "commit":
            return _fmt_gitlab_commit(project, info["sha"])
        elif resource == "blob":
            return _fmt_gitlab_blob(project, info["branch"], info["path"])
        elif resource == "tree":
            return _fmt_gitlab_tree(project, info["branch"], info.get("path", ""))
        elif resource == "releases":
            return _fmt_gitlab_release(project, info["tag"])
        elif resource == "compare":
            return _fmt_gitlab_compare(project, info["spec"])
        elif resource == "pipelines":
            return _fmt_gitlab_pipeline(project, int(info["id"]))
        elif resource == "wikis":
            return _fmt_gitlab_wiki(project, info.get("page"))
        else:
            return None
    except Exception as e:
        print(f"GitLab API error: {e}")
        return None


# ---------------------------------------------------------------------------
# HuggingFace
# ---------------------------------------------------------------------------

def _classify_huggingface_url(segments):
    """Classify a HuggingFace URL from its path segments.

    HF URLs:
    - huggingface.co/{user}/{repo}           -> model
    - huggingface.co/datasets/{user}/{repo}  -> dataset
    - huggingface.co/spaces/{user}/{repo}    -> space
    """
    if not segments:
        return None

    # Detect repo_type prefix
    repo_type = None
    segs = list(segments)
    if segs[0] in ("datasets", "spaces"):
        repo_type = segs[0].rstrip("s")  # "dataset" or "space"
        segs = segs[1:]

    if len(segs) < 2:
        return None

    repo_id = f"{segs[0]}/{segs[1]}"
    base = {"repo_id": repo_id, "repo_type": repo_type}

    if len(segs) == 2:
        return {**base, "resource": None}

    res = segs[2]

    if res == "blob" and len(segs) >= 4:
        return {**base, "resource": "blob", "revision": segs[3],
                "path": "/".join(segs[4:]) if len(segs) > 4 else ""}
    elif res == "resolve" and len(segs) >= 4:
        return {**base, "resource": "resolve", "revision": segs[3],
                "path": "/".join(segs[4:]) if len(segs) > 4 else ""}
    elif res == "tree" and len(segs) >= 4:
        return {**base, "resource": "tree", "revision": segs[3],
                "path": "/".join(segs[4:]) if len(segs) > 4 else ""}
    elif res == "commit" and len(segs) >= 4:
        return {**base, "resource": "commit", "sha": segs[3]}
    elif res == "discussions" and len(segs) >= 4:
        return {**base, "resource": "discussions", "num": segs[3]}
    else:
        return {**base, "resource": "unknown"}


# -- HuggingFace formatters --------------------------------------------------

def _fmt_hf_repo(api, repo_id, repo_type):
    info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
    parts = [f"Repository: {repo_id}"]
    if hasattr(info, "description") and info.description:
        parts.append(f"Description: {info.description}")
    if hasattr(info, "card_data") and info.card_data:
        parts.append(f"Card data: {str(info.card_data)[:1000]}")
    try:
        readme_path = api.hf_hub_download(
            repo_id=repo_id, filename="README.md", repo_type=repo_type
        )
        with open(readme_path, "r", errors="replace") as f:
            content = f.read()[:2000]
        parts.append(f"README (first 2000 chars):\n{content}")
    except Exception:
        pass
    return "\n\n".join(parts)


def _fmt_hf_commit(api, repo_id, repo_type, sha):
    commits = api.list_repo_commits(repo_id=repo_id, revision=sha, repo_type=repo_type)
    if commits:
        c = commits[0]
        return (
            f"Commit: {c.commit_id}\n"
            f"Title: {c.title}\n"
            f"Message: {c.message}\n"
            f"Authors: {', '.join(c.authors) if c.authors else 'unknown'}\n"
            f"Date: {c.created_at}"
        )
    return None


def _fmt_hf_discussion(api, repo_id, repo_type, discussion_num):
    discussion = api.get_discussion_details(
        repo_id=repo_id, discussion_num=discussion_num, repo_type=repo_type
    )
    parts = [
        f"Discussion #{discussion.num}: {discussion.title}",
        f"Status: {discussion.status}",
        f"Author: {discussion.author}",
        f"Is Pull Request: {discussion.is_pull_request}",
    ]
    comment_texts = []
    for event in discussion.events:
        if hasattr(event, "content") and event.content:
            author = event.author if hasattr(event, "author") else "unknown"
            comment_texts.append(f"  {author}: {event.content[:500]}")
        if len(comment_texts) >= 10:
            break
    if comment_texts:
        parts.append("Comments:\n" + "\n---\n".join(comment_texts))
    return "\n\n".join(parts)


def _fmt_hf_file(api, repo_id, repo_type, revision, path):
    local_path = api.hf_hub_download(
        repo_id=repo_id, filename=path, revision=revision, repo_type=repo_type
    )
    try:
        with open(local_path, "r", errors="replace") as f:
            content = f.read()
        if len(content) > 5000:
            content = content[:5000] + "\n... (content truncated)"
        return f"File: {path} (revision: {revision})\n\n{content}"
    except Exception:
        return f"File: {path} (revision: {revision})\n(binary or unreadable file)"


def _fmt_hf_tree(api, repo_id, repo_type, revision, path):
    items = api.list_repo_tree(
        repo_id=repo_id, path_in_repo=path or None,
        revision=revision, repo_type=repo_type
    )
    listing = []
    for item in items:
        if hasattr(item, "size") and item.size is not None:
            listing.append(f"  {item.rfilename} (file, {item.size} bytes)")
        else:
            listing.append(f"  {item.rfilename} (folder)")
        if len(listing) >= 100:
            listing.append("  ... (truncated)")
            break
    return f"Tree at {revision}/{path or '(root)'}:\n" + "\n".join(listing)


def fetch_huggingface_content(url):
    """Fetch detailed content from a Hugging Face URL using huggingface_hub API."""
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set.")
        return None

    api = HfApi(token=token)
    hostname, segments = _parse_url_path(url)

    if not hostname or "huggingface.co" not in hostname:
        return None

    info = _classify_huggingface_url(segments)
    if not info:
        return None

    try:
        resource = info["resource"]
        repo_id = info["repo_id"]
        repo_type = info["repo_type"]

        if resource is None:
            return _fmt_hf_repo(api, repo_id, repo_type)
        elif resource == "commit":
            return _fmt_hf_commit(api, repo_id, repo_type, info["sha"])
        elif resource == "discussions":
            return _fmt_hf_discussion(api, repo_id, repo_type, int(info["num"]))
        elif resource in ("blob", "resolve"):
            return _fmt_hf_file(api, repo_id, repo_type, info["revision"], info["path"])
        elif resource == "tree":
            return _fmt_hf_tree(api, repo_id, repo_type, info["revision"], info.get("path", ""))
        else:
            return None
    except Exception as e:
        print(f"Hugging Face API error: {e}")
        return None


# ---------------------------------------------------------------------------
# URL router
# ---------------------------------------------------------------------------

def fetch_url_content(url):
    """Main URL content fetcher that routes to platform-specific handlers."""
    if not url or not url.strip():
        return ""
    url = url.strip()
    try:
        hostname, _ = _parse_url_path(url)
        if hostname and "github.com" in hostname:
            return fetch_github_content(url)
        elif hostname and "gitlab.com" in hostname:
            return fetch_gitlab_content(url)
        elif hostname and "huggingface.co" in hostname:
            return fetch_huggingface_content(url)
    except Exception as e:
        print(f"Error fetching URL content: {e}")
    return ""


# Truncate prompt
def truncate_prompt(model_alias, models, conversation_state):
    """
    Truncate the conversation history and user input to fit within the model's context window.

    Args:
        model_alias (str): Alias for the model being used (i.e., "left", "right").
        models (dict): Dictionary mapping model aliases to their names.
        conversation_state (dict): State containing the conversation history for all models.

    Returns:
        str: Truncated conversation history and user input.
    """
    # Get the full conversation history for the model
    full_conversation = conversation_state[f"{model_alias}_chat"]

    # Get the context length for the model
    context_length = model_context_window[models[model_alias]]

    # Single loop to handle both FIFO removal and content truncation
    while len(json.dumps(full_conversation)) > context_length:
        # If we have more than one message, remove the oldest (FIFO)
        if len(full_conversation) > 1:
            full_conversation.pop(0)
        # If only one message remains, truncate its content
        else:
            current_length = len(json.dumps(full_conversation))
            # Calculate how many characters we need to remove
            excess = current_length - context_length
            # Add a buffer to ensure we remove enough (accounting for JSON encoding)
            truncation_size = min(excess + 10, len(full_conversation[0]["content"]))

            if truncation_size <= 0:
                break  # Can't truncate further

            # Truncate the content from the end to fit
            full_conversation[0]["content"] = full_conversation[0]["content"][
                :-truncation_size
            ]

    return full_conversation


def chat_with_models(model_alias, models, conversation_state, timeout=TIMEOUT):
    truncated_input = truncate_prompt(model_alias, models, conversation_state)
    response_event = threading.Event()  # Event to signal response completion
    model_response = {"content": None, "error": None}

    def request_model_response():
        try:
            # Get model_id from the model_name using the mapping
            model_name = models[model_alias]
            model_id = model_name_to_id.get(model_name, model_name)
            request_params = {"model": model_id, "messages": truncated_input}
            response = openai_client.chat.completions.create(**request_params)
            model_response["content"] = response.choices[0].message.content
        except Exception as e:
            model_response["error"] = (
                f"{models[model_alias]} model is not available. Error: {e}"
            )
        finally:
            response_event.set()  # Signal that the response is completed

    # Start the model request in a separate thread
    response_thread = threading.Thread(target=request_model_response)
    response_thread.start()

    # Wait for the specified timeout
    response_event_occurred = response_event.wait(timeout)

    if not response_event_occurred:
        raise TimeoutError(
            f"The {model_alias} model did not respond within {timeout} seconds."
        )
    elif model_response["error"]:
        raise Exception(model_response["error"])
    else:
        # Get the full conversation history for the model
        model_key = f"{model_alias}_chat"

        # Add the model's response to the conversation state
        conversation_state[model_key].append(
            {"role": "assistant", "content": model_response["content"]}
        )

        # Format the complete conversation history with different colors
        formatted_history = format_conversation_history(
            conversation_state[model_key][1:]
        )

        return formatted_history


def format_conversation_history(conversation_history):
    """
    Format the conversation history with different colors for user and model messages.

    Args:
        conversation_history (list): List of conversation messages with role and content.

    Returns:
        str: Markdown formatted conversation history.
    """
    formatted_text = ""

    for message in conversation_history:
        if message["role"] == "user":
            # Format user messages with blue text
            formatted_text += f"<div style='color: #0066cc; background-color: #f0f7ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>User:</strong> {message['content']}</div>\n\n"
        else:
            # Format assistant messages with dark green text
            formatted_text += f"<div style='color: #006633; background-color: #f0fff0; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>Model:</strong> {message['content']}</div>\n\n"

    return formatted_text


def save_content_to_hf(data, repo_name, file_name, token=None):
    """
    Save feedback content to Hugging Face repository.
    """
    # Serialize the content to JSON and encode it as bytes
    json_content = json.dumps(data, indent=4).encode("utf-8")

    # Create a binary file-like object
    file_like_object = io.BytesIO(json_content)

    # Define the path in the repository
    filename = f"{file_name}.json"

    # Ensure the user is authenticated with HF
    if token is None:
        token = HfApi().token
    if token is None:
        raise ValueError("Please log in to Hugging Face to submit votes.")

    # Upload to Hugging Face repository
    upload_file(
        path_or_fileobj=file_like_object,
        path_in_repo=filename,
        repo_id=repo_name,
        repo_type="dataset",
        token=token,
    )


def is_file_within_time_frame(file_path, days):
    try:
        # Extract timestamp from filename
        timestamp_str = file_path.split("/")[-1].split(".")[0]
        file_datetime = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        time_diff = datetime.now() - file_datetime
        return time_diff.days <= days
    except:
        return False


def load_content_from_hf(repo_name, file_name):
    """
    Read feedback content from a Hugging Face repository within the last LEADERBOARD_UPDATE_TIME_FRAME_DAYS days.

    Args:
        repo_name (str): Hugging Face repository name.
        file_name (str): Only load files under this prefix directory.

    Returns:
        list: Aggregated feedback data read from the repository.
    """
    data = []
    try:
        api = HfApi()
        # List all files in the repository, only under the file_name
        for file in api.list_repo_files(repo_id=repo_name, repo_type="dataset"):
            if not file.startswith(f"{file_name}/"):
                continue
            # Filter files by last LEADERBOARD_UPDATE_TIME_FRAME_DAYS days
            if not is_file_within_time_frame(file, LEADERBOARD_UPDATE_TIME_FRAME_DAYS):
                continue
            # Download and aggregate data
            local_path = hf_hub_download(
                repo_id=repo_name, filename=file, repo_type="dataset"
            )
            with open(local_path, "r") as f:
                entry = json.load(f)
                entry["timestamp"] = file.split("/")[-1].split(".")[0]
                data.append(entry)
        return data

    except:
        raise Exception("Error loading feedback data from Hugging Face repository.")


def get_leaderboard_data(vote_entry=None, use_cache=True):
    # Try to load cached leaderboard first
    if use_cache:
        try:
            cached_path = hf_hub_download(
                repo_id=LEADERBOARD_REPO,
                filename=f'{LEADERBOARD_FILE}.json',
                repo_type="dataset",
            )
            with open(cached_path, "r") as f:
                leaderboard_data = pd.read_json(f)
                # Round all numeric columns to two decimal places
                round_cols = {
                    "Elo Score": 2,
                    "Win Rate": 2,
                    "Conversation Efficiency Index": 2,
                    "Consistency Score": 2,
                    "Bradley-Terry Coefficient": 2,
                    "Eigenvector Centrality Value": 2,
                    "Newman Modularity Score": 2,
                    "PageRank Score": 2,
                }
                for col, decimals in round_cols.items():
                    if col in leaderboard_data.columns:
                        leaderboard_data[col] = pd.to_numeric(leaderboard_data[col], errors="coerce").round(decimals)
                return leaderboard_data
        except Exception as e:
            print(f"No cached leaderboard found, computing from votes...")

    # Load feedback data from the Hugging Face repository
    data = load_content_from_hf(VOTE_REPO, LEADERBOARD_FILE)
    vote_df = pd.DataFrame(data)

    # Concatenate the new feedback with the existing leaderboard data
    if vote_entry is not None:
        vote_df = pd.concat([vote_df, pd.DataFrame([vote_entry])], ignore_index=True)

    if vote_df.empty:
        return pd.DataFrame(
            columns=[
                "Rank",
                "Model",
                "Elo Score",
                "Win Rate",
                "Conversation Efficiency Index",
                "Consistency Score",
                "Bradley-Terry Coefficient",
                "Eigenvector Centrality Value",
                "Newman Modularity Score",
                "PageRank Score",
            ]
        )

    # Load conversation data from the Hugging Face repository
    conversation_data = load_content_from_hf(CONVERSATION_REPO, LEADERBOARD_FILE)
    conversation_df = pd.DataFrame(conversation_data)

    # Merge vote data with conversation data
    all_df = pd.merge(
        vote_df, conversation_df, on=["timestamp", "left", "right"], how="inner"
    )

    # Create dictionaries to track scores and match counts
    model_stats = {}

    # Process each row once and accumulate scores
    for _, row in all_df.iterrows():
        left_model = row["left"]
        right_model = row["right"]
        is_self_match = left_model == right_model

        # Initialize dictionaries for models if they don't exist yet
        for model in [left_model, right_model]:
            if model not in model_stats:
                model_stats[model] = {
                    "cei_sum": 0,  # Sum of per-round scores
                    "cei_max": 0,  # Sum of per-round maximums
                    "self_matches": 0,  # Count of self-matches
                    "self_draws": 0,  # Count of draws in self-matches
                }

        # Handle self-matches (same model on both sides)
        if is_self_match:
            model_stats[left_model]["self_matches"] += 1
            if row["winner"] == "both_bad" or row["winner"] == "tie":
                model_stats[left_model]["self_draws"] += 1
            continue

        # Determine scores based on winner for competitive matches
        match row["winner"]:
            case "left":
                left_score = 1
                right_score = -1
            case "right":
                left_score = -1
                right_score = 1
            case "tie":
                left_score = 0.3
                right_score = 0.3
            case "both_bad":
                left_score = -0.3
                right_score = -0.3

        # Count rounds for each side
        left_round = sum(1 for msg in row["left_chat"] if msg["role"] == "assistant")
        right_round = sum(1 for msg in row["right_chat"] if msg["role"] == "assistant")

        # Update CEI metrics
        model_stats[left_model]["cei_max"] += 1 / left_round
        model_stats[right_model]["cei_max"] += 1 / right_round
        model_stats[left_model]["cei_sum"] += left_score / left_round
        model_stats[right_model]["cei_sum"] += right_score / right_round

    # map vote to winner
    vote_df["winner"] = vote_df["winner"].map(
        {
            "left": evalica.Winner.X,
            "right": evalica.Winner.Y,
            "tie": evalica.Winner.Draw,
            "both_bad": evalica.Winner.Draw,
        }
    )

    # Calculate scores using various metrics
    avr_result = evalica.average_win_rate(
        vote_df["left"],
        vote_df["right"],
        vote_df["winner"],
        tie_weight=0,  # Chatbot Arena excludes ties
    )
    bt_result = evalica.bradley_terry(
        vote_df["left"], vote_df["right"], vote_df["winner"], tie_weight=0
    )
    newman_result = evalica.newman(
        vote_df["left"], vote_df["right"], vote_df["winner"], tie_weight=0
    )
    eigen_result = evalica.eigen(
        vote_df["left"], vote_df["right"], vote_df["winner"], tie_weight=0
    )
    elo_result = evalica.elo(
        vote_df["left"], vote_df["right"], vote_df["winner"], tie_weight=0
    )
    pagerank_result = evalica.pagerank(
        vote_df["left"], vote_df["right"], vote_df["winner"], tie_weight=0
    )

    # Clean up potential inf/NaN values in the results by extracting cleaned scores
    avr_scores = avr_result.scores.replace([float("inf"), float("-inf")], float("nan"))
    bt_scores = bt_result.scores.replace([float("inf"), float("-inf")], float("nan"))
    newman_scores = newman_result.scores.replace([float("inf"), float("-inf")], float("nan"))
    eigen_scores = eigen_result.scores.replace([float("inf"), float("-inf")], float("nan"))
    elo_scores = elo_result.scores.replace([float("inf"), float("-inf")], float("nan"))
    pagerank_scores = pagerank_result.scores.replace([float("inf"), float("-inf")], float("nan"))

    # Calculate CEI results
    cei_result = {}
    for model in elo_scores.index:
        if model in model_stats and model_stats[model]["cei_max"] > 0:
            cei_result[model] = round(
                model_stats[model]["cei_sum"] / model_stats[model]["cei_max"], 2
            )
        else:
            cei_result[model] = None
    cei_result = pd.Series(cei_result)

    # Calculate MCS results
    mcs_result = {}
    for model in elo_scores.index:
        if model in model_stats and model_stats[model]["self_matches"] > 0:
            mcs_result[model] = round(
                model_stats[model]["self_draws"] / model_stats[model]["self_matches"], 2
            )
        else:
            mcs_result[model] = None
    mcs_result = pd.Series(mcs_result)
    organization_values = [model_organization.get(model, "") for model in elo_scores.index]

    leaderboard_data = pd.DataFrame(
        {
            "Model": [name.split(": ", 1)[-1] for name in elo_scores.index],
            "Organization": organization_values,
            "Elo Score": elo_scores.values,
            "Win Rate": avr_scores.values,
            "Conversation Efficiency Index": cei_result.values,
            "Consistency Score": mcs_result.values,
            "Bradley-Terry Coefficient": bt_scores.values,
            "Eigenvector Centrality Value": eigen_scores.values,
            "Newman Modularity Score": newman_scores.values,
            "PageRank Score": pagerank_scores.values,
        }
    )

    # Round all numeric columns to two decimal places
    round_cols = {
        "Elo Score": 2,
        "Win Rate": 2,
        "Bradley-Terry Coefficient": 2,
        "Eigenvector Centrality Value": 2,
        "Newman Modularity Score": 2,
        "PageRank Score": 2,
    }
    for col, decimals in round_cols.items():
        if col in leaderboard_data.columns:
            leaderboard_data[col] = pd.to_numeric(leaderboard_data[col], errors="coerce").round(decimals)

    # Add a Rank column based on Elo scores
    leaderboard_data["Rank"] = (
        leaderboard_data["Elo Score"].rank(method="min", ascending=False).astype(int)
    )

    # Place rank in the first column
    leaderboard_data = leaderboard_data[
        ["Rank"] + [col for col in leaderboard_data.columns if col != "Rank"]
    ]

    # Save leaderboard data if this is a new vote
    if vote_entry is not None:
        try:
            # Convert DataFrame to JSON and save
            json_content = leaderboard_data.to_json(orient="records", indent=4).encode(
                "utf-8"
            )
            file_like_object = io.BytesIO(json_content)

            upload_file(
                path_or_fileobj=file_like_object,
                path_in_repo=f'{LEADERBOARD_FILE}.json',
                repo_id=LEADERBOARD_REPO,
                repo_type="dataset",
                token=HfApi().token,
            )
        except Exception as e:
            print(f"Failed to save leaderboard cache: {e}")

    return leaderboard_data


# Function to enable or disable submit buttons based on textbox content
def toggle_submit_button(text):
    if not text or text.strip() == "":
        return gr.update(interactive=False)  # Disable the button
    else:
        return gr.update(interactive=True)  # Enable the button



# Function to check initial authentication status
def check_auth_on_load(request: gr.Request):
    """Check if user is already authenticated when page loads."""
    # Try to get token from environment (for Spaces) or HfApi (for local)
    token = os.getenv("HF_TOKEN") or HfApi().token

    # Check if user is authenticated via OAuth
    is_authenticated = (hasattr(request, 'username') and request.username is not None and request.username != "")

    if is_authenticated or token:
        # User is logged in OR we have a token available
        return (
            gr.update(interactive=True),  # repo_url
            gr.update(interactive=True),  # shared_input
            gr.update(interactive=False),  # send_first (disabled until text entered)
            gr.update(interactive=True),  # feedback
            gr.update(interactive=True),  # submit_feedback_btn
            gr.update(visible=False),  # hint_markdown
            gr.update(visible=True),  # login_button (keep visible for logout)
            token,  # oauth_token
        )
    else:
        # User not logged in
        return (
            gr.update(interactive=False),  # repo_url
            gr.update(interactive=False),  # shared_input
            gr.update(interactive=False),  # send_first
            gr.update(interactive=False),  # feedback
            gr.update(interactive=False),  # submit_feedback_btn
            gr.update(visible=True),  # hint_markdown
            gr.update(visible=True),  # login_button
            None,  # oauth_token
        )


# Suppress the deprecation warning for theme parameter until Gradio 6.0 is released
warnings.filterwarnings('ignore', category=DeprecationWarning, message=".*'theme' parameter.*")
with gr.Blocks(title="SWE-Chatbot-Arena", theme=gr.themes.Soft()) as app:
    user_authenticated = gr.State(False)
    models_state = gr.State({})
    conversation_state = gr.State({})

    # Add OAuth information state to track user
    oauth_token = gr.State(None)

    with gr.Tab("🏆Leaderboard"):
        # Add title and description as a Markdown component
        gr.Markdown("# 🏆 LLM4SE Leaderboard")
        gr.Markdown(
            "Community-Driven Evaluation of Top Large Language Models (LLMs) in Software Engineering (SE) Tasks"
        )
        gr.Markdown(
            "*The SWE-Chatbot-Arena is an open-source platform designed to evaluate LLMs through human preference, "
            "fostering transparency and collaboration. This platform aims to empower the SE community to assess and compare the "
            "performance of leading LLMs in related tasks. For technical details, check out our [paper](https://arxiv.org/abs/2502.01860).*"
        )

        # Initialize the leaderboard with the DataFrame containing the expected columns
        leaderboard_component = Leaderboard(
            value=get_leaderboard_data(use_cache=True),
            select_columns=[
                "Rank",
                "Model",
                "Organization",
                "Elo Score",
                "Conversation Efficiency Index",
                "Consistency Score",
            ],
            search_columns=["Model"],
            filter_columns=[
                ColumnFilter(
                    "Elo Score",
                    min=800,
                    max=1600,
                    default=[800, 1600],
                    type="slider",
                    label="Elo Score"
                ),
                ColumnFilter(
                    "Win Rate",
                    min=0,
                    max=1,
                    default=[0, 1],
                    type="slider",
                    label="Win Rate"
                ),
                ColumnFilter(
                    "Conversation Efficiency Index",
                    min=0,
                    max=1,
                    default=[0, 1],
                    type="slider",
                    label="Conversation Efficiency Index"
                ),
                ColumnFilter(
                    "Consistency Score",
                    min=0,
                    max=1,
                    default=[0, 1],
                    type="slider",
                    label="Consistency Score"
                ),
            ],
            datatype=[
                "number",
                "str",
                "str",
                "number",
                "number",
                "number",
                "number",
                "number",
                "number",
                "number",
                "number",
            ],
        )

        # Add a divider
        gr.Markdown("---")

        # Add a citation block in Markdown
        gr.Markdown(
            """
            Made with ❤️ for SWE-Chatbot-Arena. If this work is useful to you, please consider citing our vision paper:
            ```
            @inproceedings{zhao2025se,
            title={SE Arena: An Interactive Platform for Evaluating Foundation Models in Software Engineering},
            author={Zhao, Zhimin},
            booktitle={2025 IEEE/ACM Second International Conference on AI Foundation Models and Software Engineering (Forge)},
            pages={78--81},
            year={2025},
            organization={IEEE}
            }
            ```
            """
        )
    with gr.Tab("⚔️Arena"):
        # Add title and description as a Markdown component
        gr.Markdown("# ⚔️ SWE-Chatbot-Arena")
        gr.Markdown("Explore and Test Top LLMs with SE Tasks by Community Voting")

        gr.Markdown("### 📜 How It Works")
        gr.Markdown(
            f"""
            - **Blind Comparison**: Submit a SE-related query to two anonymous LLMs randomly selected from up to {len(available_models)} top models from ChatGPT, Gemini, Grok, Claude, Qwen, Deepseek, Mistral, and others.
            - **Interactive Voting**: Engage in multi-turn dialogues with both LLMs and compare their responses. You can continue the conversation until you confidently choose the better model.
            - **Fair Play Rules**: Votes are counted only if LLM identities remain anonymous. Revealing a LLM's identity disqualifies the session.
            """
        )
        gr.Markdown(f"*Note: Due to budget constraints, responses that take longer than {TIMEOUT} seconds to generate will be discarded.*")

        # Add a divider
        gr.Markdown("---")

        # Add Hugging Face Sign In button and message
        with gr.Row():
            # Define the markdown text with or without the hint string
            markdown_text = "### Please sign in first to vote!"
            if SHOW_HINT_STRING:
                markdown_text += f"\n*{HINT_STRING}*"
            hint_markdown = gr.Markdown(markdown_text)
            with gr.Column():
                login_button = gr.LoginButton(
                    "Sign in with Hugging Face", elem_id="oauth-button"
                )

        guardrail_message = gr.Markdown("", visible=False, elem_id="guardrail-message")

        # NEW: Add a textbox for the repository URL above the user prompt
        repo_url = gr.Textbox(
            show_label=False,
            placeholder="Optional: Enter any GitHub, GitLab, or Hugging Face URL.",
            lines=1,
            interactive=False,
        )

        # Components with initial non-interactive state
        shared_input = gr.Textbox(
            show_label=False,
            placeholder="Enter your query for both models here.",
            lines=2,
            interactive=False,  # Initially non-interactive
        )
        send_first = gr.Button(
            "Submit", visible=True, interactive=False
        )  # Initially non-interactive

        # Add event listener to shared_input to toggle send_first button
        shared_input.change(
            fn=toggle_submit_button, inputs=shared_input, outputs=send_first
        )

        user_prompt_md = gr.Markdown(value="", visible=False)

        with gr.Column():
            shared_input
            user_prompt_md

        with gr.Row():
            response_a_title = gr.Markdown(value="", visible=False)
            response_b_title = gr.Markdown(value="", visible=False)

        with gr.Row():
            response_a = gr.Markdown(label="Response from Model A")
            response_b = gr.Markdown(label="Response from Model B")

        # Add a popup component for timeout notification
        with gr.Row(visible=False) as timeout_popup:
            timeout_message = gr.Markdown(
                "### Timeout\n\nOne of the models did not respond within 1 minute. Please try again."
            )
            close_popup_btn = gr.Button("Okay")

        def close_timeout_popup():
            # Re-enable or disable the submit buttons based on the current textbox content
            shared_input_state = gr.update(interactive=True)
            send_first_state = toggle_submit_button(shared_input.value)

            model_a_input_state = gr.update(interactive=True)
            model_a_send_state = toggle_submit_button(model_a_input.value)

            model_b_input_state = gr.update(interactive=True)
            model_b_send_state = toggle_submit_button(model_b_input.value)

            # Keep repo_url in sync with shared_input
            repo_url_state = gr.update(interactive=True)

            return (
                gr.update(visible=False),  # Hide the timeout popup
                shared_input_state,  # Update shared_input
                send_first_state,  # Update send_first button
                model_a_input_state,  # Update model_a_input
                model_a_send_state,  # Update model_a_send button
                model_b_input_state,  # Update model_b_input
                model_b_send_state,  # Update model_b_send button
                repo_url_state,  # Update repo_url button
            )

        # Multi-round inputs, initially hidden
        with gr.Row(visible=False) as multi_round_inputs:
            model_a_input = gr.Textbox(label="Model A Input", lines=1)
            model_a_send = gr.Button(
                "Send to Model A", interactive=False
            )  # Initially disabled

            model_b_input = gr.Textbox(label="Model B Input", lines=1)
            model_b_send = gr.Button(
                "Send to Model B", interactive=False
            )  # Initially disabled

        # Add event listeners to model_a_input and model_b_input to toggle their submit buttons
        model_a_input.change(
            fn=toggle_submit_button, inputs=model_a_input, outputs=model_a_send
        )

        model_b_input.change(
            fn=toggle_submit_button, inputs=model_b_input, outputs=model_b_send
        )

        close_popup_btn.click(
            close_timeout_popup,
            inputs=[],
            outputs=[
                timeout_popup,
                shared_input,
                send_first,
                model_a_input,
                model_a_send,
                model_b_input,
                model_b_send,
                repo_url,
            ],
        )

        def guardrail_check_se_relevance(user_input):
            """
            Use openai/gpt-oss-safeguard-20b to check if the user input is SE-related.
            Return True if it is SE-related, otherwise False.
            """
            # Example instructions for classification — adjust to your needs
            system_message = {
                "role": "system",
                "content": (
                    "You are a classifier that decides if a user's question is relevant to software engineering. "
                    "If the question is about software engineering concepts, tools, processes, or code, respond with 'Yes'. "
                    "Otherwise, respond with 'No'."
                ),
            }
            user_message = {"role": "user", "content": user_input}

            try:
                # Make the chat completion call
                response = openai_client.chat.completions.create(
                    model="openai/gpt-oss-safeguard-20b", messages=[system_message, user_message]
                )
                classification = response.choices[0].message.content.strip().lower()
                # Check if the LLM responded with 'Yes'
                return classification.lower().startswith("yes")
            except Exception as e:
                print(f"Guardrail check failed: {e}")
                # If there's an error, you might decide to fail open (allow) or fail closed (block).
                # Here we default to fail open, but you can change as needed.
                return True

        def disable_first_submit_ui():
            """First function to immediately disable UI elements"""
            return (
                # [0] guardrail_message: hide
                gr.update(visible=False),
                # [1] shared_input: disable but keep visible
                gr.update(interactive=False),
                # [2] repo_url: disable but keep visible
                gr.update(interactive=False),
                # [3] send_first: disable and show loading state
                gr.update(interactive=False, value="Processing..."),
            )

        # Function to update model titles and responses
        def update_model_titles_and_responses(
            repo_url, user_input, models_state, conversation_state
        ):
            # Guardrail check first
            if not repo_url and not guardrail_check_se_relevance(user_input):
                # Return updates to show the guardrail message and re-enable UI
                return (
                    # [0] guardrail_message: Show guardrail message
                    gr.update(
                        value="### Oops! Try asking something about software engineering. Thanks!",
                        visible=True,
                    ),
                    # [1] shared_input: clear and re-enable
                    gr.update(value="", interactive=True, visible=True),
                    # [2] repo_url: clear and re-enable
                    gr.update(value="", interactive=True, visible=True),
                    # [3] user_prompt_md: clear and hide
                    gr.update(value="", visible=False),
                    # [4] response_a_title: clear and hide
                    gr.update(value="", visible=False),
                    # [5] response_b_title: clear and hide
                    gr.update(value="", visible=False),
                    # [6] response_a: clear response
                    gr.update(value=""),
                    # [7] response_b: clear response
                    gr.update(value=""),
                    # [8] multi_round_inputs: hide
                    gr.update(visible=False),
                    # [9] vote_panel: hide
                    gr.update(visible=False),
                    # [10] send_first: re-enable button with original text
                    gr.update(visible=True, interactive=True, value="Submit"),
                    # [11] feedback: enable the selection
                    gr.update(interactive=True),
                    # [12] models_state: pass state as-is
                    models_state,
                    # [13] conversation_state: pass state as-is
                    conversation_state,
                    # [14] timeout_popup: hide
                    gr.update(visible=False),
                    # [15] model_a_send: disable
                    gr.update(interactive=False),
                    # [16] model_b_send: disable
                    gr.update(interactive=False),
                    # [17] thanks_message: hide
                    gr.update(visible=False),
                )

            # Fetch repository info if a URL is provided
            repo_info = fetch_url_content(repo_url)
            combined_user_input = (
                f"Context: {repo_info}\n\nInquiry: {user_input}"
                if repo_info
                else user_input
            )

            # Randomly select two models for the comparison
            selected_model = random.choice(available_models)
            models = {"left": selected_model, "right": selected_model}

            # Create a copy to avoid modifying the original
            conversations = models.copy()
            conversations.update(
                {
                    "url": repo_url,
                    "left_chat": [{"role": "user", "content": combined_user_input}],
                    "right_chat": [{"role": "user", "content": combined_user_input}],
                }
            )

            # Clear previous states
            models_state.clear()
            conversation_state.clear()

            # Update the states
            models_state.update(models)
            conversation_state.update(conversations)

            try:
                response_a = chat_with_models("left", models_state, conversation_state)
                response_b = chat_with_models("right", models_state, conversation_state)
            except TimeoutError as e:
                # Handle timeout by resetting components and showing a popup.
                return (
                    # [0] guardrail_message: hide
                    gr.update(visible=False),
                    # [1] shared_input: re-enable, preserve user input
                    gr.update(interactive=True, visible=True),
                    # [2] repo_url: re-enable, preserve user input
                    gr.update(interactive=True, visible=True),
                    # [3] user_prompt_md: hide
                    gr.update(value="", visible=False),
                    # [4] response_a_title: hide
                    gr.update(value="", visible=False),
                    # [5] response_b_title: hide
                    gr.update(value="", visible=False),
                    # [6] response_a: clear
                    gr.update(value=""),
                    # [7] response_b: clear
                    gr.update(value=""),
                    # [8] multi_round_inputs: hide
                    gr.update(visible=False),
                    # [9] vote_panel: hide
                    gr.update(visible=False),
                    # [10] send_first: re-enable with original text
                    gr.update(visible=True, interactive=True, value="Submit"),
                    # [11] feedback: disable
                    gr.update(interactive=False),
                    # [12] models_state: pass state as-is
                    models_state,
                    # [13] conversation_state: pass state as-is
                    conversation_state,
                    # [14] timeout_popup: show popup
                    gr.update(visible=True),
                    # [15] model_a_send: disable
                    gr.update(interactive=False),
                    # [16] model_b_send: disable
                    gr.update(interactive=False),
                    # [17] thanks_message: hide
                    gr.update(visible=False),
                )
            except Exception as e:
                # Handle other errors by resetting UI state and showing error message
                return (
                    # [0] guardrail_message: show error message
                    gr.update(value=f"### Error: {str(e)}", visible=True),
                    # [1] shared_input: re-enable, preserve user input
                    gr.update(interactive=True, visible=True),
                    # [2] repo_url: re-enable, preserve user input
                    gr.update(interactive=True, visible=True),
                    # [3] user_prompt_md: hide
                    gr.update(value="", visible=False),
                    # [4] response_a_title: hide
                    gr.update(value="", visible=False),
                    # [5] response_b_title: hide
                    gr.update(value="", visible=False),
                    # [6] response_a: clear
                    gr.update(value=""),
                    # [7] response_b: clear
                    gr.update(value=""),
                    # [8] multi_round_inputs: hide
                    gr.update(visible=False),
                    # [9] vote_panel: hide
                    gr.update(visible=False),
                    # [10] send_first: re-enable with original text
                    gr.update(visible=True, interactive=True, value="Submit"),
                    # [11] feedback: disable
                    gr.update(interactive=False),
                    # [12] models_state: pass state as-is
                    models_state,
                    # [13] conversation_state: pass state as-is
                    conversation_state,
                    # [14] timeout_popup: hide popup
                    gr.update(visible=False),
                    # [15] model_a_send: disable
                    gr.update(interactive=False),
                    # [16] model_b_send: disable
                    gr.update(interactive=False),
                    # [17] thanks_message: hide
                    gr.update(visible=False),
                )

            # Determine the initial state of the multi-round send buttons
            model_a_send_state = toggle_submit_button("")
            model_b_send_state = toggle_submit_button("")
            display_content = f"### Your Query:\n\n{user_input}"
            if repo_info:
                display_content += f"\n\n### Repo-related URL:\n\n{repo_url}"

            # Return the updates for all 18 outputs.
            return (
                # [0] guardrail_message: hide (since no guardrail issue)
                gr.update(visible=False),
                # [1] shared_input: re-enable but hide
                gr.update(interactive=True, visible=False),
                # [2] repo_url: re-enable but hide
                gr.update(interactive=True, visible=False),
                # [3] user_prompt_md: display the user's query
                gr.update(value=display_content, visible=True),
                # [4] response_a_title: show anonymized title for Model A
                gr.update(value="### Model A", visible=True),
                # [5] response_b_title: show anonymized title for Model B
                gr.update(value="### Model B", visible=True),
                # [6] response_a: display Model A response
                gr.update(value=response_a),
                # [7] response_b: display Model B response
                gr.update(value=response_b),
                # [8] multi_round_inputs: show the input section for multi-round dialogues
                gr.update(visible=True),
                # [9] vote_panel: show vote panel
                gr.update(visible=True),
                # [10] send_first: hide the submit button but restore label
                gr.update(visible=False, value="Submit"),
                # [11] feedback: enable the feedback selection
                gr.update(interactive=True),
                # [12] models_state: pass updated models_state
                models_state,
                # [13] conversation_state: pass updated conversation_state
                conversation_state,
                # [14] timeout_popup: hide any timeout popup if visible
                gr.update(visible=False),
                # [15] model_a_send: set state of the model A send button
                model_a_send_state,
                # [16] model_b_send: set state of the model B send button
                model_b_send_state,
                # [17] thanks_message: hide the thank-you message
                gr.update(visible=False),
            )

        # Feedback panel, initially hidden
        with gr.Column(visible=False) as vote_panel:
            gr.Markdown("### Which model do you prefer?")
            with gr.Row():
                feedback = gr.Radio(
                    choices=["Model A", "Model B", "Tie", "Tie (Both Bad)"],
                    show_label=False,
                    value="Tie",
                    interactive=False,
                )
                submit_feedback_btn = gr.Button("Submit Feedback", interactive=False)

        thanks_message = gr.Markdown(
            value="## Thanks for your vote!", visible=False
        )  # Add thank you message

        def hide_thanks_message():
            return gr.update(visible=False)

        # Function to handle login - uses gr.Request to get OAuth info
        def handle_login(request: gr.Request):
            """
            Handle user login using Hugging Face OAuth.
            When deployed on HF Spaces with OAuth, request contains user info.
            """
            # Try to get token from environment (for Spaces) or HfApi (for local)
            token = os.getenv("HF_TOKEN") or HfApi().token

            # Check if user is authenticated through HF Spaces OAuth
            is_authenticated = hasattr(request, 'username') and request.username

            if is_authenticated or token:
                # User is logged in
                return (
                    gr.update(interactive=True),  # repo_url -> Enable
                    gr.update(interactive=True),  # Enable shared_input
                    gr.update(interactive=False),  # Keep send_first disabled initially
                    gr.update(interactive=True),  # Enable feedback radio buttons
                    gr.update(interactive=True),  # Enable submit_feedback_btn
                    gr.update(visible=False),  # Hide the hint string
                    gr.update(visible=True),  # Keep login button visible for logout
                    token,  # Store the oauth token
                )
            else:
                # User is not logged in - instruct them to use HF login
                return (
                    gr.update(interactive=False),  # repo_url -> disable
                    gr.update(interactive=False),  # Keep shared_input disabled
                    gr.update(interactive=False),  # Keep send_first disabled
                    gr.update(interactive=False),  # Keep feedback radio buttons disabled
                    gr.update(interactive=False),  # Keep submit_feedback_btn disabled
                    gr.update(visible=True, value="## Please sign in with Hugging Face!\nClick the 'Sign in with Hugging Face' button above to authenticate."),  # Show instructions
                    gr.update(visible=True),  # Keep login button visible
                    None,  # Clear oauth_token
                )

        # First round handling
        send_first.click(
            fn=hide_thanks_message, inputs=[], outputs=[thanks_message]
        ).then(
            fn=disable_first_submit_ui,  # First disable UI
            inputs=[],
            outputs=[
                guardrail_message,
                shared_input,
                repo_url,
                send_first,  # Just the essential UI elements to update immediately
            ],
        ).then(
            fn=update_model_titles_and_responses,  # Then do the actual processing
            inputs=[repo_url, shared_input, models_state, conversation_state],
            outputs=[
                guardrail_message,
                shared_input,
                repo_url,
                user_prompt_md,
                response_a_title,
                response_b_title,
                response_a,
                response_b,
                multi_round_inputs,
                vote_panel,
                send_first,
                feedback,
                models_state,
                conversation_state,
                timeout_popup,
                model_a_send,
                model_b_send,
                thanks_message,
            ],
        )

        def disable_model_a_ui():
            """First function to immediately disable model A UI elements"""
            return (
                # [0] model_a_input: disable
                gr.update(interactive=False),
                # [1] model_a_send: disable and show loading state
                gr.update(interactive=False, value="Processing..."),
            )

        # Handle subsequent rounds
        def handle_model_a_send(user_input, models_state, conversation_state):
            try:
                conversation_state["left_chat"].append(
                    {"role": "user", "content": user_input}
                )
                response = chat_with_models("left", models_state, conversation_state)
                # Clear the input box and disable the send button
                return (
                    response,
                    conversation_state,
                    gr.update(visible=False),
                    gr.update(
                        value="", interactive=True
                    ),  # Clear and enable model_a_input
                    gr.update(
                        interactive=False, value="Send to Model A"
                    ),  # Reset button text
                )
            except TimeoutError as e:
                # Disable inputs when timeout occurs
                return (
                    gr.update(value=""),  # Clear response
                    conversation_state,
                    gr.update(visible=True),  # Show the timeout popup
                    gr.update(interactive=True),  # Re-enable model_a_input
                    gr.update(
                        interactive=True, value="Send to Model A"
                    ),  # Re-enable model_a_send button
                )
            except Exception as e:
                raise gr.Error(str(e))

        def disable_model_b_ui():
            """First function to immediately disable model B UI elements"""
            return (
                # [0] model_b_input: disable
                gr.update(interactive=False),
                # [1] model_b_send: disable and show loading state
                gr.update(interactive=False, value="Processing..."),
            )

        def handle_model_b_send(user_input, models_state, conversation_state):
            try:
                conversation_state["right_chat"].append(
                    {"role": "user", "content": user_input}
                )
                response = chat_with_models("right", models_state, conversation_state)
                # Clear the input box and disable the send button
                return (
                    response,
                    conversation_state,
                    gr.update(visible=False),
                    gr.update(
                        value="", interactive=True
                    ),  # Clear and enable model_b_input
                    gr.update(
                        interactive=False, value="Send to Model B"
                    ),  # Reset button text
                )
            except TimeoutError as e:
                # Disable inputs when timeout occurs
                return (
                    gr.update(value=""),  # Clear response
                    conversation_state,
                    gr.update(visible=True),  # Show the timeout popup
                    gr.update(interactive=True),  # Re-enable model_b_input
                    gr.update(
                        interactive=True, value="Send to Model B"
                    ),  # Re-enable model_b_send button
                )
            except Exception as e:
                raise gr.Error(str(e))

        model_a_send.click(
            fn=disable_model_a_ui,  # First disable UI
            inputs=[],
            outputs=[model_a_input, model_a_send],
        ).then(
            fn=handle_model_a_send,  # Then do the actual processing
            inputs=[model_a_input, models_state, conversation_state],
            outputs=[
                response_a,
                conversation_state,
                timeout_popup,
                model_a_input,
                model_a_send,
            ],
        )
        model_b_send.click(
            fn=disable_model_b_ui,  # First disable UI
            inputs=[],
            outputs=[model_b_input, model_b_send],
        ).then(
            fn=handle_model_b_send,  # Then do the actual processing
            inputs=[model_b_input, models_state, conversation_state],
            outputs=[
                response_b,
                conversation_state,
                timeout_popup,
                model_b_input,
                model_b_send,
            ],
        )

        def reveal_models_and_thank(models_state):
            """Immediately reveal model identities and show thanks message."""
            left_model = models_state.get("left", "Unknown")
            right_model = models_state.get("right", "Unknown")
            left_display = left_model.split(": ", 1)[-1] if ": " in left_model else left_model
            right_display = right_model.split(": ", 1)[-1] if ": " in right_model else right_model

            return (
                gr.update(value=f"### Model A: {left_display}", visible=True),
                gr.update(value=f"### Model B: {right_display}", visible=True),
                gr.update(
                    visible=True,
                    value=(
                        f"## Thanks for your vote! Identities revealed above.\n"
                        f"**Model A:** {left_display}  \n"
                        f"**Model B:** {right_display}"
                    ),
                ),
                gr.update(interactive=False),  # submit_feedback_btn
                gr.update(interactive=False),  # feedback
            )

        def submit_feedback(vote, models_state, conversation_state, token):
            # Map vote to actual model names
            match vote:
                case "Model A":
                    winner_model = "left"
                case "Model B":
                    winner_model = "right"
                case "Tie":
                    winner_model = "tie"
                case _:
                    winner_model = "both_bad"

            # Capture model display names before state is cleared
            left_model = models_state.get("left", "Unknown")
            right_model = models_state.get("right", "Unknown")
            left_display = left_model.split(": ", 1)[-1] if ": " in left_model else left_model
            right_display = right_model.split(": ", 1)[-1] if ": " in right_model else right_model

            # Create feedback entry
            vote_entry = {
                "left": models_state["left"],
                "right": models_state["right"],
                "winner": winner_model,
            }

            # Get the current datetime for file naming
            file_name = f"{LEADERBOARD_FILE}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Save feedback back to the Hugging Face dataset
            save_content_to_hf(vote_entry, VOTE_REPO, file_name, token)

            conversation_state["right_chat"][0]["content"] = conversation_state[
                "right_chat"
            ][0]["content"].split("\n\nInquiry: ")[-1]
            conversation_state["left_chat"][0]["content"] = conversation_state[
                "left_chat"
            ][0]["content"].split("\n\nInquiry: ")[-1]

            # Save conversations back to the Hugging Face dataset
            save_content_to_hf(conversation_state, CONVERSATION_REPO, file_name, token)

            # Clear state
            models_state.clear()
            conversation_state.clear()

            # Adjust output count to match the interface definition
            return (
                gr.update(
                    value="", interactive=True, visible=True
                ),  # [0] Clear shared_input textbox
                gr.update(
                    value="", interactive=True, visible=True
                ),  # [1] Clear repo_url textbox
                gr.update(
                    value="", visible=False
                ),  # [2] Hide user_prompt_md markdown component
                gr.update(
                    value="", visible=False
                ),  # [3] Hide response_a_title markdown component
                gr.update(
                    value="", visible=False
                ),  # [4] Hide response_b_title markdown component
                gr.update(value=""),  # [5] Clear Model A response markdown component
                gr.update(value=""),  # [6] Clear Model B response markdown component
                gr.update(visible=False),  # [7] Hide multi_round_inputs row
                gr.update(visible=False),  # [8] Hide vote_panel row
                gr.update(
                    value="Submit", interactive=True, visible=True
                ),  # [9] Reset send_first button
                gr.update(
                    value="Tie", interactive=True
                ),  # [10] Reset feedback radio selection
                get_leaderboard_data(vote_entry, use_cache=False),  # [11] Updated leaderboard data
                gr.update(
                    visible=True,
                    value=(
                        f"## Thanks for your vote!\n"
                        f"**Model A:** {left_display}  \n"
                        f"**Model B:** {right_display}"
                    ),
                ),  # [12] Show the thanks_message with model identities
                gr.update(interactive=True),  # [13] Re-enable submit_feedback_btn
            )

        # Update the click event for the submit feedback button
        # Step 1: Instantly reveal model identities and show thanks
        # Step 2: Upload vote data and reset UI for next round
        submit_feedback_btn.click(
            fn=reveal_models_and_thank,
            inputs=[models_state],
            outputs=[
                response_a_title,  # Reveal Model A identity
                response_b_title,  # Reveal Model B identity
                thanks_message,  # Show thanks message with identities
                submit_feedback_btn,  # Disable to prevent double-submit
                feedback,  # Disable feedback selection
            ],
        ).then(
            fn=submit_feedback,
            inputs=[feedback, models_state, conversation_state, oauth_token],
            outputs=[
                shared_input,  # Reset shared_input
                repo_url,  # Reset repo_url
                user_prompt_md,  # Hide user_prompt_md
                response_a_title,  # Hide Model A title
                response_b_title,  # Hide Model B title
                response_a,  # Clear Model A response
                response_b,  # Clear Model B response
                multi_round_inputs,  # Hide multi-round input section
                vote_panel,  # Hide vote panel
                send_first,  # Reset and update send_first button
                feedback,  # Reset feedback selection
                leaderboard_component,  # Update leaderboard data dynamically
                thanks_message,  # Show thanks with model identities
                submit_feedback_btn,  # Re-enable submit feedback button
            ],
        )

        # Add a divider
        gr.Markdown("---")

        # Add Terms of Service at the bottom
        gr.Markdown("### Terms of Service")
        gr.Markdown(
            """
            *Users are required to agree to the following terms before using the service:*

            - The service is a **research preview**. It only provides limited safety measures and may generate offensive content.
            - It must not be used for any **illegal, harmful, violent, racist, or sexual** purposes.
            - Please do not upload any **private** information.
            - The service collects user dialogue data, including both text and images, and reserves the right to distribute it under a **Creative Commons Attribution (CC-BY)** or a similar license.
            """
        )

    # Check authentication status when the app loads
    app.load(
        check_auth_on_load,
        outputs=[
            repo_url,
            shared_input,
            send_first,
            feedback,
            submit_feedback_btn,
            hint_markdown,
            login_button,
            oauth_token,
        ],
    )

    app.launch()
