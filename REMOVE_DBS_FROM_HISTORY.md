# Git 히스토리에서 DBs/ 제거 (대용량 파일 푸시 오류 해결)

GitHub는 100MB 초과 파일을 거부합니다. `DBs/`가 **과거 커밋**에 포함되어 있으면 푸시가 계속 실패합니다.  
아래 방법 중 하나로 **모든 커밋에서 DBs/를 제거**한 뒤 푸시하세요.

---

## 방법 1: filter-branch로 히스토리에서 DBs 제거 (권장)

**Git Bash**를 열고 프로젝트 폴더에서 실행:

```bash
cd /c/chat_bot_multiagent_rag

# 모든 커밋에서 DBs/ 제거
git filter-branch --force --index-filter "git rm -rf --cached --ignore-unmatch DBs/" --prune-empty HEAD
```

이후 푸시:

```bash
git push --force-with-lease origin main
```

- 로컬의 `DBs/` 폴더와 파일은 **삭제되지 않습니다** (챗봇은 그대로 동작).
- 히스토리가 바뀌므로 `--force-with-lease`로 푸시해야 합니다.

---

## 방법 2: 커밋이 거의 없을 때 — 한 번에 새 커밋으로 만들기

커밋이 1~2개뿐이고 히스토리를 유지할 필요가 없다면:

**PowerShell**에서:

```powershell
cd c:\chat_bot_multiagent_rag

# 1) DBs를 추적에서 제거
git rm -r --cached DBs/

# 2) .gitignore 반영
git add .
git status   # DBs/가 목록에 없어야 함

# 3) 모든 변경을 하나의 새 루트 커밋으로 (기존 main 히스토리 대체)
git checkout --orphan new_main
git add .
git commit -m "Initial commit (RAG data in DBs/ excluded for GitHub size limit)"

# 4) 기존 main 제거 후 new_main을 main으로
git branch -D main
git branch -m main

# 5) 푸시
git push --force origin main
```

- **주의**: 기존 `main`의 커밋 히스토리는 사라지고, 위 한 개 커밋만 남습니다.

---

## 방법 1 실행 후 백업 제거 (선택)

filter-branch는 `refs/original/`에 백업을 둡니다. 정리하려면:

```bash
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d
```

이후 필요하면 `git reflog expire --expire=now --all && git gc --prune=now` 로 정리할 수 있습니다.
