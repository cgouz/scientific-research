# creating env

## creating venv
```sh
python3 -m venv .venv
```

## activation
```sh
source .venv/bin/activate
```

## upgrading pip
```sh
python -m pip install --upgrade pip
```

## installing hf cli
```sh
pip install -U huggingface_hub
```

## checking hf cli
```sh
hf --help
```

## creating token on hf platform
```txt
Settings
→ Access Tokens
→ Create new token
```
ex: 
```token
hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## hf auth:
```sh
hf auth login
```
![alt text](image.png)

## check hf auth
```sh
hf auth whoami
```
![alt text](image-1.png)

## test: getting info about dataset
```sh
hf repo-files DavronSherbaev/uzbekvoice-filtered --repo-type dataset
```

# hf cli


## view dataset files
```sh
hf datasets ls DavronSherbaev/uzbekvoice-filtered
```

## view dataset recursive
```sh
hf datasets ls DavronSherbaev/uzbekvoice-filtered -R
```

## view with volume
```sh
hf datasets ls DavronSherbaev/uzbekvoice-filtered -R -h
```

# hf cli anatomy

## main str
```txt
hf <resource> <action> [arguments] [options]
```

ex: 
```sh
hf datasets info DavronSherbaev/uzbekvoice-filtered
```

in here:
```
hf
│
├── datasets        ← resource
│
├── info            ← action
│
└── Davron...       ← argument
```

## 1. The most top layer — hf

### in terminal
```sh
hf --help
```

### main rec
```txt
hf
│
├── auth
├── datasets
├── models
├── repos
├── spaces
├── download
├── upload
├── cache
├── collections
├── discussions
├── jobs
├── endpoints
├── buckets
├── cp
├── sync
├── env
├── version
├── update
└── ...
```

#### hf auth

```txt
hf
└── auth
    ├── login
    ├── logout
    ├── whoami
    ├── list
    ├── switch
    └── token
```

#### hf datasets

```txt
hf
└── datasets
    ├── list / ls
    ├── info
    ├── card
    ├── parquet
    ├── sql
    └── leaderboard
```

#### hf download 
hf download is one of the most important Hugging Face CLI commands. Below is the full anatomy

```txt
hf download [OPTIONS] REPO_ID [FILENAMES]...
```

```txt
hf download
│  │
│  └── action
└───── Hugging Face CLI

REPO_ID        → qayerdan download qilamiz
FILENAMES      → nimalarni download qilamiz
OPTIONS        → qanday download qilamiz
```

##### the simplest cmd
```sh
hf download ...
```

ex:
```sh
hf download openai-community/gpt2
```

##### General anatomy
Full structure:
```txt
hf download \
    REPO_ID \
    [FILENAMES...] \
    --repo-type TYPE \
    --revision REVISION \
    --include PATTERN \
    --exclude PATTERN \
    --cache-dir PATH \
    --local-dir PATH \
    --force-download \
    --dry-run \
    --token TOKEN \
    --max-workers NUMBER
```

###### REPO_ID
This is required.
Format: ```owner/repository``` ex: ```DavronSherbaev/uzbekvoice-filtered```

in here:
```
DavronSherbaev
      │
      └── owner / username

uzbekvoice-filtered
      │
      └── repository name
```
###### FILENAMES 
After REPO_ID, you can specify one or more files. 

Suppose repository contains:
```txt
README.md
config.json
data/
├── train-00000.parquet
├── train-00001.parquet
└── validation.parquet
```

If you only want: `README.md`

use:
```sh
hf download DavronSherbaev/uzbekvoice-filtered \
    README.md \
    --repo-type dataset
```

For a file inside a folder:

```sh
hf download DavronSherbaev/uzbekvoice-filtered \
    data/train-00000.parquet \
    --repo-type dataset
```

Multiple files:
```sh
hf download DavronSherbaev/uzbekvoice-filtered \
    README.md \
    data/train-00000.parquet \
    data/validation.parquet \
    --repo-type dataset
```

exs:
```sh
hf download \
  ArabicNLPWorld/arabic-russian-translation-corpus \
  --repo-type dataset \
  --local-dir /workspace/datasets/arabic-russian-translation-corpus
```

# works:
```bash
mkdir -p ./datasets/RUCAIBox-Translation

hf download RUCAIBox/Translation \
  wmt16-de-en.tgz \
  --repo-type dataset \
  --local-dir ./datasets/RUCAIBox-Translation \
  --force-download
```