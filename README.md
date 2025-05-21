# Quran Agent

# Project Structure
```text
quran_agent/                  # Root project directory
├─ src/                       # Source code directory
│  └─ quran_agent/            # Package for quran_agent
│       ├─ __init__.py
│       ├─ config.py           
│       ├─ create_vdb.py          
│       ├─ quran_client.py    
│       ├─ run_pipe.py
│       └─ utils/
│           ├─ __init__.py
│           └─ auth_utils.py                   
├─ setup.py                   # Package installation and entry points
├─ README.md                  # Overview and usage
└─ .gitignore                 # Git ignore rules
```

# Setup
#### 1. Clone this repository.
#### 2. Setup virtual environment.
#### 3. Install packages in editable mode.
```
conda create -n python=3.10 qagent
conda activate qagent
pip install -e .
```

# Environment Setup
To configure the environment, create a `.env` file in the root directory of the project and set the following variables with their respective values:

- `OPENAI_API_KEY`
- `PREPROD_CLIENT_ID`
- `PREPROD_CLIENT_SECRET`
- `PROD_CLIENT_ID`
- `PROD_CLIENT_SECRET`

These variables are essential for the application to authenticate and function correctly. Ensure the `.env` file is not shared or committed to version control to keep your credentials secure.

# Inference
At the root of this repo, run:
````
python -m quran_agent.create_vdb         #-- for database fetch from api and vector databse creation
python -m quran_agent.run_pipe           #-- for asking the agent
````