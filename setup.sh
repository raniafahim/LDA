cd ~/work/TopicModeling/ 
uv venv .venv
source .venv/bin/activate
uv pip install notebook ipykernel
uv pip install -r requirements.txt
python -m ipykernel install --user --name=my-uv-env --display-name "Python (uv)"