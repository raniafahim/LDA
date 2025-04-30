cd ~/work/TopicModeling/ 
uv venv .venv
source .venv/bin/activate
uv pip install notebook ipykernel
python -m ipykernel install --user --name=my-uv-env --display-name "Python (uv-bt)"
