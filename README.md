# SMU Smart City GenAI Hackathon

## Starting Ollama
### 1. Installing ollama
- You can download the application directly from [ollama](https://ollama.com/)
- If you are a mac user, you can install it using homebrew as well 
```
brew install ollama
```
### 2. Start the Ollama Server
- Run the following command on the CLI
```
ollama serve 
```
### 3. Download the models
- List of models available can be found [here](https://ollama.com/library)
- Example
```
ollama pull granite3.1-dense:2b
ollama pull granite3.1-dense:8b
```
- *Be sure to check the hardware requirements before downloading*
### 4. Running the model locally
- Run the following command on the CLI
```
ollama run {$MODEL_ID}
```
- Replace $MODEL_ID with the model you selected (example: granite3.1-dense:8b)

## Running Streamlit UI
- Create virtual environment and install requirements
```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
- Start application using below command
```
streamlit run src/app.py
```
