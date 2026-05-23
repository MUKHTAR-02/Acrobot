# ACROBOT - An AI Speech-to-Speech ChatBot Assistant

## Follow these steps to run 

### Step 1: Create a venv
python -m venv acrobot-env
### Step 2: Activate it
acrobot-env\Scripts\activate     # for Windows
source acrobot-env/bin/activate  # for Linux
### Step 2.1 : bypass the execution policy for windows
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
### Step 3: Upgrade pip
pip install --upgrade pip
### Step 4.1: Install all dependencies
python -m pip install groq edge-tts pygame sounddevice soundfile numpy
### Step 4.2: Install .env
python -m pip install python-dotenv
### Step 5: Set your Groq API key
GROQ_API_KEY=your_key_here
### Step 6: Run the chatbot
python acrobot.py
