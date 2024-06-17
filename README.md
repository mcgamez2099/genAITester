# genAITester
Generative AI Tester App - Chat, RAG, Web-Search and more

# tested and created with Python 3.11.4 

# Install on macOS (tested with Sonoma 14.5) / Linux (tested with Ubuntu 22.04)
git clone https://github.com/mcgamez2099/genAITester.git<br>
cd genAITester
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export NVIDIA_API_KEY=nvapi-APIKEY
chainlit run genAITester.py

# Install on Windows (tested with Windows 11)
git clone https://github.com/mcgamez2099/genAITester.git
cd genAITester
python -m venv .venv
source .venv/Scripts/activate
export NVIDIA_API_KEY=nvapi-APIKEY
chainlit run genAITester.py

# get NVIDIA API KEY
visit: https://build.nvidia.com/explore/discover
