# Install the Mac-native version instead of the generic one:
- pip install tensorflow-macos

# Optionally for GPU support:
- pip install tensorflow-metal

#  Huggingface Login
- huggingface-cli login
- Your_Personal_Access_Token
- Request access to the model via huggingface

# Add huggingface access
- HUGGINGFACE_HUB_TOKEN


# RUN THE APPLICATION
# Run Frontend
- cd chatbot-frontend
- npm run dev

# Run backend
- python backend/app.py

# THROUBLESHOT
- No module named ChatbotTextToText
  - export PYTHONPATH=$(pwd)