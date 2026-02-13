```bash
git clone https://github.com/yprakash/sentinel-audit.git
cd sentinel-audit
# Install uv - the fastest python manager
# curl -LsSf https://astral.sh/uv/install.sh | sh
brew install uv
uv init --python 3.12.12
uv add groq python-dotenv "mcp[cli]"
vi .env
uv run scripts/analyzer.py
```
