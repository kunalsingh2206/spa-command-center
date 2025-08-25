# SPA Command Center (Streamlit)

This repo contains a ready-to-deploy Streamlit app skeleton for your SPA Command Center.

What's included:
- spa_command_center.py — a minimal working app (file uploaders + CRM majority join)
- /data/Field variables.xlsx and /data/Process Flow.xlsx — packaged references
- requirements.txt — pinned versions
- .streamlit/config.toml — optional theme/server config
- .gitignore — ignores local uploads (report*.xlsx, Booking Name & CRM Executive*.xlsx)

How to run locally:
1) pip install -r requirements.txt
2) streamlit run spa_command_center.py
3) In the app UI, upload Salesforce export (report*.xlsx) and the CRM mapping file (2 columns)

Deploy on Streamlit Community Cloud:
1) Push this folder to a new GitHub repo (e.g., spa-command-center)
2) Go to share.streamlit.io and sign in with GitHub
3) New app → select your repo, branch=main, file=spa_command_center.py
4) Deploy
