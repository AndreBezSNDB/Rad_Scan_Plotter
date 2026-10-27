# Beam Lab (web)

Hydrophone beam-scan viewer in the browser. Users upload `.mat` plane scans and get the
beam map, focal metrics, slice thickness and on-axis signal vs depth, with saved views and
a personal scan library. Nothing to install on the user's side.

```
beamlab-web/
├── app.py                 Streamlit user interface
├── beamcore.py            analysis (no UI) – same maths as the desktop viewer
├── store.py               per-user library + saved views on the server's disk
├── requirements.txt
├── Dockerfile
└── .streamlit/
    ├── config.toml        light + dark themes, upload limit
    └── secrets.toml.example   sign-in template (copy to secrets.toml to require login)
```

## 1. Run it on your Mac (test)

```bash
cd ~/Downloads/beamlab-web
~/beamviewer-env/bin/pip install -r requirements.txt        # reuses your existing env
~/beamviewer-env/bin/streamlit run app.py
```

It opens http://localhost:8501. Drop a `.mat` file into the sidebar.

## 2. Share it on the office network (lab PC)

Run the same command on an always-on machine. Colleagues open `http://<that-machine>:8501`.
Without `secrets.toml` there is **no login** and everyone shares one library, so use this only
on a trusted network.

## 3. Company hosting with Microsoft sign-in

1. **IT registers an app** in Microsoft Entra ID (single tenant). Redirect URI:
   `https://<your-app-address>/oauth2callback`. Create a client secret.
2. **Copy** `.streamlit/secrets.toml.example` to `secrets.toml` and fill in the client ID,
   secret, tenant ID, redirect URI and a random `cookie_secret`. Keep `allowed_domains`
   to restrict access to company accounts.
3. **Build and run the container** on Azure App Service (or any Docker host):
   ```bash
   docker build -t beamlab .
   docker run -p 8501:8501 -v beamlab-data:/data \
       -v $PWD/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro beamlab
   ```
   On Azure, put `secrets.toml` in the app's secret store / mounted storage and attach
   persistent storage at `/data` (that's where uploads and views live).

Each signed-in person gets their own library and saved views, stored under
`/data/<email>/`.

## Notes

* **Light / dark:** app menu (⋮, top right) → Settings. Defaults to the device's setting.
* **Data:** uploaded scans are kept on the server until the user removes them (⋯ next to the
  scan). Back up `/data` like any other file share.
* **Streamlit Community Cloud** also works (deploy from a GitHub repo), but scans would sit on
  a third-party service and its disk isn't permanent – check with IT before using it for
  company test data.
* **Limits:** 200 MB per upload (`server.maxUploadSize` in `config.toml`). Very high upscale
  (16×) makes the interactive map slower; the analysis always uses the raw grid.
