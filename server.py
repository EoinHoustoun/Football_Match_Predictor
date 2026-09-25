"""F_PRED entrypoint: the Streamlit app plus the files a phone asks for.

`streamlit run server.py` serves app.py exactly as before and adds the root
files a phone requests when the page is added to its home screen: iOS asks
for /apple-touch-icon.png without being told, Android reads /manifest.json.
Needs Streamlit 1.57+ (st.App custom routes); the UI runs from .venv on 1.64.
"""
from pathlib import Path

import streamlit as st
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

ASSETS = Path(__file__).resolve().parent / "assets"


def _icon(size: int):
    async def handler(request):
        return FileResponse(ASSETS / f"icon-{size}.png", media_type="image/png",
                            headers={"Cache-Control": "public, max-age=86400"})
    return handler


async def manifest(request):
    return JSONResponse({
        "name": "F_PRED", "short_name": "F_PRED",
        "description": "Premier League draw model, paper trading",
        "start_url": "/", "display": "standalone",
        "background_color": "#0a0e1a", "theme_color": "#0a0e1a",
        "icons": [{"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
                  {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"}],
    }, media_type="application/manifest+json")


app = st.App("app.py", routes=[
    Route("/apple-touch-icon.png", _icon(180)),
    Route("/apple-touch-icon-precomposed.png", _icon(180)),
    Route("/icon-192.png", _icon(192)),
    Route("/icon-512.png", _icon(512)),
    Route("/manifest.json", manifest),
])
