from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

try:
    from .config import APP_NAME, SECRET_KEY, STATIC_DIR
    from .db import init_db
    from .routes import router
    from .services import hash_password
except ImportError:
    from config import APP_NAME, SECRET_KEY, STATIC_DIR
    from db import init_db
    from routes import router
    from services import hash_password


app = FastAPI(title=APP_NAME, default_response_class=HTMLResponse)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(router)

# Keep startup behavior the same as before: create tables/seeds on import.
init_db(hash_password)
