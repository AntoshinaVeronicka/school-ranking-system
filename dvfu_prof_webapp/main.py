from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
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


@app.exception_handler(PermissionError)
async def handle_permission_error(request: Request, exc: PermissionError):
    detail = str(exc) or "permission denied"
    accept = (request.headers.get("accept") or "").lower()
    wants_html = "text/html" in accept

    if detail == "not authenticated":
        if wants_html:
            return RedirectResponse(url="/login", status_code=303)
        return JSONResponse(status_code=401, content={"detail": detail})

    if wants_html:
        return HTMLResponse("Forbidden", status_code=403)
    return JSONResponse(status_code=403, content={"detail": detail})


# Keep startup behavior the same as before: create tables/seeds on import.
init_db(hash_password)
