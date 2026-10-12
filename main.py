"""Convenience launcher: ``python main.py`` starts the API server.

The application itself lives in ``app/main.py`` so that the package structure
stays clean; this file only exists so a newcomer can start the server with one
command.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
