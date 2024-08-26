from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware

from server.routes.musetalk import router


def init_routers(app_: FastAPI) -> None:
    musetalk_router = router
    app_.include_router(musetalk_router)

def make_middleware() -> list[Middleware]:
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        ),
    ]
    return middleware

def create_app() -> FastAPI:
    app_ = FastAPI(
        title="Hide",
        description="Hide API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        middleware=make_middleware(),
    )
    init_routers(app_=app_)
    return app_


app = create_app()