import asyncio
import logging

import pydantic
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI
from pathlib import Path
from typing import Literal, Optional, Callable, Generator, Dict, Any, Awaitable, Union, Tuple
from configs import logger, log_verbose
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, AzureOpenAI, Anthropic
import os
os.environ["OPENAI_API_KEY"] = "sk-79fcaf8f7fe24839b4abcbdd9c9e8980"
#fn 一个可等待的任务（如协程函数）。用于在任务完成或出错时通知其他任务或流程。
async def wrap_done(fn: Awaitable, event: asyncio.Event):#并在任务完成或发生异常时，利用 event 发出信号。这种结构对于控制和管理异步任务的状态非常有用。
    """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
    try:
        await fn#函数尝试直接 await 传入的任务 fn，即执行它的逻辑。
    except Exception as e:#n 在运行过程中抛出了异常，捕获异常并进行处理：
        logging.exception(e)
        # TODO: handle exception
        msg = f"Caught exception: {e}"#使用 logging.exception 和 logger.error 记录异常信息，便于调试。
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
    finally:
        # Signal the aiter to stop.
        event.set()#无论任务是正常完成还是发生了异常，finally 块都会执行。这是关键的一步，发出信号，表示任务已经完成或异常被处理。通常，这种信号会被其他等待该事件的协程或任务捕捉到，从而继续执行后续逻辑。
'''
任务同步：在异步编程中，你可能有多个并发任务在运行，需要知道某个任务何时完成，以便触发其他操作。通过 asyncio.Event，你可以实现这种任务同步。

异常处理：在复杂的异步系统中，处理任务异常是至关重要的。wrap_done 提供了一个集中处理异常的地方，同时确保即使任务失败，也会通过 event 通知系统。

流控制：在某些情况下，你可能希望在特定任务完成后，控制整个任务流的继续。通过使用 wrap_done，你可以实现这种精细的控制。
'''

def get_ChatOpenAI(
        model_name: str,
        temperature: float,
        max_tokens: int = None,
        streaming: bool = True,
        callbacks: List[Callable] = [],
        verbose: bool = True,
        **kwargs: Any,
) -> ChatOpenAI:

    model = ChatOpenAI(
        streaming=streaming,
        verbose=verbose,
        callbacks=callbacks,
        openai_api_key='sk-79fcaf8f7fe24839b4abcbdd9c9e8980',
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus",
        temperature=0.7,
        max_tokens=max_tokens,
        **kwargs
    )
    return model
def get_OpenAI(

) -> OpenAI:
    model = OpenAI(
        openai_api_key="sk-79fcaf8f7fe24839b4abcbdd9c9e8980",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus",
    )

    return model

class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="API status code")
    msg: str = pydantic.Field("success", description="API status message")
    data: Any = pydantic.Field(None, description="API data")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }

def MakeFastAPIOffline(
        app: FastAPI,
        static_dir=Path(__file__).parent / "static",
        static_url="/static-offline-docs",
        docs_url: Optional[str] = "/docs",
        redoc_url: Optional[str] = "/redoc",
) -> None:
    """patch the FastAPI obj that doesn't rely on CDN for the documentation page"""
    from fastapi import Request
    from fastapi.openapi.docs import (
        get_redoc_html,
        get_swagger_ui_html,
        get_swagger_ui_oauth2_redirect_html,
    )
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import HTMLResponse

    openapi_url = app.openapi_url
    swagger_ui_oauth2_redirect_url = app.swagger_ui_oauth2_redirect_url

    def remove_route(url: str) -> None:
        '''
        remove original route from app
        '''
        index = None
        for i, r in enumerate(app.routes):
            if r.path.lower() == url.lower():
                index = i
                break
        if isinstance(index, int):
            app.routes.pop(index)

    # Set up static file mount
    app.mount(
        static_url,
        StaticFiles(directory=Path(static_dir).as_posix()),
        name="static-offline-docs",
    )

    if docs_url is not None:
        remove_route(docs_url)
        remove_route(swagger_ui_oauth2_redirect_url)

        # Define the doc and redoc pages, pointing at the right files
        @app.get(docs_url, include_in_schema=False)
        async def custom_swagger_ui_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"
            return get_swagger_ui_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - Swagger UI",
                oauth2_redirect_url=swagger_ui_oauth2_redirect_url,
                swagger_js_url=f"{root}{static_url}/swagger-ui-bundle.js",
                swagger_css_url=f"{root}{static_url}/swagger-ui.css",
                swagger_favicon_url=favicon,
            )

        @app.get(swagger_ui_oauth2_redirect_url, include_in_schema=False)
        async def swagger_ui_redirect() -> HTMLResponse:
            return get_swagger_ui_oauth2_redirect_html()

    if redoc_url is not None:
        remove_route(redoc_url)

        @app.get(redoc_url, include_in_schema=False)
        async def redoc_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"

            return get_redoc_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - ReDoc",
                redoc_js_url=f"{root}{static_url}/redoc.standalone.js",
                with_google_fonts=False,
                redoc_favicon_url=favicon,
            )
def get_prompt_template(type: str, name: str) -> Optional[str]:#	•	返回模板内容的字符串（str），如果找不到则返回 None（Optional[str]）。
    '''
    从prompt_config中加载模板内容
    type: "llm_chat","agent_chat","knowledge_base_chat","search_engine_chat"的其中一种，如果有新功能，应该进行加入。
    '''

    from configs import prompt_config
    import importlib
    importlib.reload(prompt_config)  # TODO: 检查configs/prompt_config.py文件有修改再重新加载动态重新加载 prompt_config 模块。
    return prompt_config.PROMPT_TEMPLATES[type].get(name)#type = llm_chat；name = default或者with_history






