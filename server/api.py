import argparse

import uvicorn

from starlette.responses import RedirectResponse

from utils import FastAPI,MakeFastAPIOffline

from server.utils import BaseResponse
from server.chat.chat import chat
def create_app(run_mode: str = None):
    app = FastAPI(
        title="API Server",
        version='1.0'
    )
    MakeFastAPIOffline(app)
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    mount_app_routes(app, run_mode=run_mode)
    return app

#重定向（Redirect）是指 将用户的请求从一个 URL 地址自动引导到另一个 URL 地址 的操作。
async def document():
    return RedirectResponse(url="/docs")#如果用户记不住 /docs，你可以用更短、更好记的路径（如 /document）进行重定向。如果需要多个路由指向同一个文档页面，可以使用这种方式。

def mount_app_routes(app: FastAPI, run_mode: str = None):
    app.get("/",
            response_model = BaseResponse,
            summary="swagger 文档"
    )(document)#这个设计可以提升用户体验，让用户无需记住 docs 路径即可快速访问 Swagger 文档页面。

    app.post("/chat/chat",
             tags=["Chat"],
             summary="与llm模型对话(通过LLMChain)",
             )(chat)
    # app.post("/chat/search_engine_chat",
    #          tags=["Chat"],
    #          summary="与搜索引擎对话",
    #          )(search_engine_chat)

def run_api(host,port,**kwargs):
    #在这段代码中，ssl_keyfile 和 ssl_certfile 是配置 SSL/TLS 的文件，用于启用 HTTPS（加密的 HTTP）协议。
    
    '''
    私钥文件（通常为 .key 文件）。
    公共证书文件（通常为 .crt 或 .pem 文件）。
    '''
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(app,
                    host=host,
                    port=port,
                    ssl_keyfile = kwargs.get("ssl_keyfile"),
                    ssl_certfile = kwargs.get("ssl_certfile")
        )
    else:
        uvicorn.run(app,host=host,port=port)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="chatchat",
                                    description = "知识库问答大模型"
    )
    parser.add_argument("--host",type=str,default="0.0.0.0")
    parser.add_argument("--port",type=int,default=7861)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)

    args = parser.parse_args()
    args_dict = vars(args)
    # 打印解析的参数
    print("解析后的参数:", args_dict)
    app = create_app()

    run_api(host = args.host,port=args.port,ssl_keyfile=args.ssl_keyfile,ssl_certfile=args.ssl_certfile)


