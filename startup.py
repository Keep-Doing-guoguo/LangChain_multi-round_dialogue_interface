import asyncio
import multiprocessing as mp
import os
import subprocess
import sys
from multiprocessing import Process
from datetime import datetime
from pprint import pprint

# 设置numexpr最大线程数，默认为CPU核心数
try:
    import numexpr

    n_cores = numexpr.utils.detect_number_of_cores()
    os.environ["NUMEXPR_MAX_THREADS"] = str(n_cores)
except:
    pass

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs import (
    LOG_PATH,
    log_verbose,
    logger,
    LLM_MODELS,
    EMBEDDING_MODEL,
    FSCHAT_CONTROLLER,
    FSCHAT_OPENAI_API,
    FSCHAT_MODEL_WORKERS,
    WEBUI_SERVER,
    API_SERVER,
    FSCHAT_OPENAI_API,
)
from server.utils import (fschat_controller_address, fschat_model_worker_address,
                          fschat_openai_api_address,
                          get_model_worker_config, get_all_model_worker_configs,
                          MakeFastAPIOffline, FastAPI, llm_device, embedding_device)

import argparse
from typing import Tuple, List, Dict



def create_controller_app(
        dispatch_method: str,
        log_level: str = "INFO",
) -> FastAPI:
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    from fastchat.serve.controller import app, Controller, logger
    logger.setLevel(log_level)

    controller = Controller(dispatch_method)#那是因为在这里更改了一个参数，所以才有下面的添加。
    sys.modules["fastchat.serve.controller"].controller = controller#sys.modules["fastchat.serve.controller"].controller = controller 是为了让 controller 实例在整个项目中都能被访问到，相当于做了全局注册。

    MakeFastAPIOffline(app)
    app.title = "FastChat Controller"
    app._controller = controller
    return app


def create_model_worker_app(log_level: str = "INFO", **kwargs) -> FastAPI:
    """
    kwargs包含的字段如下：
    host:
    port:
    model_names:[`model_name`]
    controller_address:
    worker_address:

    对于Langchain支持的模型：
        langchain_model:True
        不会使用fschat
    对于online_api:
        online_api:True
        worker_class: `provider`
    对于离线模型：
        model_path: `model_name_or_path`,huggingface的repo-id或本地路径
        device:`LLM_DEVICE`
    """
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args([])

    for k, v in kwargs.items():
        setattr(args, k, v)
    if worker_class := kwargs.get("langchain_model"):  # Langchian支持的模型不用做操作
        from fastchat.serve.base_model_worker import app
        worker = ""
    # 在线模型API
    elif worker_class := kwargs.get("worker_class"):
        from fastchat.serve.base_model_worker import app

        worker = worker_class(model_names=args.model_names,
                              controller_addr=args.controller_address,
                              worker_addr=args.worker_address)
        # sys.modules["fastchat.serve.base_model_worker"].worker = worker
        sys.modules["fastchat.serve.base_model_worker"].logger.setLevel(log_level)
    # 本地模型
    else:
        from configs.model_config import VLLM_MODEL_DICT
        if kwargs["model_names"][0] in VLLM_MODEL_DICT and args.infer_turbo == "vllm":
            import fastchat.serve.vllm_worker
            from fastchat.serve.vllm_worker import VLLMWorker, app, worker_id
            from vllm import AsyncLLMEngine
            from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs

            args.tokenizer = args.model_path  # 如果tokenizer与model_path不一致在此处添加
            args.tokenizer_mode = 'auto'
            args.trust_remote_code = True
            args.download_dir = None
            args.load_format = 'auto'
            args.dtype = 'auto'
            args.seed = 0
            args.worker_use_ray = False
            args.pipeline_parallel_size = 1
            args.tensor_parallel_size = 1
            args.block_size = 16
            args.swap_space = 4  # GiB
            args.gpu_memory_utilization = 0.90
            args.max_num_batched_tokens = None  # 一个批次中的最大令牌（tokens）数量，这个取决于你的显卡和大模型设置，设置太大显存会不够
            args.max_num_seqs = 256
            args.disable_log_stats = False
            args.conv_template = None
            args.limit_worker_concurrency = 5
            args.no_register = False
            args.num_gpus = 1  # vllm worker的切分是tensor并行，这里填写显卡的数量
            args.engine_use_ray = False
            args.disable_log_requests = False

            # 0.2.1 vllm后要加的参数, 但是这里不需要
            args.max_model_len = None
            args.revision = None
            args.quantization = None
            args.max_log_len = None
            args.tokenizer_revision = None

            # 0.2.2 vllm需要新加的参数
            args.max_paddings = 256

            if args.model_path:
                args.model = args.model_path
            if args.num_gpus > 1:
                args.tensor_parallel_size = args.num_gpus

            for k, v in kwargs.items():
                setattr(args, k, v)

            engine_args = AsyncEngineArgs.from_cli_args(args)
            engine = AsyncLLMEngine.from_engine_args(engine_args)

            worker = VLLMWorker(
                controller_addr=args.controller_address,
                worker_addr=args.worker_address,
                worker_id=worker_id,
                model_path=args.model_path,
                model_names=args.model_names,
                limit_worker_concurrency=args.limit_worker_concurrency,
                no_register=args.no_register,
                llm_engine=engine,
                conv_template=args.conv_template,
            )
            sys.modules["fastchat.serve.vllm_worker"].engine = engine
            sys.modules["fastchat.serve.vllm_worker"].worker = worker
            sys.modules["fastchat.serve.vllm_worker"].logger.setLevel(log_level)

        else:
            from fastchat.serve.model_worker import app, GptqConfig, AWQConfig, ModelWorker, worker_id

            args.gpus = "0"  # GPU的编号,如果有多个GPU，可以设置为"0,1,2,3"
            args.max_gpu_memory = "22GiB"
            args.num_gpus = 1  # model worker的切分是model并行，这里填写显卡的数量

            args.load_8bit = False
            args.cpu_offloading = None
            args.gptq_ckpt = None
            args.gptq_wbits = 16
            args.gptq_groupsize = -1
            args.gptq_act_order = False
            args.awq_ckpt = None
            args.awq_wbits = 16
            args.awq_groupsize = -1
            args.model_names = [""]
            args.conv_template = None
            args.limit_worker_concurrency = 5
            args.stream_interval = 2
            args.no_register = False
            args.embed_in_truncate = False
            for k, v in kwargs.items():
                setattr(args, k, v)
            if args.gpus:
                if args.num_gpus is None:
                    args.num_gpus = len(args.gpus.split(','))
                if len(args.gpus.split(",")) < args.num_gpus:
                    raise ValueError(
                        f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
                    )
                os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
            gptq_config = GptqConfig(
                ckpt=args.gptq_ckpt or args.model_path,
                wbits=args.gptq_wbits,
                groupsize=args.gptq_groupsize,
                act_order=args.gptq_act_order,
            )
            awq_config = AWQConfig(
                ckpt=args.awq_ckpt or args.model_path,
                wbits=args.awq_wbits,
                groupsize=args.awq_groupsize,
            )

            worker = ModelWorker(
                controller_addr=args.controller_address,
                worker_addr=args.worker_address,
                worker_id=worker_id,
                model_path=args.model_path,
                model_names=args.model_names,
                limit_worker_concurrency=args.limit_worker_concurrency,
                no_register=args.no_register,
                device=args.device,
                num_gpus=args.num_gpus,
                max_gpu_memory=args.max_gpu_memory,
                load_8bit=args.load_8bit,
                cpu_offloading=args.cpu_offloading,
                gptq_config=gptq_config,
                awq_config=awq_config,
                stream_interval=args.stream_interval,
                conv_template=args.conv_template,
                embed_in_truncate=args.embed_in_truncate,
            )
            sys.modules["fastchat.serve.model_worker"].args = args
            sys.modules["fastchat.serve.model_worker"].gptq_config = gptq_config
            # sys.modules["fastchat.serve.model_worker"].worker = worker
            sys.modules["fastchat.serve.model_worker"].logger.setLevel(log_level)

    MakeFastAPIOffline(app)
    app.title = f"FastChat LLM Server ({args.model_names[0]})"
    app._worker = worker
    return app


def create_openai_api_app(
        controller_address: str,
        api_keys: List = [],
        log_level: str = "INFO",
) -> FastAPI:
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    from fastchat.serve.openai_api_server import app, CORSMiddleware, app_settings
    from fastchat.utils import build_logger
    logger = build_logger("openai_api", "openai_api.log")
    logger.setLevel(log_level)

    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    sys.modules["fastchat.serve.openai_api_server"].logger = logger
    app_settings.controller_address = controller_address
    app_settings.api_keys = api_keys

    MakeFastAPIOffline(app)
    app.title = "FastChat OpeanAI API Server"
    return app

#当前 FastAPI 应用已完成启动，可以通知父进程或控制逻辑继续下一步
def _set_app_event(app: FastAPI, started_event: mp.Event = None):
    @app.on_event("startup")#	FastAPI 有一个生命周期钩子
    async def on_startup():#这个函数在服务 启动完成时自动执行
        if started_event is not None:#started_event.set() 是触发一个 mp.Event（跨进程的事件通知）
            started_event.set()

#启动 FastChat 的 Controller 控制器，负责调度所有模型 worker，实现请求分发和模型状态管理。
def run_controller(log_level: str = "INFO", started_event: mp.Event = None):
    import uvicorn
    import httpx
    from fastapi import Body
    import time
    import sys

    app = create_controller_app(
        dispatch_method=FSCHAT_CONTROLLER.get("dispatch_method"),#是 FastChat 中用于指定 模型分发策略 的参数
        log_level=log_level,
    )
    _set_app_event(app, started_event)
    #这段逻辑的目的是在服务成功启动时，用 started_event.set() 来通知主进程/其他子进程，属于多进程协作的同步机制。这样可以做到：等控制器起来了，别的模块再启动，比如 model_worker。
    # add interface to release and load model worker
    @app.post("/release_worker")
    def release_worker(
            model_name: str = Body(..., description="要释放模型的名称", samples=["chatglm-6b"]),
            # worker_address: str = Body(None, description="要释放模型的地址，与名称二选一", samples=[FSCHAT_CONTROLLER_address()]),
            new_model_name: str = Body(None, description="释放后加载该模型"),
            keep_origin: bool = Body(False, description="不释放原模型，加载新模型")
    ) -> Dict:
        available_models = app._controller.list_models()
        if new_model_name in available_models:
            msg = f"要切换的LLM模型 {new_model_name} 已经存在"
            logger.info(msg)
            return {"code": 500, "msg": msg}

        if new_model_name:
            logger.info(f"开始切换LLM模型：从 {model_name} 到 {new_model_name}")
        else:
            logger.info(f"即将停止LLM模型： {model_name}")

        if model_name not in available_models:
            msg = f"the model {model_name} is not available"
            logger.error(msg)
            return {"code": 500, "msg": msg}

        worker_address = app._controller.get_worker_address(model_name)
        if not worker_address:
            msg = f"can not find model_worker address for {model_name}"
            logger.error(msg)
            return {"code": 500, "msg": msg}



    host = FSCHAT_CONTROLLER["host"]
    port = FSCHAT_CONTROLLER["port"]

    if log_level == "ERROR":
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())

#启动本地或远程的 模型推理进程（支持离线、LangChain、在线API、多卡并行等），绑定模型到控制器。
def run_model_worker(
        model_name: str = LLM_MODELS[0],
        controller_address: str = "",
        log_level: str = "INFO",
        q: mp.Queue = None,
        started_event: mp.Event = None,
):
    import uvicorn
    from fastapi import Body
    import sys


    kwargs = get_model_worker_config(model_name)
    host = kwargs.pop("host")
    port = kwargs.pop("port")
    kwargs["model_names"] = [model_name]
    kwargs["controller_address"] = controller_address or fschat_controller_address()
    kwargs["worker_address"] = fschat_model_worker_address(model_name)
    model_path = kwargs.get("model_path", "")
    kwargs["model_path"] = model_path

    app = create_model_worker_app(log_level=log_level, **kwargs)
    _set_app_event(app, started_event)
    if log_level == "ERROR":
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    # add interface to release and load model
    @app.post("/release")
    def release_model(
            new_model_name: str = Body(None, description="释放后加载该模型"),
            keep_origin: bool = Body(False, description="不释放原模型，加载新模型")
    ) -> Dict:
        if keep_origin:
            if new_model_name:
                q.put([model_name, "start", new_model_name])
        else:
            if new_model_name:
                q.put([model_name, "replace", new_model_name])
            else:
                q.put([model_name, "stop", None])
        return {"code": 200, "msg": "done"}

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())

#启动一个兼容 OpenAI 接口规范的服务（即 /v1/chat/completions），用于无缝对接 OpenAI SDK。
def run_openai_api(log_level: str = "INFO", started_event: mp.Event = None):
    import uvicorn
    import sys

    controller_addr = fschat_controller_address()
    app = create_openai_api_app(controller_addr, log_level=log_level)  # TODO: not support keys yet.
    _set_app_event(app, started_event)#通知api_server已经启动成功。

    host = FSCHAT_OPENAI_API["host"]
    port = FSCHAT_OPENAI_API["port"]
    if log_level == "ERROR":
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    uvicorn.run(app, host=host, port=port)

#启动自定义的 API 接口（如文档管理、上传、模型切换等）供前端或管理后台使用。





def parse_args():
    parser = argparse.ArgumentParser(description="示例参数解析")
    parser.add_argument("--openai-api", action="store_true", help="启用 OpenAI API",default=True)
    parser.add_argument("--model-worker", action="store_true", help="启用模型 Worker",default=True)
    parser.add_argument("--api", action="store_true", help="启用 API 服务器",default=True)
    parser.add_argument("--api-worker", action="store_true", help="启用 API Worker",default=True)
    parser.add_argument("--webui", action="store_true", help="启用 Web UI",default=True)
    parser.add_argument("--model-name", action="store_true", help="启用 Web UI",default=LLM_MODELS)
    #controller_address
    parser.add_argument("--controller_address",default="http://127.0.0.1:21001")

    return parser.parse_args(), parser


def start_main_server():
    import time
    mp.set_start_method("spawn")
    manager = mp.Manager()#创建一个共享对象管理器，可以用于多个进程之间的通信。
    run_mode = None
    queue = manager.Queue()#共享队列
    args, parser = parse_args()
    processes = {
        "online_api": {},
        "model_worker": {}
    }
    def process_count():
        return len(processes) + len(processes["online_api"]) + len(processes["model_worker"]) - 2
    log_level = "INFO"
    controller_started = manager.Event()#同步信号：一个进程通知另一个进程

    if args.openai_api:#创建两个子进程，并放入 processes 字典中管理。
        process = Process(
            target=run_controller,
            name=f"controller",
            kwargs=dict(log_level=log_level, started_event=controller_started),
            daemon=True,
        )  # 创建进程。FastChat Controller 服务（用于管理所有模型 worker 的状态调度）
        processes["controller"] = process

        process = Process(
            target=run_openai_api,
            name=f"openai_api",
            daemon=True,
        )  # 创建进程。OpenAI API 服务（兼容 OpenAI 接口风格的服务）
        processes["openai_api"] = process

    model_worker_started = []#worker是最重要的。
    if args.model_worker:#是为了启动 本地模型（offline）或非 API 接入模型 worker。
        for model_name in args.model_name:
            config = get_model_worker_config(model_name)
            if not config.get("online_api"):#config.get("online_api") 是 False，即不是通过 API 接入的在线模型。
                e = manager.Event()
                model_worker_started.append(e)
                process = Process(
                    target=run_model_worker,
                    name=f"model_worker - {model_name}",#启动的进程名为 model_worker - {model_name}。
                    kwargs=dict(model_name=model_name,
                                controller_address=args.controller_address,
                                log_level=log_level,
                                q=queue,
                                started_event=e),
                    daemon=True,
                )  # 创建进程。	•	对应的 worker 用于加载模型、注册到 controller，并接受推理请求。
                processes["model_worker"][model_name] = process

    if args.api_worker:#是为了启动 在线模型（通过 API 提供服务）的 worker。
        for model_name in args.model_name:
            config = get_model_worker_config(model_name)#
            if (config.get("online_api")#	config.get("online_api") 是 True，说明是调用远程 API 的模型，比如 OpenAI 或阿里通义。
                    and config.get("worker_class")#config.get("worker_class") 指定了用哪个类来处理这种 API worker。
                    and model_name in FSCHAT_MODEL_WORKERS):#这些模型不是本地加载的，而是通过 HTTP 请求远程模型服务。
                e = manager.Event()
                model_worker_started.append(e)#这些 worker 实际上是 “代理转发”角色，负责接收任务，然后通过 API 请求远程模型服务。
                process = Process(
                    target=run_model_worker,
                    name=f"api_worker - {model_name}",
                    kwargs=dict(model_name=model_name,
                                controller_address=args.controller_address,
                                log_level=log_level,
                                q=queue,
                                started_event=e),
                    daemon=True,
                )
                processes["online_api"][model_name] = process

    if process_count() == 0:
        parser.print_help()

if __name__ == "__main__":

    start_main_server()


# 服务启动后接口调用示例：
# import openai
# openai.api_key = "EMPTY" # Not support yet
# openai.api_base = "http://localhost:8888/v1"

# model = "chatglm2-6b"

# # create a chat completion
# completion = openai.ChatCompletion.create(
#   model=model,
#   messages=[{"role": "user", "content": "Hello! What is your name?"}]
# )
# # print the completion
# print(completion.choices[0].message.content)
