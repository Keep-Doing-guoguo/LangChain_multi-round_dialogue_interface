import sys
from configs.model_config import LLM_DEVICE

# API 是否开启跨域，默认为False，如果需要开启，请设置为True
# is open cross domain
OPEN_CROSS_DOMAIN = False
# 各服务器默认绑定host。如改为"0.0.0.0"需要修改下方所有XX_SERVER的host
DEFAULT_BIND_HOST = "0.0.0.0" if sys.platform != "win32" else "127.0.0.1"
HTTPX_DEFAULT_TIMEOUT = 300.0
# webui.py server
WEBUI_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": 8501,
}

# api.py server
API_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": 7861,
}

# fastchat openai_api server
FSCHAT_OPENAI_API = {
    "host": DEFAULT_BIND_HOST,
    "port": 20000,
}

#这段代码定义了 FSCHAT_MODEL_WORKERS 配置字典，用于管理和定义多个模型的运行参数、设备配置以及推理服务的端口。以下是详细的解释：
FSCHAT_MODEL_WORKERS = {
    # 所有模型共用的默认配置，可在模型专项配置中进行覆盖。
    "default": {
        "host": DEFAULT_BIND_HOST,
        "port": 20002,
        "device": LLM_DEVICE,
        # False,'vllm',使用的推理加速框架,使用vllm如果出现HuggingFace通信问题，参见doc/FAQ
        # vllm对一些模型支持还不成熟，暂时默认关闭
        # fschat=0.2.33的代码有bug, 如需使用，源码修改fastchat.server.vllm_worker，
        # 将103行中sampling_params = SamplingParams的参数stop=list(stop)修改为stop= [i for i in stop if i!=""]
        "infer_turbo": False,

        # model_worker多卡加载需要配置的参数
        # "gpus": None, # 使用的GPU，以str的格式指定，如"0,1"，如失效请使用CUDA_VISIBLE_DEVICES="0,1"等形式指定
        # "num_gpus": 1, # 使用GPU的数量
        # "max_gpu_memory": "20GiB", # 每个GPU占用的最大显存

        # 以下为model_worker非常用参数，可根据需要配置
        # "load_8bit": False, # 开启8bit量化
        # "cpu_offloading": None,
        # "gptq_ckpt": None,
        # "gptq_wbits": 16,
        # "gptq_groupsize": -1,
        # "gptq_act_order": False,
        # "awq_ckpt": None,
        # "awq_wbits": 16,
        # "awq_groupsize": -1,
        # "model_names": LLM_MODELS,
        # "conv_template": None,
        # "limit_worker_concurrency": 5,
        # "stream_interval": 2,
        # "no_register": False,
        # "embed_in_truncate": False,

        # 以下为vllm_worker配置参数,注意使用vllm必须有gpu，仅在Linux测试通过

        # tokenizer = model_path # 如果tokenizer与model_path不一致在此处添加
        # 'tokenizer_mode':'auto',
        # 'trust_remote_code':True,
        # 'download_dir':None,
        # 'load_format':'auto',
        # 'dtype':'auto',
        # 'seed':0,
        # 'worker_use_ray':False,
        # 'pipeline_parallel_size':1,
        # 'tensor_parallel_size':1,
        # 'block_size':16,
        # 'swap_space':4 , # GiB
        # 'gpu_memory_utilization':0.90,
        # 'max_num_batched_tokens':2560,
        # 'max_num_seqs':256,
        # 'disable_log_stats':False,
        # 'conv_template':None,
        # 'limit_worker_concurrency':5,
        # 'no_register':False,
        # 'num_gpus': 1
        # 'engine_use_ray': False,
        # 'disable_log_requests': False

    },
    # 可以如下示例方式更改默认配置
    # "Qwen-1_8B-Chat": { # 使用default中的IP和端口
    #    "device": "cpu",
    # },
    "chatglm3-6b": {  # 使用default中的IP和端口
        "device": "cuda",
    },

    # 以下配置可以不用修改，在model_config中设置启动的模型
    "zhipu-api": {
        "port": 21001,
    },
    "minimax-api": {
        "port": 21002,
    },
    "xinghuo-api": {
        "port": 21003,
    },
    "qianfan-api": {
        "port": 21004,
    },
    "fangzhou-api": {
        "port": 21005,
    },
    "qwen-api": {
        "port": 21006,
    },

}

# fastchat controller server
FSCHAT_CONTROLLER = {
    "host": DEFAULT_BIND_HOST,
    "port": 20001,
    "dispatch_method": "shortest_queue",
}
'''
dispatch_method 是 FastChat 中用于指定 模型分发策略 的参数，作用是：
🎯 控制多个模型 worker 时，如何分配请求给它们
⸻
✅ 举个例子：

假设你有 3 个模型 worker 实例都服务同一个模型（比如 "chatglm-6b"），那么 controller 就要决定：
	•	哪一个 worker 来处理当前这个请求？
⸻
dispatch_method 常见的取值有：

值	含义说明
"shortest_queue"	默认值，选择当前等待队列最短的 worker（响应快）
"round_robin"	轮询方式分发（每个轮一遍）
"fixed"	固定分配，第一个注册的 worker 永远处理请求
"random"	随机选择一个 worker
⸻
dispatch_method 决定了 请求转发给哪个模型 worker，默认是 shortest_queue（也就是谁最空闲分给谁）。你可以根据业务需求切换分发策略。
'''