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