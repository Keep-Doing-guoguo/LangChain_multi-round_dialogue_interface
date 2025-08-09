fastchat启动模型：
启动 Controller（控制器）​
python3 -m fastchat.serve.controller --host 0.0.0.0 --port 21001



启动 Model Worker（模型工作器）​---int8的不能使用mac来进行加载，因为需要使用到gpu。
python3 -m fastchat.serve.model_worker \
  --model-path /Volumes/PSSD/models/Qwen/Qwen-1_8B-Chat \
  --controller http://localhost:21001 \
  --port 31000 \
  --worker http://localhost:31000 \
  --limit-worker-concurrency 5 \
  --device mps



启动 API 服务
python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8800
python -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8800
启动 Gradio 界面进行对话测试：
python3 -m fastchat.serve.webui --port 7860
