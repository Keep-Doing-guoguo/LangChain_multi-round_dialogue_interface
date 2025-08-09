

## 项目结构说明

```
.
├── configs/                       # 配置文件（预留）
├── knowledge_base/               # 知识库管理（基于 SQLite）
│   ├── check_sql_data.py         # 数据检查与导入脚本
│   └── info.db                   # SQLite 数据库文件
├── server/                       # 核心服务代码
│   ├── callback_handler/         # 回调处理模块
│   │   └── conversation_callback_handler.py
│   ├── chat/                     # 对话逻辑
│   │   ├── chat.py               # 主聊天入口
│   │   └── utils.py              # 工具函数
│   ├── db/                       # 数据库模型与会话持久化
│   │   ├── models/               # ORM 数据模型
│   │   │   ├── conversation_model.py
│   │   │   └── message_model.py
│   │   ├── repository/           # 数据访问层（Repository 模式）
│   │   │   ├── conversation_repository.py
│   │   │   └── message_repository.py
│   │   ├── base.py               # Base 类定义
│   │   └── session.py            # DB 会话控制
│   ├── knowledge_base/           # 知识库数据处理逻辑
│   │   └── migrate.py
│   ├── memory/                   # 会话记忆管理
│   │   └── conversation_db_buffer_memory.py
│   ├── reranker/                 # reranker 模块（重排序）
│   │   └── reranker.py
│   ├── static/                   # 静态文件（预留）
│   ├── api.py                    # API 接口路由
│   └── utils.py                  # 通用工具函数
├── webui_pages/                  # WebUI 前端页面（Gradio/Streamlit）
```


## 一、项目整体介绍

这是一个 **轻量级、模块化、可扩展的本地知识库问答系统**，结合了：
- **大语言模型（LLM）**：用于生成回答
- **向量数据库（如 Milvus/FAISS）**：用于知识检索
- **SQLite 数据库**：用于持久化对话记录
- **LangChain 框架**：构建对话链、记忆管理、回调处理
- **FastAPI**：提供 API 接口
- **Gradio/Streamlit（预留）**：未来可接入 WebUI

> 适用于教学场景：从数据库设计 → 会话管理 → 知识检索 → 流式输出，覆盖完整 AI 应用开发流程。

---

## 二、模块详解（逐层解析）

### `configs/`
- **用途**：存放配置文件（如模型名称、API 密钥、向量库参数等）
---

### `knowledge_base/`

#### `check_sql_data.py`
- **功能**：检查并打印数据库表结构和数据，用于调试

#### `info.db`
- **功能**：SQLite 数据库文件
- **包含表**：
  - `conversation`：对话元信息（ID、名称、创建时间）
  - `message`：消息记录（query/response/conversation_id）
  - `knowledge_base`, `knowledge_file`, `file_doc`：知识库相关表
---

### `server/` —— 核心服务层

#### `callback_handler/conversation_callback_handler.py`
- **功能**：监听 LLM 输出，自动将 response 写入数据库
- **关键技术**：
  - 继承 `BaseCallbackHandler`
  - `on_llm_end` 事件捕获生成结果
  - 调用 `update_message()` 持久化
---

#### `chat/chat.py`
- **功能**：主聊天接口，处理用户输入，调用 LLM，返回流式响应
- **关键技术**：
  - FastAPI 接口定义
  - `EventSourceResponse` 实现 SSE 流式输出
  - 使用 `AsyncIteratorCallbackHandler` 支持逐字生成
  - 支持 `conversation_id` 管理多轮对话
---

#### `chat/utils.py`
- **功能**：辅助函数，如历史消息处理、提示词构造
---

#### `db/models/`
- **功能**：使用 SQLAlchemy 定义 ORM 模型
  - `conversation_model.py` → `Conversation` 表
  - `message_model.py` → `Message` 表
---

#### `db/repository/`
- **功能**：数据访问层（DAO），实现 Repository 模式
  - `conversation_repository.py`：增删改查 conversation
  - `message_repository.py`：增删改查 message
---

####  `db/base.py`
- **功能**：定义 Base 类，用于所有模型继承
---

####  `db/session.py`
- **功能**：数据库会话管理
  - `session_scope()`：上下文管理器，自动 commit/rollback/close
  - `@with_session`：装饰器，自动注入 session
---

####  `knowledge_base/migrate.py`
- **功能**：数据库迁移脚本，用于创建表结构

---

####  `memory/conversation_db_buffer_memory.py`
- **功能**：基于数据库的对话记忆模块
- **关键技术**：
  - 继承 LangChain 的 `ConversationBufferMemory`
  - 从数据库加载历史消息

---

####  `reranker/reranker.py`
- **功能**：对检索结果进行重排序，提升相关性

---

####  `static/`
- **用途**：静态资源（CSS/JS/图片），预留

---

####  `server/api.py`
- **功能**：API 路由入口，聚合所有接口
---

####  `server/utils.py`
- **功能**：通用工具函数
  - `get_ChatOpenAI()`：获取 LLM 实例
  - `get_prompt_template()`：加载 prompt 模板

---
####  `webui_pages/`
- **功能**：WebUI 页面（Gradio 或 Streamlit）
---