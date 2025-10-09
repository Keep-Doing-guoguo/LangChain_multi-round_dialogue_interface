import logging
from typing import Any, List, Dict

from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import get_buffer_string, BaseMessage, HumanMessage, AIMessage
from langchain.schema.language_model import BaseLanguageModel
from server.db.repository.message_repository import filter_message
from typing import Any, List, Callable
from langchain.chat_models import ChatOpenAI
import os
os.environ["OPENAI_API_KEY"] = ""

# 注意测试的时候，需要将filter_message函数打开，将上面的filter_message导入进行关闭。
# def filter_message(conversation_id: str, limit: int):
#     """模拟从数据库读取最近 N 条对话"""
#     print(f"[DB] 读取 conversation_id={conversation_id} 的最近 {limit} 条消息...")
#     return [
#         {"query": "你好", "response": "你好呀！"},
#         {"query": "今天天气怎么样", "response": "今天晴，气温25度。"},
#         {"query": "帮我写一首诗", "response": "春风又绿江南岸，明月何时照我还。"}
#     ]
class ConversationBufferDBMemory(BaseChatMemory):
    conversation_id: str
    human_prefix: str = "Human"
    ai_prefix: str = "Assistant"
    llm: BaseLanguageModel
    memory_key: str = "history"
    max_token_limit: int = 2000
    message_limit: int = 10

    @property#@property 是 Python 的一个装饰器，表示将一个类的方法定义为一个只读属性。使用 @property 后，可以像访问普通属性一样调用方法，而无需加括号。
    def buffer(self) -> List[BaseMessage]:
        """String buffer of memory."""
        # fetch limited messages desc, and return reversed

        messages = filter_message(conversation_id=self.conversation_id, limit=self.message_limit)
        # 返回的记录按时间倒序，转为正序
        messages = list(reversed(messages))
        chat_messages: List[BaseMessage] = []
        for message in messages:
            chat_messages.append(HumanMessage(content=message["query"]))
            chat_messages.append(AIMessage(content=message["response"]))

        if not chat_messages:
            return []

        # prune the chat message if it exceeds the max token limit
        curr_buffer_length = self.llm.get_num_tokens(get_buffer_string(chat_messages))
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit and chat_messages:
                pruned_memory.append(chat_messages.pop(0))
                curr_buffer_length = self.llm.get_num_tokens(get_buffer_string(chat_messages))

        return chat_messages

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        buffer: Any = self.buffer
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: final_buffer}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Nothing should be saved or changed"""
        pass

    def clear(self) -> None:
        """Nothing to clear, got a memory like a vault."""
        pass

#####测试部分#####
from typing import Any, List
from langchain.schema import get_buffer_string, BaseMessage, HumanMessage, AIMessage
# ====== 封装模型加载函数 ======
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
        openai_api_key='sk-',
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus",
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
    return model


# ====== main 函数 ======
def main():
    model = get_ChatOpenAI(
        model_name="qwen-plus",
        temperature=0.7,
        max_tokens=512,
        callbacks=[],
    )

    memory = ConversationBufferDBMemory(conversation_id="test_001", llm=model)
    messages = memory.buffer

    print("\n=== 对话历史 ===")
    for msg in messages:
        role = "🧑 Human" if isinstance(msg, HumanMessage) else "🤖 Assistant"
        print(f"{role}: {msg.content}")

    print("\n=== 拼接为 prompt ===")
    print(get_buffer_string(messages))


if __name__ == "__main__":
    main()