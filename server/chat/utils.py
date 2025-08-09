from pydantic import BaseModel, Field
'''
	•	BaseModel: Pydantic 的核心类，用于定义数据模型。它提供了强大的数据验证和自动类型转换功能。
	•	Field: 用于为 BaseModel 的字段定义元数据，例如默认值、字段描述、验证规则等。
'''
from langchain.prompts.chat import ChatMessagePromptTemplate
from configs import logger, log_verbose
from typing import List, Tuple, Dict, Union

class History(BaseModel):
    role :str = Field(...)
    content:str = Field(...)

    def to_msg_tuple(self):
        return "ai" if self.role=="assistant" else "human" ,self.content

    def to_msg_template(self,is_raw=True) -> ChatMessagePromptTemplate:
        #将消息对象转换为一个 ChatMessagePromptTemplate 实例，用于构建聊天消息的模板。它可能是一个自定义类中的方法，处理聊天消息时进行格式化或转换，以支持更复杂的模板化需求，比如在构建基于 Jinja2 模板的消息时。

        role_maps = {
            "ai": "assistant",
            "human": "user",
        }
        role = role_maps.get(self.role, self.role)
        '''
        	•	使用 self.role 获取当前对象的角色值：
            •	如果 self.role 是 "ai" 或 "human"，使用对应的映射值。
            •	如果没有映射值，则直接使用原始的 self.role。
        '''
        if is_raw:
            content = "{% raw %}" + self.content + "{% endraw %}"
        else:
            content = self.content  
        
        #{% raw %}Hello, {{ user_name }}!{% endraw %} ===结果：Hello, {{ user_name }}!（模板变量 user_name 不会被解析）。
        return ChatMessagePromptTemplate.from_template(
            content,
            "jinja2",
            role=role,
        )
    @classmethod
    def from_data(cls, h: Union[List, Tuple, Dict]) -> "History":
        if isinstance(h, (list,tuple)) and len(h) >= 2:
            h = cls(role=h[0], content=h[1])
        elif isinstance(h, dict):
            h = cls(**h)

        return h
# history = [History(role='user', content='我们来玩成语接龙，我先来，生龙活虎'), History(role='assistant', content='虎头虎脑')]
# history = [History.from_data(h) for h in history]
#
# print(history)