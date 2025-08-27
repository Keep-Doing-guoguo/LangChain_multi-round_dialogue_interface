
from server.db.models.conversation_model import ConversationModel
from server.db.models.message_model import MessageModel

from server.db.base import Base, engine
from server.db.session import session_scope
import os
from dateutil.parser import parse
from typing import Literal, List


def create_tables():
    Base.metadata.create_all(bind=engine)


def reset_tables():
    Base.metadata.drop_all(bind=engine)
    create_tables()

reset_tables()
