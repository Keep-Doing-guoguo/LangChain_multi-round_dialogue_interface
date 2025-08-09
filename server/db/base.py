#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/1/2 16:01
@source from: 
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy.orm import sessionmaker

from configs import SQLALCHEMY_DATABASE_URI
import json


engine = create_engine(
    SQLALCHEMY_DATABASE_URI,
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base: DeclarativeMeta = declarative_base()

'''
	•	生成了一个基类 Base。
	•	这个基类 Base 用来创建所有的 ORM 模型。
	•	类型注解 表明它是一个使用 DeclarativeMeta 元类的类。
'''
