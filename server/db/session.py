#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/1/2 16:02
@source from: 
"""
from functools import wraps
from contextlib import contextmanager
from server.db.base import SessionLocal#其实这里写到这个session.py里面也行
from sqlalchemy.orm import Session
@contextmanager
def session_scope() -> Session:
    """上下文管理器用于自动获取 Session, 避免错误"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()
# 装饰器
def with_session(f):
    @wraps(f)  # 保留原始函数的元数据
    def wrapper(*args, **kwargs):
        print(f"函数 {f.__name__} 被调用")
        print(f"位置参数: {args}")
        print(f"关键字参数: {kwargs}")

        with session_scope() as session:
            try:
                result = f(session, *args, **kwargs)
                session.commit()
                print(f"函数 {f.__name__} 返回值: {result}")
                return result
            except Exception as e:
                session.rollback()
                print(f"函数 {f.__name__} 出现异常: {e}")
                raise

    return wrapper