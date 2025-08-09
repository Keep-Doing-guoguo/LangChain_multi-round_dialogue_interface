#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/18 10:29
@source from: 
"""

import sqlite3
from typing import List, Tuple

class SQLiteViewer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def get_tables(self) -> List[str]:
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in self.cursor.fetchall()]

    def get_table_columns(self, table_name: str) -> List[Tuple]:
        self.cursor.execute(f"PRAGMA table_info({table_name});")
        return self.cursor.fetchall()

    def get_table_preview(self, table_name: str, limit: int = 10) -> List[Tuple]:
        self.cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit};")
        return self.cursor.fetchall()

    def delete_all_from_table(self, table_name: str):
        self.cursor.execute(f"DELETE FROM {table_name};")
        self.conn.commit()
        print(f"✅ 已清空表：{table_name}")

    def close(self):
        self.conn.close()



def main():
    db_path = '/Volumes/PSSD/未命名文件夹/donwload/Bymyself1/knowledge_base/info.db'  # 👈 替换为你的 .db 文件路径
    viewer = SQLiteViewer(db_path)

    print("📋 所有表：")
    tables = viewer.get_tables()
    print(tables)
    #viewer.delete_all_from_table("message")
    for table in tables:
        print(f"\n📌 表结构（{table}）：")
        columns = viewer.get_table_columns(table)
        for col in columns:
            print(col)

        print(f"\n🔍 数据预览（{table}）：")
        preview = viewer.get_table_preview(table)
        for row in preview:
            print(row)

    viewer.close()

if __name__ == '__main__':
    main()