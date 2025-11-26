"""
数据库管理器
支持SQLite和MySQL数据库，用于存储渠道信息和系统配置
"""
import sqlite3
import pymysql
import os
import json
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import contextmanager

from src.utils.logger import setup_logger
from src.utils.env_config import env_config
from src.utils.encryption import encryption_manager

logger = setup_logger("database")


class DatabaseManager:
    """数据库管理器，支持SQLite和MySQL"""
    
    def __init__(self, db_path: str = None):
        self.db_type = env_config.database_type
        self.db_path = db_path or env_config.database_path
        self._initialized = False
        
        if self.db_type == "sqlite":
            self._ensure_data_dir()
        
        # 立即验证数据库连接，不再使用懒加载
        self._ensure_initialized()
    
    def _ensure_initialized(self):
        """确保数据库已初始化"""
        if not self._initialized:
            try:
                self._init_database()
                self._initialized = True
            except Exception as e:
                logger.error(f"Database initialization failed: {e}")
                # 直接抛出异常，不再自动回退到SQLite
                raise RuntimeError(f"Failed to initialize {self.db_type} database: {e}")
    
    def _ensure_data_dir(self):
        """确保SQLite数据目录存在"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
    def _get_raw_connection(self):
        """获取原始数据库连接"""
        if self.db_type == "sqlite":
            return sqlite3.connect(self.db_path)
        elif self.db_type == "mysql":
            return pymysql.connect(
                host=env_config.mysql_host,
                port=env_config.mysql_port,
                user=env_config.mysql_user,
                password=env_config.mysql_password,
                database=env_config.mysql_database,
                charset='utf8mb4',
                autocommit=False
            )
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = self._get_raw_connection()
        try:
            yield conn
        finally:
            conn.close()
    
    def _execute_query(self, conn, query: str, params: tuple = None):
        """执行SQL查询"""
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        return cursor
    
    def _init_database(self):
        """初始化数据库表"""
        conn = self._get_raw_connection()
        try:
            cursor = conn.cursor()
            
            if self.db_type == "sqlite":
                # SQLite表结构
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS channels (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        base_url TEXT NOT NULL,
                        api_key TEXT NOT NULL,
                        timeout INTEGER DEFAULT 30,
                        max_retries INTEGER DEFAULT 3,
                        enabled BOOLEAN DEFAULT 1,
                        models_mapping TEXT NOT NULL,
                        use_proxy BOOLEAN DEFAULT 0,
                        proxy_type TEXT,
                        proxy_host TEXT,
                        proxy_port INTEGER,
                        proxy_username TEXT,
                        proxy_password TEXT,
                        weight INTEGER DEFAULT 1,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_config (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                
            elif self.db_type == "mysql":
                # MySQL表结构
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS channels (
                        id VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        provider VARCHAR(100) NOT NULL,
                        base_url TEXT NOT NULL,
                        api_key TEXT NOT NULL,
                        timeout INT DEFAULT 30,
                        max_retries INT DEFAULT 3,
                        enabled TINYINT(1) DEFAULT 1,
                        models_mapping TEXT NOT NULL,
                        use_proxy TINYINT(1) DEFAULT 0,
                        proxy_type VARCHAR(20),
                        proxy_host VARCHAR(255),
                        proxy_port INT,
                        proxy_username VARCHAR(255),
                        proxy_password TEXT,
                        weight INT DEFAULT 1,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_config (
                        `key` VARCHAR(255) PRIMARY KEY,
                        `value` TEXT NOT NULL,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)
            
            conn.commit()
            
            # 进行数据库迁移 - 添加代理字段（如果不存在）
            self._migrate_proxy_fields(cursor, conn)
            
            # 进行数据库迁移 - 添加权重字段（如果不存在）
            self._migrate_weight_field(cursor, conn)
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
        finally:
            conn.close()
    
    def _migrate_proxy_fields(self, cursor, conn):
        """迁移代理字段（如果不存在）"""
        try:
            # 检查是否已存在代理字段
            cursor.execute("PRAGMA table_info(channels)")
            columns = [row[1] for row in cursor.fetchall()]
            
            proxy_fields = ['use_proxy', 'proxy_type', 'proxy_host', 'proxy_port', 'proxy_username', 'proxy_password']
            for field in proxy_fields:
                if field not in columns:
                    if self.db_type == "sqlite":
                        cursor.execute(f"ALTER TABLE channels ADD COLUMN {field} {'TEXT' if field in ['proxy_type', 'proxy_host', 'proxy_username'] else ('INTEGER' if field == 'proxy_port' else 'BOOLEAN')}")
                    elif self.db_type == "mysql":
                        field_type = {
                            'use_proxy': 'TINYINT(1)',
                            'proxy_type': 'VARCHAR(20)',
                            'proxy_host': 'VARCHAR(255)',
                            'proxy_port': 'INT',
                            'proxy_username': 'VARCHAR(255)',
                            'proxy_password': 'TEXT'
                        }[field]
                        cursor.execute(f"ALTER TABLE channels ADD COLUMN {field} {field_type}")
                    logger.info(f"Added {field} column to channels table")
            
            conn.commit()
        except Exception as e:
            logger.warning(f"Migration warning (proxy fields may already exist): {e}")
    
    def _migrate_weight_field(self, cursor, conn):
        """迁移权重字段（如果不存在）"""
        try:
            # 检查是否已存在权重字段
            cursor.execute("PRAGMA table_info(channels)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'weight' not in columns:
                cursor.execute("ALTER TABLE channels ADD COLUMN weight INT DEFAULT 1")
                logger.info("Added weight column to channels table")
            
            conn.commit()
        except Exception as e:
            logger.warning(f"Migration warning (weight field may already exist): {e}")
    
    def add_channel(
        self,
        name: str,
        provider: str,
        base_url: str,
        api_key: str,
        timeout: int = 30,
        max_retries: int = 3,
        models_mapping: Optional[Dict[str, str]] = None,
        use_proxy: bool = False,
        proxy_type: Optional[str] = None,
        proxy_host: Optional[str] = None,
        proxy_port: Optional[int] = None,
        proxy_username: Optional[str] = None,
        proxy_password: Optional[str] = None,
        weight: int = 1
    ) -> str:
        """添加新渠道"""
        channel_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        # 验证models_mapping必填
        if not models_mapping:
            raise ValueError("models_mapping is required")
        
        models_mapping_json = json.dumps(models_mapping)
        
        # 验证API密钥不是明显的JavaScript错误信息
        if api_key.startswith('script.js:') or 'Uncaught TypeError' in api_key:
            logger.error(f"Rejecting JavaScript error message as API key: {api_key[:50]}...")
            raise ValueError("Invalid API key: JavaScript error message detected")
        
        # 加密API密钥
        encrypted_api_key = encryption_manager.encrypt_api_key(api_key)
        
        # 加密代理密码（如果存在）
        encrypted_proxy_password = None
        if proxy_password:
            encrypted_proxy_password = encryption_manager.encrypt_api_key(proxy_password)
        
        with self.get_connection() as conn:
            try:
                cursor = self._execute_query(conn, """
                    INSERT INTO channels 
                    (id, name, provider, base_url, api_key, timeout, max_retries, 
                     enabled, models_mapping, use_proxy, proxy_type, proxy_host, proxy_port, 
                     proxy_username, proxy_password, weight, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    channel_id, name, provider, base_url, encrypted_api_key,
                    timeout, max_retries, True, models_mapping_json, use_proxy, proxy_type,
                    proxy_host, proxy_port, proxy_username, encrypted_proxy_password, weight, now, now
                ))
                
                conn.commit()
                logger.info(f"Added new channel: {name} ({provider}) with ID: {channel_id}")
                return channel_id
            except (sqlite3.IntegrityError, pymysql.IntegrityError) as e:
                raise ValueError(f"Database integrity error: {e}")
    
    def update_channel(
        self,
        channel_id: str,
        name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        enabled: Optional[bool] = None,
        models_mapping: Optional[Dict[str, str]] = None,
        use_proxy: Optional[bool] = None,
        proxy_type: Optional[str] = None,
        proxy_host: Optional[str] = None,
        proxy_port: Optional[int] = None,
        proxy_username: Optional[str] = None,
        proxy_password: Optional[str] = None,
        weight: Optional[int] = None
    ) -> bool:
        """更新渠道信息"""
        updates = []
        params = []
        
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if base_url is not None:
            updates.append("base_url = ?")
            params.append(base_url)
        if api_key is not None:
            # 验证API密钥不是明显的JavaScript错误信息
            if api_key.startswith('script.js:') or 'Uncaught TypeError' in api_key:
                logger.error(f"Rejecting JavaScript error message as API key: {api_key[:50]}...")
                raise ValueError("Invalid API key: JavaScript error message detected")
            
            updates.append("api_key = ?")
            # 加密API密钥
            encrypted_api_key = encryption_manager.encrypt_api_key(api_key)
            params.append(encrypted_api_key)
        if timeout is not None:
            updates.append("timeout = ?")
            params.append(timeout)
        if max_retries is not None:
            updates.append("max_retries = ?")
            params.append(max_retries)
        if enabled is not None:
            updates.append("enabled = ?")
            params.append(enabled)
        if models_mapping is not None:
            updates.append("models_mapping = ?")
            params.append(json.dumps(models_mapping))
        if use_proxy is not None:
            updates.append("use_proxy = ?")
            params.append(use_proxy)
        if proxy_type is not None:
            updates.append("proxy_type = ?")
            params.append(proxy_type)
        if proxy_host is not None:
            updates.append("proxy_host = ?")
            params.append(proxy_host)
        if proxy_port is not None:
            updates.append("proxy_port = ?")
            params.append(proxy_port)
        if proxy_username is not None:
            updates.append("proxy_username = ?")
            params.append(proxy_username)
        if proxy_password is not None:
            updates.append("proxy_password = ?")
            # 加密代理密码
            encrypted_proxy_password = encryption_manager.encrypt_api_key(proxy_password)
            params.append(encrypted_proxy_password)
        if weight is not None:
            updates.append("weight = ?")
            params.append(weight)
        
        if not updates:
            return False
        
        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(channel_id)
        
        with self.get_connection() as conn:
            try:
                cursor = self._execute_query(conn, f"""
                    UPDATE channels 
                    SET {", ".join(updates)}
                    WHERE id = ?
                """, tuple(params))
                
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                logger.error(f"Failed to update channel: {e}")
                return False
    
    def delete_channel(self, channel_id: str) -> bool:
        """删除渠道"""
        with self.get_connection() as conn:
            try:
                cursor = self._execute_query(conn, "DELETE FROM channels WHERE id = ?", (channel_id,))
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                logger.error(f"Failed to delete channel: {e}")
                return False
    
    def get_channel(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """获取单个渠道信息"""
        with self.get_connection() as conn:
            try:
                cursor = self._execute_query(conn, "SELECT * FROM channels WHERE id = ?", (channel_id,))
                row = cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    result = dict(zip(columns, row))
                    # 解密API密钥
                    if result['api_key']:
                        result['api_key'] = encryption_manager.decrypt_api_key(result['api_key'])
                    # 解密代理密码
                    if result['proxy_password']:
                        result['proxy_password'] = encryption_manager.decrypt_api_key(result['proxy_password'])
                    # 解析models_mapping JSON字符串
                    if result['models_mapping']:
                        try:
                            result['models_mapping'] = json.loads(result['models_mapping'])
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"Failed to parse models_mapping for channel {channel_id}: {e}")
                            result['models_mapping'] = {}
                    else:
                        result['models_mapping'] = {}
                    return result
                return None
            except Exception as e:
                logger.error(f"Failed to get channel: {e}")
                return None
    
    def get_channels_by_provider(self, provider: str) -> List[Dict[str, Any]]:
        """按提供商获取渠道列表"""
        with self.get_connection() as conn:
            try:
                cursor = self._execute_query(conn, "SELECT * FROM channels WHERE provider = ?", (provider,))
                rows = cursor.fetchall()
                channels = []
                for row in rows:
                    columns = [description[0] for description in cursor.description]
                    channel = dict(zip(columns, row))
                    # 解密API密钥
                    if channel['api_key']:
                        channel['api_key'] = encryption_manager.decrypt_api_key(channel['api_key'])
                    # 解密代理密码
                    if channel['proxy_password']:
                        channel['proxy_password'] = encryption_manager.decrypt_api_key(channel['proxy_password'])
                    # 解析models_mapping JSON字符串
                    if channel['models_mapping']:
                        try:
                            channel['models_mapping'] = json.loads(channel['models_mapping'])
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"Failed to parse models_mapping for channel {channel.get('id', 'unknown')}: {e}")
                            channel['models_mapping'] = {}
                    else:
                        channel['models_mapping'] = {}
                    channels.append(channel)
                return channels
            except Exception as e:
                logger.error(f"Failed to get channels by provider: {e}")
                return []
    
    def get_all_channels(self) -> List[Dict[str, Any]]:
        """获取所有渠道"""
        with self.get_connection() as conn:
            try:
                cursor = self._execute_query(conn, "SELECT * FROM channels ORDER BY created_at DESC")
                rows = cursor.fetchall()
                channels = []
                for row in rows:
                    columns = [description[0] for description in cursor.description]
                    channel = dict(zip(columns, row))
                    # 解密API密钥
                    if channel['api_key']:
                        channel['api_key'] = encryption_manager.decrypt_api_key(channel['api_key'])
                    # 解密代理密码
                    if channel['proxy_password']:
                        channel['proxy_password'] = encryption_manager.decrypt_api_key(channel['proxy_password'])
                    # 解析models_mapping JSON字符串
                    if channel['models_mapping']:
                        try:
                            channel['models_mapping'] = json.loads(channel['models_mapping'])
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"Failed to parse models_mapping for channel {channel.get('id', 'unknown')}: {e}")
                            channel['models_mapping'] = {}
                    else:
                        channel['models_mapping'] = {}
                    channels.append(channel)
                return channels
            except Exception as e:
                logger.error(f"Failed to get all channels: {e}")
                return []
    
    def get_enabled_channels(self) -> List[Dict[str, Any]]:
        """获取所有启用的渠道"""
        with self.get_connection() as conn:
            try:
                cursor = self._execute_query(conn, "SELECT * FROM channels WHERE enabled = 1 ORDER BY created_at DESC")
                rows = cursor.fetchall()
                channels = []
                for row in rows:
                    columns = [description[0] for description in cursor.description]
                    channel = dict(zip(columns, row))
                    # 解密API密钥
                    if channel['api_key']:
                        channel['api_key'] = encryption_manager.decrypt_api_key(channel['api_key'])
                    # 解密代理密码
                    if channel['proxy_password']:
                        channel['proxy_password'] = encryption_manager.decrypt_api_key(channel['proxy_password'])
                    # 解析models_mapping JSON字符串
                    if channel['models_mapping']:
                        try:
                            channel['models_mapping'] = json.loads(channel['models_mapping'])
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"Failed to parse models_mapping for channel {channel.get('id', 'unknown')}: {e}")
                            channel['models_mapping'] = {}
                    else:
                        channel['models_mapping'] = {}
                    channels.append(channel)
                return channels
            except Exception as e:
                logger.error(f"Failed to get enabled channels: {e}")
                return []
    
    def get_config(self, key: str) -> Optional[str]:
        """获取配置"""
        with self.get_connection() as conn:
            try:
                cursor = self._execute_query(conn, "SELECT value FROM system_config WHERE key = ?", (key,))
                row = cursor.fetchone()
                return row[0] if row else None
            except Exception as e:
                logger.error(f"Failed to get config: {e}")
                return None
    
    def set_config(self, key: str, value: str):
        """设置配置"""
        with self.get_connection() as conn:
            try:
                now = datetime.now().isoformat()
                cursor = self._execute_query(conn, 
                    "INSERT OR REPLACE INTO system_config (key, value, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (key, value, now, now))
                conn.commit()
                logger.info(f"Config set: {key}")
            except Exception as e:
                logger.error(f"Failed to set config: {e}")
                raise
    
    def delete_config(self, key: str) -> bool:
        """删除配置"""
        with self.get_connection() as conn:
            try:
                cursor = self._execute_query(conn, "DELETE FROM system_config WHERE key = ?", (key,))
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                logger.error(f"Failed to delete config: {e}")
                return False
    
    def get_configs_by_prefix(self, prefix: str) -> List[Dict[str, Any]]:
        """按前缀获取配置列表"""
        with self.get_connection() as conn:
            try:
                cursor = self._execute_query(conn, "SELECT key, value, created_at, updated_at FROM system_config WHERE key LIKE ? ORDER BY key", (f"{prefix}%",))
                rows = cursor.fetchall()
                configs = []
                for row in rows:
                    columns = [description[0] for description in cursor.description]
                    config = dict(zip(columns, row))
                    configs.append(config)
                return configs
            except Exception as e:
                logger.error(f"Failed to get configs by prefix: {e}")
                return []


# 全局数据库管理器实例
def get_db_manager():
    """获取数据库管理器实例"""
    return DatabaseManager()


# 创建全局实例
db_manager = get_db_manager()