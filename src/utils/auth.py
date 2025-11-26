"""
认证和授权管理
提供会话认证和管理员API Key认证功能
"""
import hashlib
import secrets
import os
from typing import Optional
from datetime import datetime, timedelta

from fastapi import HTTPException, Header, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.utils.database import db_manager
from src.utils.logger import setup_logger
from src.utils.env_config import env_config

logger = setup_logger("auth")


class AuthManager:
    """认证管理器"""
    
    def __init__(self):
        # 固定会话超时时间为1天
        self.session_timeout = timedelta(days=1)
        self._ensure_admin_password()
    
    def _ensure_admin_password(self):
        """确保管理员密码已设置"""
        # 环境变量优先：先读取环境变量中的密码
        env_password = env_config.admin_password
        stored_password_hash = db_manager.get_config("admin_password_hash")
        
        if not stored_password_hash:
            # 数据库中没有密码，将环境变量密码存入数据库（启动时不清除会话）
            self.set_admin_password(env_password, invalidate_sessions=False)
            password_prefix = env_password[:3] + "***" if len(env_password) >= 3 else "***"
            logger.info(f"Admin password initialized from environment config (prefix: {password_prefix})")
        else:
            # 数据库中有密码，检查是否与环境变量密码一致
            if self.verify_password(env_password, stored_password_hash):
                # 环境变量密码与数据库密码一致，不做任何操作
                logger.info("Environment password matches stored password")
            else:
                # 环境变量密码与数据库密码不一致，用环境变量密码更新数据库
                # 启动时如果密码不一致，也要清除会话（可能是环境配置被修改）
                self.set_admin_password(env_password, invalidate_sessions=True)
                password_prefix = env_password[:3] + "***" if len(env_password) >= 3 else "***"
                logger.info(f"Admin password updated from environment config (prefix: {password_prefix})")
    
    def hash_password(self, password: str) -> str:
        """对密码进行哈希"""
        # 使用随机盐值
        salt = secrets.token_hex(32)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """验证密码"""
        try:
            salt, stored_hash = password_hash.split(':')
            password_hash_check = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return stored_hash == password_hash_check.hex()
        except Exception:
            return False
    
    def set_admin_password(self, password: str, invalidate_sessions: bool = True):
        """设置管理员密码"""
        password_hash = self.hash_password(password)
        db_manager.set_config("admin_password_hash", password_hash)
        
        # 密码修改后，为安全起见，使所有现有会话失效
        if invalidate_sessions:
            invalidated_count = self.invalidate_all_sessions()
            logger.info(f"Admin password updated, invalidated {invalidated_count} sessions")
        else:
            logger.info("Admin password updated")
    
    def verify_admin_password(self, password: str) -> bool:
        """验证管理员密码"""
        stored_hash = db_manager.get_config("admin_password_hash")
        if not stored_hash:
            return False
        return self.verify_password(password, stored_hash)
    
    def generate_session_token(self) -> str:
        """生成会话令牌"""
        return secrets.token_urlsafe(32)
    
    def create_session(self, password: str) -> Optional[str]:
        """创建会话"""
        if not self.verify_admin_password(password):
            return None
        
        session_token = self.generate_session_token()
        expires_at = (datetime.now() + self.session_timeout).isoformat()
        
        # 存储会话信息
        db_manager.set_config(f"session:{session_token}", expires_at)
        
        logger.info("New admin session created")
        return session_token
    
    def verify_session(self, session_token: str) -> bool:
        """验证会话"""
        if not session_token:
            return False
        
        expires_at_str = db_manager.get_config(f"session:{session_token}")
        if not expires_at_str:
            return False
        
        try:
            expires_at = datetime.fromisoformat(expires_at_str)
            if datetime.now() > expires_at:
                # 会话已过期，删除
                self.delete_session(session_token)
                return False
            return True
        except Exception:
            return False
    
    def delete_session(self, session_token: str):
        """删除会话"""
        if not session_token:
            return
        
        session_key = f"session:{session_token}"
        deleted = db_manager.delete_config(session_key)
        
        from src.utils.security import mask_api_key
        if deleted:
            logger.info(f"Session {mask_api_key(session_token)} deleted successfully")
        else:
            logger.warning(f"Failed to delete session {mask_api_key(session_token)} - not found")
    
    def invalidate_all_sessions(self):
        """使所有会话失效 - 用于密码修改后的安全措施"""
        try:
            # 获取所有session配置
            session_configs = db_manager.get_configs_by_prefix("session:")
            deleted_count = 0
            
            for config in session_configs:
                session_key = config["key"]
                if db_manager.delete_config(session_key):
                    deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"Invalidated {deleted_count} sessions due to password change")
            else:
                logger.info("No active sessions to invalidate")
                
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to invalidate sessions: {e}")
            return 0
    
    def cleanup_expired_sessions(self):
        """清理过期会话"""
        try:
            # 获取所有session配置
            session_configs = db_manager.get_configs_by_prefix("session:")
            current_time = datetime.now()
            cleaned_count = 0
            
            for config in session_configs:
                session_key = config["key"]
                expires_at_str = config["value"]
                
                try:
                    expires_at = datetime.fromisoformat(expires_at_str)
                    if current_time > expires_at:
                        # 会话已过期，删除
                        if db_manager.delete_config(session_key):
                            cleaned_count += 1
                            session_token = session_key.replace("session:", "")
                            from src.utils.security import mask_api_key
                            logger.debug(f"Cleaned expired session: {mask_api_key(session_token)}")
                except Exception as e:
                    # 无效的时间格式，删除这个配置
                    logger.warning(f"Invalid session expiry format for {session_key}: {e}")
                    if db_manager.delete_config(session_key):
                        cleaned_count += 1
            
            logger.info(f"Session cleanup completed. Removed {cleaned_count} expired sessions")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
            return 0


# 管理员API Key配置
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
if not ADMIN_API_KEY:
    logger.warning("ADMIN_API_KEY environment variable not set. Admin authentication will be disabled.")
    ADMIN_API_KEY = "admin-default-key-change-in-production"  # 默认密钥，仅用于开发

# HTTP Bearer认证
security = HTTPBearer(auto_error=False)


def get_admin_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_admin_api_key: Optional[str] = Header(None, alias="x-admin-api-key")
) -> str:
    """
    获取管理员API Key，支持多种认证方式：
    1. Authorization Bearer header
    2. x-admin-api-key header
    """
    # 优先使用 x-admin-api-key header
    if x_admin_api_key:
        logger.debug("Admin auth via x-admin-api-key header")
        api_key = x_admin_api_key
    # 然后尝试 Authorization Bearer header
    elif credentials and credentials.scheme == "Bearer":
        logger.debug("Admin auth via Authorization Bearer header")
        api_key = credentials.credentials
    else:
        logger.error("Missing admin authentication")
        raise HTTPException(
            status_code=401,
            detail="Missing admin API key. Provide it via 'x-admin-api-key' header or 'Authorization: Bearer <key>' header."
        )
    
    # 从数据库获取当前设置的管理员API Key
    stored_admin_key = db_manager.get_config("admin_api_key")
    
    # 如果数据库中没有设置，使用环境变量或默认值
    if not stored_admin_key:
        stored_admin_key = ADMIN_API_KEY
        logger.warning("No admin API key set in database, using fallback")
    
    # 验证API Key
    if api_key != stored_admin_key:
        logger.error("Invalid admin API key provided")
        raise HTTPException(
            status_code=403,
            detail="Invalid admin API key"
        )
    
    logger.debug("Admin authentication successful")
    return api_key


def get_user_api_key(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    authorization_gemini: Optional[str] = Header(None, alias="authorization")
) -> str:
    """
    获取用户API Key，支持多种API格式：
    - OpenAI格式: Authorization: Bearer <key>
    - Anthropic格式: x-api-key: <key>
    - Gemini格式: Authorization: Bearer <key> 或 x-goog-api-key: <key>
    """
    # OpenAI格式
    if authorization and authorization.startswith("Bearer "):
        logger.debug("User auth via OpenAI Bearer format")
        return authorization[7:]
    
    # Anthropic格式
    if x_api_key:
        logger.debug("User auth via Anthropic x-api-key format")
        return x_api_key
    
    # Gemini格式（也使用Bearer，但可能在不同的header中）
    if authorization_gemini and authorization_gemini.startswith("Bearer "):
        logger.debug("User auth via Gemini Bearer format")
        return authorization_gemini[7:]
    
    logger.error("Missing user API key")
    raise HTTPException(
        status_code=401,
        detail="Missing API key. Please provide your API key using the appropriate format for your AI service provider."
    )


def mask_api_key(api_key: str) -> str:
    """遮蔽API Key用于日志记录"""
    if not api_key:
        return "None"
    
    if len(api_key) <= 8:
        return "***"
    
    return f"{api_key[:4]}...{api_key[-4:]}"


# 认证依赖项
AdminAuth = Depends(get_admin_api_key)
UserAuth = Depends(get_user_api_key)


# 全局认证管理器实例
auth_manager = AuthManager()
