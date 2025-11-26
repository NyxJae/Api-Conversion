"""
渠道选择器
负责根据模型名和权重选择合适的渠道
"""
import random
import time
from typing import Dict, List, Optional, Any
from threading import Lock

from src.utils.logger import setup_logger
from src.utils.exceptions import ChannelError
from src.channels.channel_manager import ChannelInfo, channel_manager

logger = setup_logger("channel_selector")


class ChannelSelector:
    """渠道选择器，实现智能负载均衡"""
    
    def __init__(self, cache_ttl: int = 300):
        """
        初始化渠道选择器
        
        Args:
            cache_ttl: 缓存生存时间（秒），默认5分钟
        """
        self.cache_ttl = cache_ttl
        self._model_cache: Dict[str, List[ChannelInfo]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._lock = Lock()
        
        logger.info(f"ChannelSelector initialized with cache TTL: {cache_ttl}s")
    
    def select_channel_by_model(self, model_name: str) -> Optional[ChannelInfo]:
        """
        根据模型名选择渠道
        
        Args:
            model_name: 模型名称
            
        Returns:
            选中的渠道信息，如果没有可用渠道则返回None
            
        Raises:
            ChannelError: 渠道选择过程中发生错误
        """
        try:
            # 获取支持该模型的所有渠道
            available_channels = self.get_available_channels_for_model(model_name)
            
            if not available_channels:
                logger.warning(f"No available channels found for model: {model_name}")
                return None
            
            # 按权重随机选择渠道
            selected_channel = self._select_by_weight(available_channels)
            
            logger.info(f"Selected channel '{selected_channel.name}' for model '{model_name}' "
                       f"(weight: {selected_channel.weight})")
            
            return selected_channel
            
        except Exception as e:
            logger.error(f"Error selecting channel for model '{model_name}': {e}")
            raise ChannelError(f"Failed to select channel for model '{model_name}': {e}")
    
    def get_available_channels_for_model(self, model_name: str) -> List[ChannelInfo]:
        """
        获取支持指定模型的所有渠道
        
        Args:
            model_name: 模型名称
            
        Returns:
            支持该模型的渠道列表
        """
        try:
            # 检查缓存
            cached_channels = self._get_from_cache(model_name)
            if cached_channels is not None:
                logger.debug(f"Using cached channels for model: {model_name}")
                return cached_channels
            
            # 从数据库获取所有启用的渠道
            all_enabled_channels = channel_manager.get_enabled_channels()
            
            # 筛选支持该模型的渠道
            matching_channels = []
            for channel in all_enabled_channels:
                if self._channel_supports_model(channel, model_name):
                    matching_channels.append(channel)
            
            # 更新缓存
            self._update_cache(model_name, matching_channels)
            
            logger.debug(f"Found {len(matching_channels)} channels for model '{model_name}'")
            
            return matching_channels
            
        except Exception as e:
            logger.error(f"Error getting available channels for model '{model_name}': {e}")
            return []
    
    def _channel_supports_model(self, channel: ChannelInfo, model_name: str) -> bool:
        """
        检查渠道是否支持指定模型
        
        Args:
            channel: 渠道信息
            model_name: 模型名称
            
        Returns:
            是否支持该模型
        """
        try:
            # 检查模型映射中是否包含该模型
            if model_name in channel.models_mapping:
                return True
            
            # 检查是否有通配符匹配
            for mapped_model in channel.models_mapping.keys():
                if self._is_pattern_match(mapped_model, model_name):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking model support for channel '{channel.name}': {e}")
            return False
    
    def _is_pattern_match(self, pattern: str, model_name: str) -> bool:
        """
        检查模型名是否匹配模式
        
        Args:
            pattern: 模式（支持通配符）
            model_name: 模型名称
            
        Returns:
            是否匹配
        """
        # 简单的通配符匹配实现
        if '*' in pattern:
            # 将 * 转换为正则表达式
            import re
            regex_pattern = pattern.replace('*', '.*')
            return re.match(f'^{regex_pattern}$', model_name) is not None
        else:
            return pattern == model_name
    
    def _select_by_weight(self, channels: List[ChannelInfo]) -> ChannelInfo:
        """
        根据权重随机选择渠道
        
        Args:
            channels: 渠道列表
            
        Returns:
            选中的渠道
            
        Raises:
            ValueError: 渠道列表为空
        """
        if not channels:
            raise ValueError("Channel list is empty")
        
        # 计算总权重
        total_weight = sum(channel.weight for channel in channels)
        
        if total_weight <= 0:
            # 如果所有权重都为0，则随机选择
            logger.warning("All channels have zero weight, selecting randomly")
            return random.choice(channels)
        
        # 生成随机数
        random_value = random.uniform(0, total_weight)
        
        # 按权重选择
        current_weight = 0
        for channel in channels:
            current_weight += channel.weight
            if random_value <= current_weight:
                return channel
        
        # 由于浮点数精度问题，可能没有选中任何渠道，返回最后一个
        return channels[-1]
    
    def _get_from_cache(self, model_name: str) -> Optional[List[ChannelInfo]]:
        """
        从缓存获取渠道列表
        
        Args:
            model_name: 模型名称
            
        Returns:
            缓存的渠道列表，如果缓存过期或不存在则返回None
        """
        with self._lock:
            if model_name not in self._model_cache:
                return None
            
            # 检查缓存是否过期
            cache_time = self._cache_timestamps.get(model_name, 0)
            if time.time() - cache_time > self.cache_ttl:
                # 缓存过期，清除
                del self._model_cache[model_name]
                del self._cache_timestamps[model_name]
                logger.debug(f"Cache expired for model: {model_name}")
                return None
            
            return self._model_cache[model_name].copy()
    
    def _update_cache(self, model_name: str, channels: List[ChannelInfo]) -> None:
        """
        更新缓存
        
        Args:
            model_name: 模型名称
            channels: 渠道列表
        """
        with self._lock:
            self._model_cache[model_name] = channels.copy()
            self._cache_timestamps[model_name] = time.time()
            logger.debug(f"Updated cache for model: {model_name}")
    
    def update_cache(self) -> None:
        """
        手动更新所有缓存
        """
        with self._lock:
            # 清除所有缓存
            self._model_cache.clear()
            self._cache_timestamps.clear()
            logger.info("All cache cleared")
    
    def clear_cache_for_model(self, model_name: str) -> None:
        """
        清除指定模型的缓存
        
        Args:
            model_name: 模型名称
        """
        with self._lock:
            if model_name in self._model_cache:
                del self._model_cache[model_name]
            if model_name in self._cache_timestamps:
                del self._cache_timestamps[model_name]
            logger.debug(f"Cache cleared for model: {model_name}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息
        """
        with self._lock:
            current_time = time.time()
            cache_info = {}
            
            for model_name, cache_time in self._cache_timestamps.items():
                age = current_time - cache_time
                cache_info[model_name] = {
                    'channel_count': len(self._model_cache.get(model_name, [])),
                    'age_seconds': age,
                    'is_expired': age > self.cache_ttl
                }
            
            return {
                'total_cached_models': len(self._model_cache),
                'cache_ttl': self.cache_ttl,
                'cache_details': cache_info
            }
    
    def get_selection_statistics(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型选择统计信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            选择统计信息
        """
        try:
            available_channels = self.get_available_channels_for_model(model_name)
            
            if not available_channels:
                return {
                    'model_name': model_name,
                    'available_channels': 0,
                    'total_weight': 0,
                    'channels': []
                }
            
            total_weight = sum(channel.weight for channel in available_channels)
            
            channel_stats = []
            for channel in available_channels:
                weight_percentage = (channel.weight / total_weight * 100) if total_weight > 0 else 0
                channel_stats.append({
                    'channel_id': channel.id,
                    'channel_name': channel.name,
                    'provider': channel.provider,
                    'weight': channel.weight,
                    'weight_percentage': round(weight_percentage, 2),
                    'supported_models': list(channel.models_mapping.keys())
                })
            
            return {
                'model_name': model_name,
                'available_channels': len(available_channels),
                'total_weight': total_weight,
                'channels': channel_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting selection statistics for model '{model_name}': {e}")
            return {
                'model_name': model_name,
                'error': str(e)
            }
    
    def validate_channel_health(self, channel: ChannelInfo) -> bool:
        """
        验证渠道健康状态
        
        Args:
            channel: 渠道信息
            
        Returns:
            渠道是否健康
        """
        try:
            # 基本检查
            if not channel.enabled:
                logger.debug(f"Channel '{channel.name}' is disabled")
                return False
            
            if not channel.api_key:
                logger.warning(f"Channel '{channel.name}' has no API key")
                return False
            
            if not channel.models_mapping:
                logger.warning(f"Channel '{channel.name}' has no models mapping")
                return False
            
            # 这里可以添加更多的健康检查逻辑
            # 例如：测试连接、检查配额等
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating channel health for '{channel.name}': {e}")
            return False


# 全局渠道选择器实例
channel_selector = ChannelSelector()