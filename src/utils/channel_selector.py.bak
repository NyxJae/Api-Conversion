"""
渠道选择器
根据模型名和权重选择合适的渠道进行负载均衡
"""
import random
from typing import List, Optional, Dict, Any
from channels.channel_manager import channel_manager, ChannelInfo
from src.utils.logger import setup_logger

logger = setup_logger("channel_selector")


class ChannelSelector:
    """渠道选择器，实现负载均衡和模型匹配"""
    
    def __init__(self):
        self.channel_manager = channel_manager
    
    def select_channel(self, model_name: str) -> Optional[ChannelInfo]:
        """
        根据模型名选择合适的渠道
        
        Args:
            model_name: 请求的模型名
            
        Returns:
            选中的渠道，如果没有找到合适的渠道则返回None
        """
        logger.info(f"Selecting channel for model: {model_name}")
        
        # 获取所有启用的渠道
        all_channels = self.channel_manager.get_all_channels()
        enabled_channels = [ch for ch in all_channels if ch.enabled]
        
        if not enabled_channels:
            logger.warning("No enabled channels available")
            return None
        
        # 找到支持该模型的渠道
        candidate_channels = []
        for channel in enabled_channels:
            if self._channel_supports_model(channel, model_name):
                candidate_channels.append(channel)
        
        if not candidate_channels:
            logger.warning(f"No channels support model: {model_name}")
            # 列出所有可用的模型映射用于调试
            available_models = set()
            for channel in enabled_channels:
                if channel.models_mapping:
                    available_models.update(channel.models_mapping.keys())
            logger.info(f"Available models in mappings: {sorted(available_models)}")
            return None
        
        # 根据权重进行加权随机选择
        selected_channel = self._weighted_random_select(candidate_channels)
        
        logger.info(f"Selected channel: {selected_channel.name} (provider: {selected_channel.provider}, weight: {getattr(selected_channel, 'weight', 1)})")
        return selected_channel
    
    def _channel_supports_model(self, channel: ChannelInfo, model_name: str) -> bool:
        """
        检查渠道是否支持指定的模型
        
        Args:
            channel: 渠道信息
            model_name: 模型名
            
        Returns:
            是否支持该模型
        """
        # 检查模型的映射配置
        if channel.models_mapping and model_name in channel.models_mapping:
            logger.debug(f"Channel {channel.name} supports model {model_name} via mapping")
            return True
        
        # 如果没有映射配置，检查是否为通用模型（如gpt-3.5-turbo等）
        # 这里可以添加更多的默认模型支持逻辑
        common_models = {
            "openai": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"],
            "anthropic": ["claude-3-sonnet", "claude-3-opus", "claude-3-haiku"],
            "gemini": ["gemini-pro", "gemini-pro-vision"]
        }
        
        if model_name in common_models.get(channel.provider, []):
            logger.debug(f"Channel {channel.name} supports common model {model_name} for provider {channel.provider}")
            return True
        
        return False
    
    def _weighted_random_select(self, channels: List[ChannelInfo]) -> ChannelInfo:
        """
        根据权重进行加权随机选择
        
        Args:
            channels: 候选渠道列表
            
        Returns:
            选中的渠道
        """
        if not channels:
            raise ValueError("No channels provided for selection")
        
        if len(channels) == 1:
            return channels[0]
        
        # 计算总权重
        total_weight = 0
        weights = []
        
        for channel in channels:
            weight = getattr(channel, 'weight', 1)
            total_weight += weight
            weights.append(weight)
        
        # 加权随机选择
        random_value = random.uniform(0, total_weight)
        current_weight = 0
        
        for i, weight in enumerate(weights):
            current_weight += weight
            if random_value <= current_weight:
                return channels[i]
        
        # 兜底：返回最后一个渠道
        return channels[-1]
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """
        获取所有可用的模型列表
        
        Returns:
            按提供商分组的模型列表
        """
        models_by_provider = {
            "openai": [],
            "anthropic": [],
            "gemini": []
        }
        
        all_channels = self.channel_manager.get_all_channels()
        enabled_channels = [ch for ch in all_channels if ch.enabled]
        
        for channel in enabled_channels:
            provider = channel.provider
            if channel.models_mapping:
                models_by_provider[provider].extend(channel.models_mapping.keys())
        
        # 去重并排序
        for provider in models_by_provider:
            models_by_provider[provider] = sorted(list(set(models_by_provider[provider])))
        
        return models_by_provider
    
    def get_channel_stats(self) -> Dict[str, Any]:
        """
        获取渠道统计信息
        
        Returns:
            渠道统计信息
        """
        all_channels = self.channel_manager.get_all_channels()
        enabled_channels = [ch for ch in all_channels if ch.enabled]
        
        stats = {
            "total_channels": len(all_channels),
            "enabled_channels": len(enabled_channels),
            "disabled_channels": len(all_channels) - len(enabled_channels),
            "channels_by_provider": {
                "openai": 0,
                "anthropic": 0,
                "gemini": 0
            },
            "total_weight": 0,
            "channels": []
        }
        
        for channel in enabled_channels:
            provider = channel.provider
            weight = getattr(channel, 'weight', 1)
            
            stats["channels_by_provider"][provider] += 1
            stats["total_weight"] += weight
            
            stats["channels"].append({
                "id": channel.id,
                "name": channel.name,
                "provider": provider,
                "weight": weight,
                "models_count": len(channel.models_mapping) if channel.models_mapping else 0
            })
        
        return stats


# 全局渠道选择器实例
channel_selector = ChannelSelector()