"""
CCKS2021因果关系抽取数据集适配器
适配真实的CCKS数据格式
"""

import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
from causality_extraction import CausalEvent, CausalPair


class CCKSDatasetAdapter:
    """
    CCKS数据集适配器
    
    数据格式:
    {
        "text_id": "1291633",
        "text": "文本内容", 
        "result": [
            {
                "reason_type": "台风",
                "reason_product": "",
                "reason_region": "巴西",
                "reason_industry": "",
                "result_type": "供给减少",
                "result_product": "铁矿石",
                "result_region": "",
                "result_industry": ""
            }
        ]
    }
    """
    
    # 事件类型映射
    EVENT_TYPES = {
        # 原因类型
        "台风": "自然灾害",
        "洪涝": "自然灾害", 
        "寒潮": "自然灾害",
        "其他自然灾害": "自然灾害",
        "猪瘟": "疫情",
        "其他畜牧疫情": "疫情",
        "需求增加": "市场变化",
        "需求减少": "市场变化",
        "供给增加": "市场变化",
        "供给减少": "市场变化",
        "市场价格提升": "价格变动",
        "市场价格下降": "价格变动",
        "限产": "政策变动",
        "进口下降": "贸易变化",
        "出口下降": "贸易变化",
        "其他贸易摩擦": "贸易变化",
        "运营成本提升": "成本变化",
        "运营成本下降": "成本变化",
        "销量（消费）增加": "消费变化",
        "销量（消费）减少": "消费变化",
        "负向影响": "市场影响",
        "正向影响": "市场影响",
        # 结果类型
        "产品利润下降": "利润变化"
    }
    
    @staticmethod
    def parse_ccks_sample(sample: Dict) -> Dict:
        """
        解析CCKS数据样本
        
        Args:
            sample: CCKS格式的样本
            
        Returns:
            解析后的数据
        """
        text = sample['text']
        text_id = sample['text_id']
        
        # 提取因果关系
        causal_relations = sample.get('result', [])
        
        # 提取所有事件(原因和结果)
        events = []
        event_pairs = []
        
        for idx, relation in enumerate(causal_relations):
            # 构造原因事件
            reason_text = CCKSDatasetAdapter._build_event_text(
                relation, 'reason'
            )
            reason_event = CausalEvent(
                text=reason_text,
                start_pos=0,  # CCKS数据没有位置信息,使用默认值
                end_pos=len(reason_text),
                event_type=CCKSDatasetAdapter.EVENT_TYPES.get(
                    relation.get('reason_type', ''), '未知类型'
                ),
                arguments={
                    'product': relation.get('reason_product', ''),
                    'region': relation.get('reason_region', ''),
                    'industry': relation.get('reason_industry', ''),
                    'type': relation.get('reason_type', '')
                }
            )
            
            # 构造结果事件
            result_text = CCKSDatasetAdapter._build_event_text(
                relation, 'result'
            )
            result_event = CausalEvent(
                text=result_text,
                start_pos=len(reason_text),
                end_pos=len(reason_text) + len(result_text),
                event_type=CCKSDatasetAdapter.EVENT_TYPES.get(
                    relation.get('result_type', ''), '未知类型'
                ),
                arguments={
                    'product': relation.get('result_product', ''),
                    'region': relation.get('result_region', ''),
                    'industry': relation.get('result_industry', ''),
                    'type': relation.get('result_type', '')
                }
            )
            
            # 添加事件
            if reason_event not in events:
                events.append(reason_event)
            if result_event not in events:
                events.append(result_event)
            
            # 添加因果对
            reason_idx = events.index(reason_event) if reason_event in events else len(events) - 2
            result_idx = events.index(result_event) if result_event in events else len(events) - 1
            event_pairs.append((reason_idx, result_idx))
        
        return {
            'text_id': text_id,
            'text': text,
            'events': events,
            'causal_pairs': event_pairs
        }
    
    @staticmethod
    def _build_event_text(relation: Dict, event_type: str) -> str:
        """
        构建事件文本描述
        
        Args:
            relation: 因果关系字典
            event_type: 'reason' 或 'result'
            
        Returns:
            事件文本
        """
        parts = []
        
        # 区域
        region = relation.get(f'{event_type}_region', '')
        if region:
            parts.append(region)
        
        # 行业
        industry = relation.get(f'{event_type}_industry', '')
        if industry:
            parts.append(industry)
        
        # 产品
        product = relation.get(f'{event_type}_product', '')
        if product:
            # 处理多个产品(用逗号分隔)
            products = product.split(',')
            parts.extend(products)
        
        # 类型(作为主要描述)
        event_type_str = relation.get(f'{event_type}_type', '')
        if event_type_str:
            parts.append(event_type_str)
        
        # 如果没有任何信息,使用类型
        if not parts:
            parts.append(event_type_str or '未知事件')
        
        return ' '.join(parts)


class CCKSCausalityDataset(Dataset):
    """
    CCKS2021因果关系数据集
    """
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: 数据文件路径(.txt格式,每行一个JSON)
        """
        self.data = self.load_data(data_path)
        self.adapter = CCKSDatasetAdapter()
    
    def load_data(self, data_path: str) -> List[Dict]:
        """加载数据"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        sample = json.loads(line)
                        data.append(sample)
                    except json.JSONDecodeError as e:
                        print(f"解析JSON失败: {e}")
                        continue
        
        print(f"加载了 {len(data)} 条数据")
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取一个样本"""
        sample = self.data[idx]
        parsed = self.adapter.parse_ccks_sample(sample)
        return parsed


def analyze_ccks_dataset(data_path: str):
    """
    分析CCKS数据集统计信息
    """
    dataset = CCKSCausalityDataset(data_path)
    
    print("\n" + "="*80)
    print("CCKS数据集统计分析")
    print("="*80)
    
    # 统计信息
    total_samples = len(dataset)
    total_events = 0
    total_causal_pairs = 0
    
    event_types_count = {}
    reason_types_count = {}
    result_types_count = {}
    
    for i in range(len(dataset)):
        sample = dataset[i]
        events = sample['events']
        pairs = sample['causal_pairs']
        
        total_events += len(events)
        total_causal_pairs += len(pairs)
        
        # 统计事件类型
        for event in events:
            event_type = event.event_type
            event_types_count[event_type] = event_types_count.get(event_type, 0) + 1
            
            # 统计原因/结果类型
            arg_type = event.arguments.get('type', '')
            if arg_type:
                if any(keyword in arg_type for keyword in ['需求', '供给', '价格', '成本', '销量']):
                    reason_types_count[arg_type] = reason_types_count.get(arg_type, 0) + 1
                else:
                    result_types_count[arg_type] = result_types_count.get(arg_type, 0) + 1
    
    print(f"\n样本总数: {total_samples}")
    print(f"事件总数: {total_events}")
    print(f"因果对总数: {total_causal_pairs}")
    print(f"平均每个样本的事件数: {total_events/total_samples:.2f}")
    print(f"平均每个样本的因果对数: {total_causal_pairs/total_samples:.2f}")
    
    print(f"\n事件类型分布 (Top 10):")
    sorted_types = sorted(event_types_count.items(), key=lambda x: x[1], reverse=True)
    for event_type, count in sorted_types[:10]:
        print(f"  {event_type}: {count} ({count/total_events*100:.1f}%)")
    
    print(f"\n原因类型分布 (Top 10):")
    sorted_reasons = sorted(reason_types_count.items(), key=lambda x: x[1], reverse=True)
    for reason_type, count in sorted_reasons[:10]:
        print(f"  {reason_type}: {count}")
    
    print(f"\n结果类型分布 (Top 10):")
    sorted_results = sorted(result_types_count.items(), key=lambda x: x[1], reverse=True)
    for result_type, count in sorted_results[:10]:
        print(f"  {result_type}: {count}")
    
    print("\n示例数据:")
    print("-"*80)
    sample = dataset[0]
    print(f"Text ID: {sample['text_id']}")
    print(f"Text: {sample['text'][:100]}...")
    print(f"事件数: {len(sample['events'])}")
    print("事件列表:")
    for i, event in enumerate(sample['events'][:3]):
        print(f"  事件{i+1}: {event.text} ({event.event_type})")
        print(f"    论元: {event.arguments}")
    print(f"因果对: {sample['causal_pairs']}")
    print("="*80)


if __name__ == '__main__':
    # 分析训练集
    print("分析训练集...")
    analyze_ccks_dataset('/mnt/user-data/uploads/ccks_task2_train.txt')
    
    # 分析测试集
    print("\n\n分析测试集...")
    analyze_ccks_dataset('/mnt/user-data/uploads/ccks_task2_eval_data.txt')
