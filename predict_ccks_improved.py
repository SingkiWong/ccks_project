"""
CCKS2021因果关系抽取 - 改进的预测脚本
使用更智能的事件识别
"""

import json
import torch
import re
from typing import List, Dict
import sys
from tqdm import tqdm

sys.path.insert(0, '.')

from causality_extraction import CausalExtractionModel, CausalEvent


class ImprovedCCKSPredictor:
    """改进的CCKS预测器"""
    
    # 从训练集学到的高频事件类型
    EVENT_TYPES = {
        '供给减少', '供给增加', '市场价格提升', '市场价格下降',
        '需求减少', '需求增加', '产品利润下降', '产品利润增加',
        '销量（消费）减少', '销量（消费）增加', '运营成本提升', '运营成本下降',
        '负向影响', '正向影响', '进口下降', '出口下降'
    }
    
    def __init__(self, model_path: str):
        print("加载模型...")
        self.model = CausalExtractionModel(
            bert_model_name='bert-base-chinese',
            hidden_dim=768,
            num_gnn_layers=3,
            num_heads=4
        )
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ 成功加载模型: {model_path}")
        except Exception as e:
            print(f"⚠ 加载模型失败: {e}")
            print("使用未训练的模型")

        self.model.eval()

    def _extract_by_connectives(self, text: str) -> List[CausalEvent]:
        """基于因果连接词切分的事件候选"""
        patterns = [
            re.compile(r'(?:由于|因为)([^，。；！？]*?)(?:因此|所以|致使|从而|以致|导致|引发|使得)([^，。；！？]*)'),
            re.compile(r'([^，。；！？]*?)(?:导致|引发|致使|促使|使得|结果)([^，。；！？]+)')
        ]

        events: List[CausalEvent] = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                cause_span = match.group(1).strip()
                effect_span = match.group(2).strip()

                if len(cause_span) < 2 or len(effect_span) < 2:
                    continue

                cause_start = text.find(cause_span, match.start())
                effect_start = text.find(effect_span, match.start())

                cause_event = CausalEvent(
                    text=cause_span,
                    start_pos=cause_start,
                    end_pos=cause_start + len(cause_span),
                    event_type=self._infer_event_type(cause_span),
                    arguments={'phrase': cause_span}
                )
                effect_event = CausalEvent(
                    text=effect_span,
                    start_pos=effect_start,
                    end_pos=effect_start + len(effect_span),
                    event_type=self._infer_event_type(effect_span),
                    arguments={'phrase': effect_span}
                )

                events.extend([cause_event, effect_event])

        return events
    
    def extract_events_from_text(self, text: str) -> List[CausalEvent]:
        """
        智能事件提取 - 基于训练数据的模式
        """
        events = []

        # 0. 通过因果连接词切分句子，优先捕获明确的因果链
        events.extend(self._extract_by_connectives(text))
        
        # 1. 基于关键词的精确匹配
        keywords = {
            '价格': ['价格上涨', '价格下跌', '价格提升', '价格下降', '涨价', '降价', '价格回落', '价格上行'],
            '供给': ['供给减少', '供给增加', '供应减少', '供应增加', '产量下降', '产量增加', '供应紧张', '供应充足'],
            '需求': ['需求增加', '需求减少', '需求旺盛', '需求疲软', '订单减少', '订单增加'],
            '成本': ['成本上升', '成本下降', '成本增加', '成本提升', '成本降低'],
            '利润': ['利润下降', '利润增加', '盈利下降', '盈利增加', '收益下滑', '收益增长'],
            '销量': ['销量增加', '销量减少', '销售增长', '销售下滑', '销量下滑'],
            '库存': ['库存增加', '库存减少', '库存高企', '库存下降'],
            '进出口': ['进口增加', '进口减少', '出口增加', '出口下降', '出口下滑'],
        }
        
        for category, phrases in keywords.items():
            for phrase in phrases:
                if phrase in text:
                    # 扩展上下文
                    idx = text.find(phrase)
                    start = max(0, idx - 5)
                    end = min(len(text), idx + len(phrase) + 5)
                    context = text[start:end].strip()
                    
                    # 提取产品名称（假设在关键词前）
                    product_match = re.search(r'([^，。；！？]{1,8})' + re.escape(phrase), text)
                    product = product_match.group(1).strip() if product_match else ''
                    
                    event = CausalEvent(
                        text=context if len(context) > len(phrase) else phrase,
                        start_pos=idx,
                        end_pos=idx + len(phrase),
                        event_type=self._categorize_event(phrase),
                        arguments={'product': product, 'phrase': phrase}
                    )
                    events.append(event)
        
        # 2. 如果事件太少，用句子分割
        if len(events) < 2:
            # 按标点符号分割
            sentences = re.split(r'[，。；、]', text)
            for i, sent in enumerate(sentences):
                sent = sent.strip()
                if len(sent) >= 6:  # 至少6个字
                    # 判断句子类型
                    event_type = self._infer_event_type(sent)
                    
                    event = CausalEvent(
                        text=sent,
                        start_pos=text.find(sent),
                        end_pos=text.find(sent) + len(sent),
                        event_type=event_type,
                        arguments={}
                    )
                    events.append(event)
                    
                    if len(events) >= 8:  # 最多8个事件
                        break
        
        # 3. 去重
        seen = set()
        unique_events = []
        for event in events:
            key = (event.text, event.event_type)
            if key not in seen:
                seen.add(key)
                unique_events.append(event)
        
        return unique_events[:10]  # 最多10个
    
    def _categorize_event(self, phrase: str) -> str:
        """根据短语判断事件类型"""
        phrase_to_type = {
            '价格上涨': '市场价格提升', '涨价': '市场价格提升', '价格回落': '市场价格下降',
            '价格下降': '市场价格下降', '降价': '市场价格下降', '价格上行': '市场价格提升',
            '供给减少': '供给减少', '供应减少': '供给减少', '供应紧张': '供给减少',
            '产量下降': '供给减少', '供给增加': '供给增加', '供应增加': '供给增加',
            '需求增加': '需求增加', '需求旺盛': '需求增加', '订单增加': '需求增加',
            '需求减少': '需求减少', '需求疲软': '需求减少', '订单减少': '需求减少',
            '成本上升': '运营成本提升', '成本增加': '运营成本提升', '成本提升': '运营成本提升',
            '成本下降': '运营成本下降', '成本降低': '运营成本下降',
            '利润下降': '产品利润下降', '盈利下降': '产品利润下降', '收益下滑': '产品利润下降',
            '利润增加': '产品利润增加', '盈利增加': '产品利润增加', '收益增长': '产品利润增加',
            '销量减少': '销量（消费）减少', '销量下滑': '销量（消费）减少', '销售下滑': '销量（消费）减少',
            '销量增加': '销量（消费）增加', '销售增长': '销量（消费）增加',
            '库存增加': '库存增加', '库存减少': '库存减少',
            '进口减少': '进口下降', '出口下降': '出口下降', '出口下滑': '出口下降'
        }

        for key_phrase, mapped_type in phrase_to_type.items():
            if key_phrase in phrase:
                return mapped_type

        if '价格' in phrase or '涨' in phrase or '跌' in phrase:
            return '价格变动'
        elif '供给' in phrase or '供应' in phrase or '产量' in phrase:
            return '市场变化'
        elif '需求' in phrase:
            return '市场变化'
        elif '成本' in phrase:
            return '成本变化'
        elif '利润' in phrase:
            return '利润变化'
        elif '销量' in phrase or '销售' in phrase:
            return '消费变化'
        else:
            return '市场影响'
    
    def _infer_event_type(self, text: str) -> str:
        """推断句子的事件类型"""
        if any(word in text for word in ['上涨', '上升', '提高', '增长', '增加']):
            if '价格' in text:
                return '价格变动'
            elif '成本' in text:
                return '成本变化'
            else:
                return '市场变化'
        elif any(word in text for word in ['下降', '下跌', '减少', '降低']):
            if '价格' in text:
                return '价格变动'
            elif '成本' in text:
                return '成本变化'
            else:
                return '市场变化'
        else:
            return '市场影响'
    
    def predict_sample(self, sample: Dict, threshold: float = 0.5) -> Dict:
        """预测单个样本"""
        text = sample['text']
        text_id = sample['text_id']
        
        # 提取事件
        events = self.extract_events_from_text(text)
        
        if len(events) < 2:
            return {
                'text_id': text_id,
                'text': text,
                'result': []
            }
        
        # 预测因果关系
        try:
            causal_pairs = self.model.predict(text, events, threshold=threshold)
        except Exception as e:
            print(f"\n预测失败 {text_id}: {e}")
            return {
                'text_id': text_id,
                'text': text,
                'result': []
            }
        
        # 转换为CCKS格式
        result = []
        for pair in causal_pairs:
            # 提取产品和类型信息
            cause_product = pair.cause.arguments.get('product', '')
            effect_product = pair.effect.arguments.get('product', '')
            
            # 推断事件类型
            cause_type = self._map_to_ccks_type(pair.cause)
            effect_type = self._map_to_ccks_type(pair.effect)
            
            relation = {
                'reason_type': cause_type,
                'reason_product': cause_product,
                'reason_region': '',
                'reason_industry': '',
                'result_type': effect_type,
                'result_product': effect_product,
                'result_region': '',
                'result_industry': ''
            }
            result.append(relation)
        
        return {
            'text_id': text_id,
            'text': text,
            'result': result
        }
    
    def _map_to_ccks_type(self, event: CausalEvent) -> str:
        """映射到CCKS事件类型"""
        text = event.text
        phrase = event.arguments.get('phrase', '')

        phrase_to_ccks = {
            '供应紧张': '供给减少', '供应充足': '供给增加', '价格回落': '市场价格下降',
            '价格上行': '市场价格提升', '收益下滑': '产品利润下降', '收益增长': '产品利润增加',
            '销量下滑': '销量（消费）减少', '订单减少': '需求减少', '订单增加': '需求增加',
            '出口下滑': '出口下降'
        }

        for key, mapped in phrase_to_ccks.items():
            if key in phrase or key in text:
                return mapped

        # 基于关键词判断
        if '供给' in text or '供应' in text or '产量' in text:
            if any(w in text for w in ['减少', '下降', '不足']):
                return '供给减少'
            elif any(w in text for w in ['增加', '上升', '充足']):
                return '供给增加'
            return '供给减少'  # 默认
        
        elif '价格' in text or '涨' in text or '跌' in text:
            if any(w in text for w in ['上涨', '上升', '提高', '增长']):
                return '市场价格提升'
            elif any(w in text for w in ['下跌', '下降', '降低', '减少']):
                return '市场价格下降'
            return '市场价格提升'
        
        elif '需求' in text:
            if any(w in text for w in ['增加', '上升', '旺盛']):
                return '需求增加'
            elif any(w in text for w in ['减少', '下降', '疲软']):
                return '需求减少'
            return '需求增加'
        
        elif '成本' in text:
            if any(w in text for w in ['上升', '增加', '提高']):
                return '运营成本提升'
            elif any(w in text for w in ['下降', '减少', '降低']):
                return '运营成本下降'
            return '运营成本提升'
        
        elif '利润' in text:
            if any(w in text for w in ['下降', '减少', '降低']):
                return '产品利润下降'
            elif any(w in text for w in ['增加', '上升', '提高']):
                return '产品利润增加'
            return '产品利润下降'
        
        elif '销量' in text or '销售' in text:
            if any(w in text for w in ['增加', '上升', '增长']):
                return '销量（消费）增加'
            elif any(w in text for w in ['减少', '下降', '下滑']):
                return '销量（消费）减少'
            return '销量（消费）减少'
        
        else:
            return '负向影响'  # 默认
    
    def predict_file(self, input_path: str, output_path: str, threshold: float = 0.5):
        """预测整个文件"""
        print(f"\n预测文件: {input_path}")
        print(f"输出到: {output_path}")
        print(f"置信度阈值: {threshold}")
        print("-" * 80)
        
        # 加载数据
        samples = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        
        print(f"共 {len(samples)} 个样本")
        
        # 预测
        results = []
        for sample in tqdm(samples, desc='预测中'):
            result = self.predict_sample(sample, threshold=threshold)
            results.append(result)
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # 统计
        total_relations = sum(len(r['result']) for r in results)
        samples_with_relations = sum(1 for r in results if r['result'])
        
        print(f"\n✓ 预测完成!")
        print(f"统计信息:")
        print(f"  总样本数: {len(results)}")
        print(f"  包含因果关系的样本: {samples_with_relations}")
        print(f"  因果关系总数: {total_relations}")
        print(f"  平均每个样本: {total_relations/len(results):.2f}个因果关系")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='ccks_task2_eval_data.txt')
    parser.add_argument('--output', default='predictions.txt')
    parser.add_argument('--model', default='checkpoints/best_model.pt')
    parser.add_argument('--threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    
    predictor = ImprovedCCKSPredictor(args.model)
    predictor.predict_file(args.input, args.output, args.threshold)


if __name__ == '__main__':
    main()