"""
CCKS2021因果关系抽取 - 评估脚本
计算 Precision, Recall, F1 等指标
"""

import json
from typing import List, Dict, Set, Tuple
from collections import defaultdict


class CCKSEvaluator:
    """CCKS评估器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置统计"""
        self.total_pred = 0      # 预测的因果关系总数
        self.total_gold = 0      # 真实的因果关系总数
        self.total_correct = 0   # 正确预测的数量

        # 事件论元抽取（EAE）统计：论元角色、类型与跨度都匹配才计为正确
        self.eae_pred = 0
        self.eae_gold = 0
        self.eae_correct = 0

        # 因果事件类型（CET）统计：仅比较因果类型正确性
        self.cet_pred = 0
        self.cet_gold = 0
        self.cet_correct = 0
        
        # 详细统计
        self.stats = {
            'total_samples': 0,
            'samples_with_pred': 0,
            'samples_with_gold': 0,
            'samples_correct': 0,
        }
        
        # 按事件类型统计
        self.type_stats = defaultdict(lambda: {'pred': 0, 'gold': 0, 'correct': 0})
    
    def normalize_relation(self, rel: Dict) -> Tuple:
        """
        标准化因果关系，用于比较
        
        返回: (reason_type, reason_product, result_type, result_product)
        """
        return (
            rel.get('reason_type', '').strip(),
            rel.get('reason_product', '').strip(),
            rel.get('result_type', '').strip(),
            rel.get('result_product', '').strip(),
        )

    def normalize_arguments(self, rel: Dict) -> List[Tuple]:
        """将一个因果关系拆解为论元集合（包含角色信息）。"""
        reason = (
            'reason',
            rel.get('reason_type', '').strip(),
            rel.get('reason_product', '').strip(),
            rel.get('reason_region', '').strip(),
            rel.get('reason_industry', '').strip(),
        )
        result = (
            'result',
            rel.get('result_type', '').strip(),
            rel.get('result_product', '').strip(),
            rel.get('result_region', '').strip(),
            rel.get('result_industry', '').strip(),
        )
        return [reason, result]

    def normalize_cet(self, rel: Dict) -> Tuple:
        """仅保留因果类型对，用于 Cause-Effect Type (CET) 评价。"""
        return (
            rel.get('reason_type', '').strip(),
            rel.get('result_type', '').strip(),
        )
    
    def relations_to_set(self, relations: List[Dict]) -> Set[Tuple]:
        """将关系列表转换为集合，便于比较"""
        return set(self.normalize_relation(rel) for rel in relations)
    
    def evaluate_sample(self, pred_result: List[Dict], gold_result: List[Dict]):
        """
        评估单个样本
        
        Args:
            pred_result: 预测的因果关系列表
            gold_result: 真实的因果关系列表
        """
        pred_set = self.relations_to_set(pred_result)
        gold_set = self.relations_to_set(gold_result)

        # 事件论元集合
        pred_args = set(arg for rel in pred_result for arg in self.normalize_arguments(rel))
        gold_args = set(arg for rel in gold_result for arg in self.normalize_arguments(rel))

        # 因果类型集合（忽略论元）
        pred_cet = set(self.normalize_cet(rel) for rel in pred_result)
        gold_cet = set(self.normalize_cet(rel) for rel in gold_result)
        
        # 计算交集（正确预测）
        correct_set = pred_set & gold_set
        correct_args = pred_args & gold_args
        correct_cet = pred_cet & gold_cet
        
        # 更新统计
        self.total_pred += len(pred_set)
        self.total_gold += len(gold_set)
        self.total_correct += len(correct_set)

        # 论元/EAE统计
        self.eae_pred += len(pred_args)
        self.eae_gold += len(gold_args)
        self.eae_correct += len(correct_args)

        # 因果类型/CET统计
        self.cet_pred += len(pred_cet)
        self.cet_gold += len(gold_cet)
        self.cet_correct += len(correct_cet)
        
        self.stats['total_samples'] += 1
        if pred_result:
            self.stats['samples_with_pred'] += 1
        if gold_result:
            self.stats['samples_with_gold'] += 1
        if correct_set:
            self.stats['samples_correct'] += 1
        
        # 按类型统计
        for rel in pred_result:
            rel_type = rel.get('reason_type', '') + '->' + rel.get('result_type', '')
            self.type_stats[rel_type]['pred'] += 1
        
        for rel in gold_result:
            rel_type = rel.get('reason_type', '') + '->' + rel.get('result_type', '')
            self.type_stats[rel_type]['gold'] += 1
        
        for rel in correct_set:
            rel_type = rel[0] + '->' + rel[2]  # reason_type -> result_type
            self.type_stats[rel_type]['correct'] += 1
    
    def compute_metrics(self) -> Dict:
        """
        计算评估指标
        
        Returns:
            包含各种指标的字典
        """
        # ECE：完整因果关系
        precision = self.total_correct / self.total_pred if self.total_pred > 0 else 0
        recall = self.total_correct / self.total_gold if self.total_gold > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # EAE：论元级别
        eae_precision = self.eae_correct / self.eae_pred if self.eae_pred > 0 else 0
        eae_recall = self.eae_correct / self.eae_gold if self.eae_gold > 0 else 0
        eae_f1 = 2 * eae_precision * eae_recall / (eae_precision + eae_recall) if (eae_precision + eae_recall) > 0 else 0

        # CET：因果类型
        cet_precision = self.cet_correct / self.cet_pred if self.cet_pred > 0 else 0
        cet_recall = self.cet_correct / self.cet_gold if self.cet_gold > 0 else 0
        cet_f1 = 2 * cet_precision * cet_recall / (cet_precision + cet_recall) if (cet_precision + cet_recall) > 0 else 0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'eae_precision': eae_precision,
            'eae_recall': eae_recall,
            'eae_f1': eae_f1,
            'cet_precision': cet_precision,
            'cet_recall': cet_recall,
            'cet_f1': cet_f1,
            'total_pred': self.total_pred,
            'total_gold': self.total_gold,
            'total_correct': self.total_correct,
            'eae_pred': self.eae_pred,
            'eae_gold': self.eae_gold,
            'eae_correct': self.eae_correct,
            'cet_pred': self.cet_pred,
            'cet_gold': self.cet_gold,
            'cet_correct': self.cet_correct,
            'stats': self.stats
        }
        
        return metrics
    
    def print_report(self, detailed: bool = True):
        """
        打印评估报告
        
        Args:
            detailed: 是否打印详细信息
        """
        metrics = self.compute_metrics()
        
        print("\n" + "="*80)
        print("CCKS2021 因果关系抽取评估结果")
        print("="*80)

        # 主要指标
        print(f"\n📊 主要指标 (Micro):")
        print(f"  EAE  精确率: {metrics['eae_precision']:.4f}  召回率: {metrics['eae_recall']:.4f}  F1: {metrics['eae_f1']:.4f}")
        print(f"  CET  精确率: {metrics['cet_precision']:.4f}  召回率: {metrics['cet_recall']:.4f}  F1: {metrics['cet_f1']:.4f}")
        print(f"  ECE  精确率: {metrics['precision']:.4f}  召回率: {metrics['recall']:.4f}  F1: {metrics['f1']:.4f}")
        
        # 统计信息
        print(f"\n📈 统计信息:")
        print(f"  预测的因果关系数:   {metrics['total_pred']}")
        print(f"  真实的因果关系数:   {metrics['total_gold']}")
        print(f"  正确预测的数量:     {metrics['total_correct']}")
        print(f"  总样本数:          {self.stats['total_samples']}")
        
        # 样本级别统计
        print(f"\n📋 样本级别统计:")
        print(f"  有预测结果的样本:   {self.stats['samples_with_pred']} "
              f"({self.stats['samples_with_pred']/self.stats['total_samples']*100:.1f}%)")
        print(f"  有真实标注的样本:   {self.stats['samples_with_gold']} "
              f"({self.stats['samples_with_gold']/self.stats['total_samples']*100:.1f}%)")
        print(f"  至少一个正确的样本: {self.stats['samples_correct']} "
              f"({self.stats['samples_correct']/self.stats['total_samples']*100:.1f}%)")
        
        # 详细的类型统计
        if detailed and self.type_stats:
            print(f"\n🔍 按事件类型统计 (Top 10):")
            print(f"{'事件类型':<30} {'预测':<8} {'真实':<8} {'正确':<8} {'P':<8} {'R':<8} {'F1':<8}")
            print("-" * 80)
            
            # 按F1排序
            type_metrics = []
            for rel_type, counts in self.type_stats.items():
                p = counts['correct'] / counts['pred'] if counts['pred'] > 0 else 0
                r = counts['correct'] / counts['gold'] if counts['gold'] > 0 else 0
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0
                type_metrics.append((rel_type, counts, p, r, f))
            
            type_metrics.sort(key=lambda x: x[4], reverse=True)  # 按F1排序
            
            for rel_type, counts, p, r, f in type_metrics[:10]:
                print(f"{rel_type:<30} {counts['pred']:<8} {counts['gold']:<8} "
                      f"{counts['correct']:<8} {p:.4f}   {r:.4f}   {f:.4f}")
        
        print("\n" + "="*80)
        
        return metrics


def evaluate_predictions(pred_file: str, gold_file: str, detailed: bool = True):
    """
    评估预测结果
    
    Args:
        pred_file: 预测结果文件
        gold_file: 真实标注文件
        detailed: 是否显示详细信息
    """
    print(f"加载预测结果: {pred_file}")
    print(f"加载真实标注: {gold_file}")
    
    # 加载预测结果
    predictions = {}
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                predictions[data['text_id']] = data.get('result', [])
    
    # 加载真实标注
    gold_data = {}
    with open(gold_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                gold_data[data['text_id']] = data.get('result', [])
    
    print(f"预测样本数: {len(predictions)}")
    print(f"真实样本数: {len(gold_data)}")
    
    # 创建评估器
    evaluator = CCKSEvaluator()
    
    # 逐样本评估
    missing_in_gold = []
    missing_in_pred = []
    
    for text_id in gold_data:
        if text_id in predictions:
            evaluator.evaluate_sample(
                predictions[text_id],
                gold_data[text_id]
            )
        else:
            missing_in_pred.append(text_id)
    
    # 检查预测中多余的样本
    for text_id in predictions:
        if text_id not in gold_data:
            missing_in_gold.append(text_id)
    
    # 打印报告
    evaluator.print_report(detailed=detailed)
    
    # 警告信息
    if missing_in_pred:
        print(f"\n⚠️  警告: {len(missing_in_pred)} 个样本在预测结果中缺失")
    if missing_in_gold:
        print(f"⚠️  警告: {len(missing_in_gold)} 个样本在真实标注中不存在")
    
    return evaluator.compute_metrics()


def compare_models(pred_files: List[str], gold_file: str, labels: List[str]):
    """
    比较多个模型的性能
    
    Args:
        pred_files: 预测文件列表
        gold_file: 真实标注文件
        labels: 模型标签列表
    """
    print("\n" + "="*80)
    print("模型对比")
    print("="*80)
    
    results = []
    for pred_file, label in zip(pred_files, labels):
        print(f"\n评估模型: {label}")
        print("-" * 80)
        
        evaluator = CCKSEvaluator()
        
        # 加载数据
        predictions = {}
        with open(pred_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    predictions[data['text_id']] = data.get('result', [])
        
        gold_data = {}
        with open(gold_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    gold_data[data['text_id']] = data.get('result', [])
        
        # 评估
        for text_id in gold_data:
            if text_id in predictions:
                evaluator.evaluate_sample(
                    predictions[text_id],
                    gold_data[text_id]
                )
        
        metrics = evaluator.compute_metrics()
        results.append((label, metrics))
    
    # 对比表格
    print("\n" + "="*80)
    print("对比结果")
    print("="*80)
    print(f"\n{'模型':<20} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 80)
    
    for label, metrics in results:
        print(f"{label:<20} {metrics['precision']:.4f}       "
              f"{metrics['recall']:.4f}       {metrics['f1']:.4f}")
    
    print("="*80)


def analyze_errors(pred_file: str, gold_file: str, num_examples: int = 10):
    """
    分析错误案例
    
    Args:
        pred_file: 预测文件
        gold_file: 真实标注文件
        num_examples: 显示多少个错误案例
    """
    print("\n" + "="*80)
    print("错误案例分析")
    print("="*80)
    
    # 加载数据
    predictions = {}
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                predictions[data['text_id']] = {
                    'text': data['text'],
                    'result': data.get('result', [])
                }
    
    gold_data = {}
    with open(gold_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                gold_data[data['text_id']] = {
                    'text': data['text'],
                    'result': data.get('result', [])
                }
    
    evaluator = CCKSEvaluator()
    
    # 收集错误
    false_positives = []  # 误报
    false_negatives = []  # 漏报
    
    for text_id in gold_data:
        if text_id not in predictions:
            continue
        
        pred_set = evaluator.relations_to_set(predictions[text_id]['result'])
        gold_set = evaluator.relations_to_set(gold_data[text_id]['result'])
        
        # 误报 = 预测有但真实没有
        fp = pred_set - gold_set
        if fp:
            false_positives.append({
                'text_id': text_id,
                'text': predictions[text_id]['text'],
                'errors': list(fp)
            })
        
        # 漏报 = 真实有但预测没有
        fn = gold_set - pred_set
        if fn:
            false_negatives.append({
                'text_id': text_id,
                'text': gold_data[text_id]['text'],
                'errors': list(fn)
            })
    
    # 显示误报案例
    print(f"\n❌ 误报案例 (False Positives): {len(false_positives)} 个")
    print("-" * 80)
    for i, case in enumerate(false_positives[:num_examples]):
        print(f"\n案例 {i+1}:")
        print(f"ID: {case['text_id']}")
        print(f"文本: {case['text'][:80]}...")
        print(f"误报的因果关系:")
        for rel in case['errors']:
            print(f"  • {rel[0]} ({rel[1]}) -> {rel[2]} ({rel[3]})")
    
    # 显示漏报案例
    print(f"\n❌ 漏报案例 (False Negatives): {len(false_negatives)} 个")
    print("-" * 80)
    for i, case in enumerate(false_negatives[:num_examples]):
        print(f"\n案例 {i+1}:")
        print(f"ID: {case['text_id']}")
        print(f"文本: {case['text'][:80]}...")
        print(f"漏报的因果关系:")
        for rel in case['errors']:
            print(f"  • {rel[0]} ({rel[1]}) -> {rel[2]} ({rel[3]})")
    
    print("\n" + "="*80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CCKS2021因果关系抽取评估')
    parser.add_argument('--pred', required=True, help='预测结果文件')
    parser.add_argument('--gold', required=True, help='真实标注文件')
    parser.add_argument('--detailed', action='store_true', help='显示详细统计')
    parser.add_argument('--errors', action='store_true', help='分析错误案例')
    parser.add_argument('--num-errors', type=int, default=10, help='显示多少个错误案例')
    
    args = parser.parse_args()
    
    # 评估
    metrics = evaluate_predictions(args.pred, args.gold, detailed=args.detailed)
    
    # 错误分析
    if args.errors:
        analyze_errors(args.pred, args.gold, num_examples=args.num_errors)
    
    # 保存结果
    output_file = args.pred.replace('.txt', '_metrics.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 评估结果已保存到: {output_file}")


if __name__ == '__main__':
    main()