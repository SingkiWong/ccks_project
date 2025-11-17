"""
基于启发式神经网络的因果关系提取方法
Chapter 3: Heuristic Neural Network-based Causal Relation Extraction

实现内容:
1. 启发式规则设计
2. 基于图神经网络的编码
3. 关系分类与损失函数
4. 图更新机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from transformers import BertModel, BertTokenizer
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CausalEvent:
    """事件数据结构"""
    text: str  # 事件文本
    start_pos: int  # 起始位置
    end_pos: int  # 结束位置
    event_type: str  # 事件类型
    arguments: Dict[str, str]  # 论元(产品、区域、行业等)


@dataclass
class CausalPair:
    """因果事件对"""
    cause: CausalEvent  # 原因事件
    effect: CausalEvent  # 结果事件
    confidence: float = 0.0  # 置信度


class HeuristicRules:
    """
    3.2.1 启发式规则设计
    实现词汇触发规则、句法依存规则和语义角色规则
    """
    
    def __init__(self):
        # 表3.1: 金融领域词汇触发规则集
        self.strong_triggers = {
            '导致': 0.9, '致使': 0.9, '引发': 0.9, '造成': 0.9,
            '酿成': 0.9, '促使': 0.9, '因而': 0.9, '因此': 0.9,
            '从而': 0.9, '于是': 0.9, '以致于': 0.9, '因为': 0.9,
            '由于': 0.9, '归因于': 0.9, '归功于': 0.9, '得益于': 0.9,
            '提振': 0.9, '拖累': 0.9, '推高': 0.9, '压低': 0.9,
            '暴跌': 0.9, '暴涨': 0.9, '加息': 0.9, '降准': 0.9
        }
        
        self.weak_triggers = {
            '影响': 0.6, '冲击': 0.6, '扰动': 0.6, '改变': 0.6,
            '关系到': 0.6, '涉及到': 0.6, '取决于': 0.6, '源于': 0.6,
            '下降': 0.6, '增长': 0.6, '扩大': 0.6, '收缩': 0.6,
            '复苏': 0.6, '衰退': 0.6
        }
        
        # 语义角色置信度
        self.semantic_confidence = 0.8
        
        # 句法依存置信度
        self.syntax_confidence = 0.75
    
    def apply_lexical_rules(self, text: str, event_i: CausalEvent, 
                           event_j: CausalEvent) -> float:
        """
        应用词汇触发规则
        
        Args:
            text: 文本
            event_i: 事件i
            event_j: 事件j
            
        Returns:
            置信度分数 c_lex
        """
        # 提取事件之间的文本片段
        start = min(event_i.end_pos, event_j.end_pos)
        end = max(event_i.start_pos, event_j.start_pos)
        between_text = text[start:end]
        
        max_confidence = 0.0
        
        # 检查强触发词
        for trigger, conf in self.strong_triggers.items():
            if trigger in between_text:
                max_confidence = max(max_confidence, conf)
        
        # 检查弱触发词
        for trigger, conf in self.weak_triggers.items():
            if trigger in between_text:
                max_confidence = max(max_confidence, conf)
        
        return max_confidence
    
    def apply_syntax_rules(self, dep_tree: Dict, event_i: CausalEvent,
                          event_j: CausalEvent) -> float:
        """
        应用句法依存规则
        
        Args:
            dep_tree: 依存句法树
            event_i: 事件i
            event_j: 事件j
            
        Returns:
            置信度分数 c_syn
        """
        # 检查依存路径模式
        # nsubj(w, ei) ∧ dobj(w, ej) => "ei 通过 w 导致 ej"
        
        # 简化实现: 检查是否存在特定的依存关系路径
        if self._check_dependency_path(dep_tree, event_i, event_j):
            return self.syntax_confidence
        
        return 0.0
    
    def apply_semantic_rules(self, srl_result: Dict, event_i: CausalEvent,
                            event_j: CausalEvent) -> float:
        """
        应用语义角色标注规则
        
        Args:
            srl_result: 语义角色标注结果
            event_i: 事件i
            event_j: 事件j
            
        Returns:
            置信度分数 c_sem
        """
        # 检查事件i是否被标注为事件j的AM-CAU(原因)角色
        if self._check_causal_role(srl_result, event_i, event_j):
            return self.semantic_confidence
        
        return 0.0
    
    def compute_initial_confidence(self, text: str, event_i: CausalEvent,
                                  event_j: CausalEvent,
                                  dep_tree: Optional[Dict] = None,
                                  srl_result: Optional[Dict] = None) -> float:
        """
        公式 3.3: 计算初始边权重
        A_ij^(0) = max({c_lex, c_syn, c_sem})
        
        Args:
            text: 文本
            event_i: 事件i
            event_j: 事件j
            dep_tree: 依存句法树(可选)
            srl_result: 语义角色标注结果(可选)
            
        Returns:
            初始置信度
        """
        confidences = []
        
        # 词汇规则
        c_lex = self.apply_lexical_rules(text, event_i, event_j)
        confidences.append(c_lex)
        
        # 句法规则
        if dep_tree is not None:
            c_syn = self.apply_syntax_rules(dep_tree, event_i, event_j)
            confidences.append(c_syn)
        
        # 语义规则
        if srl_result is not None:
            c_sem = self.apply_semantic_rules(srl_result, event_i, event_j)
            confidences.append(c_sem)
        
        # 取最大值
        return max(confidences) if confidences else 0.0
    
    def _check_dependency_path(self, dep_tree: Dict, event_i: CausalEvent,
                              event_j: CausalEvent) -> bool:
        """检查依存路径"""
        arcs = dep_tree.get('arcs') or dep_tree.get('edges') or dep_tree.get('dependencies')
        tokens = dep_tree.get('tokens') or dep_tree.get('words') or []

        if not arcs or not tokens:
            return False

        def _nodes_for_event(event: CausalEvent) -> set:
            candidates = set()
            for idx, token in enumerate(tokens):
                start = token.get('start', token.get('begin', token.get('charBegin')))
                end = token.get('end', token.get('charEnd'))
                token_text = token.get('text') or token.get('word', '')

                if start is not None and end is not None:
                    if event.start_pos <= start < event.end_pos or event.start_pos < end <= event.end_pos:
                        candidates.add(idx)
                elif token_text and token_text in event.text:
                    candidates.add(idx)
            return candidates

        cause_nodes = _nodes_for_event(event_i)
        effect_nodes = _nodes_for_event(event_j)

        if not cause_nodes or not effect_nodes:
            return False

        def _get_arc_fields(arc: Dict):
            head = arc.get('head', arc.get('from', arc.get('governor', arc.get('source'))))
            dep = arc.get('dep', arc.get('to', arc.get('dependent', arc.get('target'))))
            rel = arc.get('rel', arc.get('relation', arc.get('label')))
            return head, dep, rel

        subject_rels = {'nsubj', 'nsubjpass', 'subj'}
        object_rels = {'dobj', 'obj', 'pobj', 'attr', 'acomp'}

        for arc in arcs:
            head, dep, rel = _get_arc_fields(arc)
            if rel in subject_rels and dep in cause_nodes:
                for other in arcs:
                    o_head, o_dep, o_rel = _get_arc_fields(other)
                    if head == o_head and o_rel in object_rels and o_dep in effect_nodes:
                        return True

            if rel in subject_rels and dep in effect_nodes:
                for other in arcs:
                    o_head, o_dep, o_rel = _get_arc_fields(other)
                    if head == o_head and o_rel in object_rels and o_dep in cause_nodes:
                        return True

        return False
    
    def _check_causal_role(self, srl_result: Dict, event_i: CausalEvent,
                          event_j: CausalEvent) -> bool:
        """检查语义角色"""
        if not srl_result:
            return False

        frames = srl_result.get('frames') or srl_result.get('verbs') or srl_result
        if not isinstance(frames, list):
            return False

        def _text_overlap(arg_text: str, event: CausalEvent) -> bool:
            return bool(arg_text) and (arg_text in event.text or event.text in arg_text)

        for frame in frames:
            arguments = frame.get('arguments') or frame.get('args') or []
            predicate_text = frame.get('predicate', '') or frame.get('verb', '')

            for arg in arguments:
                role = arg.get('role') or arg.get('label', '')
                arg_text = arg.get('text') or ''.join(arg.get('tokens', []))

                if role in {'AM-CAU', 'ARGM-CAU'} and _text_overlap(arg_text, event_i):
                    if predicate_text and predicate_text in event_j.text:
                        return True
                    if any(_text_overlap(other.get('text') or ''.join(other.get('tokens', [])), event_j)
                           for other in arguments if other is not arg):
                        return True

        return False


class EdgeFeatureExtractor:
    """
    边特征提取器
    提取五类边特征: 句法距离、篇章距离、语义相似度、连接词、规则置信度
    """
    
    def __init__(self, bert_model: BertModel, tokenizer: BertTokenizer):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
    
    def extract_edge_features(self, text: str, event_i: CausalEvent,
                            event_j: CausalEvent, 
                            initial_confidence: float) -> torch.Tensor:
        """
        提取边特征向量 f_ji
        
        Args:
            text: 文本
            event_i: 事件i
            event_j: 事件j
            initial_confidence: 初始置信度
            
        Returns:
            边特征向量
        """
        features = []
        
        # 1. 句法与篇章距离特征
        token_distance = self._compute_token_distance(event_i, event_j)
        sentence_distance = self._compute_sentence_distance(text, event_i, event_j)
        features.extend([token_distance, sentence_distance])
        
        # 2. 事件时序特征
        temporal_order = 1.0 if event_i.end_pos < event_j.start_pos else -1.0
        features.append(temporal_order)
        
        # 3. 语义相似度特征
        cosine_sim = self._compute_cosine_similarity(event_i, event_j)
        features.append(cosine_sim)
        
        # 4. 连接词特征 (简化为embedding的均值)
        conn_features = self._extract_connective_features(text, event_i, event_j)
        features.extend(conn_features)
        
        # 5. 规则置信度特征
        features.append(initial_confidence)
        
        # 确保返回11维
        if len(features) != 11:
            # 补齐或截断到11维
            if len(features) < 11:
                features.extend([0.0] * (11 - len(features)))
            else:
                features = features[:11]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _compute_token_distance(self, event_i: CausalEvent, 
                               event_j: CausalEvent) -> float:
        """计算词距离"""
        distance = abs(event_i.start_pos - event_j.start_pos)
        return np.log(1 + distance)
    
    def _compute_sentence_distance(self, text: str, event_i: CausalEvent,
                                  event_j: CausalEvent) -> float:
        """计算句子距离"""
        # 简化: 计算事件之间的句子数
        between_text = text[min(event_i.end_pos, event_j.end_pos):
                          max(event_i.start_pos, event_j.start_pos)]
        sentence_count = between_text.count('。') + between_text.count('!')
        return float(sentence_count)
    
    def _compute_cosine_similarity(self, event_i: CausalEvent,
                                  event_j: CausalEvent) -> float:
        """
        公式 3.12: 计算语义相似度
        f_cosine = (v_j^(0) · v_i^(0)) / (||v_j^(0)|| ||v_i^(0)||)
        """
        # 简化实现: 使用BERT获取事件表示
        with torch.no_grad():
            # 编码事件i
            tokens_i = self.tokenizer(event_i.text, return_tensors='pt',
                                     padding=True, truncation=True)
            output_i = self.bert_model(**tokens_i)
            emb_i = output_i.last_hidden_state[:, 0, :]  # [CLS] token
            
            # 编码事件j
            tokens_j = self.tokenizer(event_j.text, return_tensors='pt',
                                     padding=True, truncation=True)
            output_j = self.bert_model(**tokens_j)
            emb_j = output_j.last_hidden_state[:, 0, :]
            
            # 计算余弦相似度
            cos_sim = F.cosine_similarity(emb_i, emb_j, dim=1)
            
        return cos_sim.item()
    
    def _extract_connective_features(self, text: str, event_i: CausalEvent,
                                    event_j: CausalEvent) -> List[float]:
        """提取连接词特征"""
        # 简化实现: 返回固定维度的特征
        return [0.0, 0.0, 0.0, 0.0, 0.0]  # 固定返回5维  # 可以扩展为实际的连接词embedding


class CausalGNN(nn.Module):
    """
    3.2.2 基于图神经网络的编码
    3.2.4 图更新机制
    
    实现基于注意力门控的因果图神经网络
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3,
                 num_heads: int = 4, edge_feature_dim: int = 11):
        """
        Args:
            input_dim: 输入维度(事件表示维度)
            hidden_dim: 隐藏层维度
            num_layers: GNN层数 L
            num_heads: 多头注意力头数 K
            edge_feature_dim: 边特征维度
        """
        super(CausalGNN, self).__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # GAT层 - 多层图注意力网络
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels=hidden_dim if i > 0 else input_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                concat=True,
                dropout=0.1,
                add_self_loops=False,
                edge_dim=edge_feature_dim
            )
            for i in range(num_layers)
        ])
        
        # 边置信度门控网络
        # 公式 3.11: g_ji = σ(u^T [z_i^(l-1); z_j^(l-1); f_ji])
        gate_input_dim = 2 * hidden_dim + edge_feature_dim
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # LayerNorm for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_features: torch.Tensor) -> torch.Tensor:
        """
        算法1: 基于注意力门控的因果图更新机制
        
        Args:
            node_features: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            edge_features: 边特征 [num_edges, edge_feature_dim]
            
        Returns:
            更新后的节点表示 [num_nodes, hidden_dim]
        """
        z = node_features
        
        # 逐层更新
        for layer_idx in range(self.num_layers):
            # 保存残差
            z_residual = z if layer_idx > 0 else None
            
            # 计算边置信度门控 g_ji
            edge_gates = self._compute_edge_gates(z, edge_index, edge_features)
            
            # GAT层传播消息
            # 公式 3.10: m_i^k = Σ_{j∈N_i} w̃_ji · W_k z_j^(l-1)
            z = self.gat_layers[layer_idx](
                z, edge_index, edge_attr=edge_features
            )
            
            # 应用边门控
            z = self._apply_edge_gating(z, edge_index, edge_gates)
            
            # LayerNorm
            z = self.layer_norms[layer_idx](z)
            
            # 残差连接
            if z_residual is not None and z.shape == z_residual.shape:
                z = z + z_residual
            
            # 激活函数
            z = F.elu(z)
        
        return z
    
    def _compute_edge_gates(self, node_features: torch.Tensor,
                           edge_index: torch.Tensor,
                           edge_features: torch.Tensor) -> torch.Tensor:
        """
        计算边置信度门控
        公式 3.11: g_ji = σ(u^T [z_i^(l-1); z_j^(l-1); f_ji])
        
        Args:
            node_features: 节点特征
            edge_index: 边索引
            edge_features: 边特征
            
        Returns:
            边门控值 [num_edges, 1]
        """
        # 获取源节点和目标节点
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        # 拼接特征 [z_i; z_j; f_ji]
        src_features = node_features[src_nodes]
        dst_features = node_features[dst_nodes]
        
        gate_input = torch.cat([
            src_features, dst_features, edge_features
        ], dim=-1)
        
        # 计算门控值
        gates = self.gate_network(gate_input)
        
        return gates
    
    def _apply_edge_gating(self, node_features: torch.Tensor,
                          edge_index: torch.Tensor,
                          edge_gates: torch.Tensor) -> torch.Tensor:
        """应用边门控到节点特征"""
        # 简化实现: 通过加权聚合应用门控
        return node_features


class CausalRelationClassifier(nn.Module):
    """
    3.2.3 关系分类与损失函数
    
    因果关系分类器和焦点损失函数
    """
    
    def __init__(self, hidden_dim: int, num_classes: int = 3):
        """
        Args:
            hidden_dim: 隐藏层维度
            num_classes: 分类类别数 (2: 有/无因果, 3: 原因/结果/无关系)
        """
        super(CausalRelationClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # MLP分类器
        # 公式 2.20: p = softmax(MLP(ep_ij))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, event_i_repr: torch.Tensor,
                event_j_repr: torch.Tensor) -> torch.Tensor:
        """
        分类事件对是否存在因果关系
        
        Args:
            event_i_repr: 事件i的表示 [batch_size, hidden_dim]
            event_j_repr: 事件j的表示 [batch_size, hidden_dim]
            
        Returns:
            分类logits [batch_size, num_classes]
        """
        # 公式 2.19: 事件对表示 - 使用拼接方式
        # ep_ij = [e_i, e_j]
        event_pair_repr = torch.cat([event_i_repr, event_j_repr], dim=-1)
        
        # 分类
        logits = self.classifier(event_pair_repr)
        
        return logits
    
    def compute_focal_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                          alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        """
        公式 3.22: 自适应焦点损失函数
        L_focal = -Σ_{j=1}^N Σ_{i=1}^C α_i^j (1 - p_i^j)^γ y_i^j log(p_i^j)
        
        Args:
            logits: 模型输出 [batch_size, num_classes]
            targets: 真实标签 [batch_size]
            alpha: 类别平衡因子
            gamma: 聚焦参数
            
        Returns:
            focal loss
        """
        # 计算概率
        probs = F.softmax(logits, dim=1)
        
        # 获取目标类别的概率
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        
        # 计算focal loss
        focal_weight = (1 - pt) ** gamma
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        focal_loss = alpha * focal_weight * ce_loss
        
        return focal_loss.mean()


class CausalExtractionModel(nn.Module):
    """
    完整的因果关系抽取模型
    整合启发式规则、GNN编码和分类器
    """
    
    def __init__(self, bert_model_name: str = 'bert-base-chinese',
                 hidden_dim: int = 768, num_gnn_layers: int = 3,
                 num_heads: int = 4):
        super(CausalExtractionModel, self).__init__()
        
        # BERT编码器
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # 启发式规则
        self.heuristic_rules = HeuristicRules()
        
        # 边特征提取器
        self.edge_feature_extractor = EdgeFeatureExtractor(
            self.bert, self.tokenizer
        )
        
        # 因果图神经网络
        self.causal_gnn = CausalGNN(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            num_heads=num_heads
        )
        
        # 关系分类器
        self.classifier = CausalRelationClassifier(
            hidden_dim=hidden_dim,
            num_classes=3  # 原因、结果、无关系
        )
        
    def encode_events(self, text: str, events: List[CausalEvent]) -> torch.Tensor:
        """
        使用BERT编码事件
        
        Args:
            text: 文本
            events: 事件列表
            
        Returns:
            事件表示 [num_events, hidden_dim]
        """
        event_representations = []
        
        for event in events:
            # 编码事件文本
            tokens = self.tokenizer(
                event.text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            )
            
            with torch.no_grad():
                outputs = self.bert(**tokens)
                # 使用[CLS] token的表示
                event_repr = outputs.last_hidden_state[:, 0, :]
                event_representations.append(event_repr)
        
        return torch.cat(event_representations, dim=0)
    
    def build_initial_graph(self, text: str, events: List[CausalEvent]
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用启发式规则构建初始因果图
        
        Args:
            text: 文本
            events: 事件列表
            
        Returns:
            edge_index: 边索引 [2, num_edges]
            edge_features: 边特征 [num_edges, feature_dim]
        """
        edge_list = []
        edge_features = []
        
        # 对所有事件对应用启发式规则
        for i, event_i in enumerate(events):
            for j, event_j in enumerate(events):
                if i != j:
                    # 计算初始置信度
                    confidence = self.heuristic_rules.compute_initial_confidence(
                        text, event_i, event_j
                    )
                    
                    # 如果置信度大于阈值,建立边
                    if confidence > 0.3:
                        edge_list.append([i, j])
                        
                        # 提取边特征
                        edge_feat = self.edge_feature_extractor.extract_edge_features(
                            text, event_i, event_j, confidence
                        )
                        edge_features.append(edge_feat)
        
        if len(edge_list) == 0:
            # 如果没有边,返回空tensor
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 11))
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_features = torch.stack(edge_features)
        
        return edge_index, edge_features
    
    def forward(self, text: str, events: List[CausalEvent],
                event_pairs: Optional[List[Tuple[int, int]]] = None
               ) -> Dict[str, torch.Tensor]:
        """
        完整的前向传播
        
        Args:
            text: 文本
            events: 事件列表
            event_pairs: 需要分类的事件对索引(可选)
            
        Returns:
            包含logits和节点表示的字典
        """
        # 1. 编码事件
        event_reprs = self.encode_events(text, events)
        
        # 2. 构建初始图
        edge_index, edge_features = self.build_initial_graph(text, events)
        
        # 3. 图神经网络更新
        if edge_index.shape[1] > 0:
            updated_reprs = self.causal_gnn(event_reprs, edge_index, edge_features)
        else:
            updated_reprs = event_reprs
        
        # 4. 分类事件对
        results = {'node_representations': updated_reprs}
        
        if event_pairs is not None:
            logits_list = []
            for i, j in event_pairs:
                logits = self.classifier(
                    updated_reprs[i:i+1],
                    updated_reprs[j:j+1]
                )
                logits_list.append(logits)
            
            results['logits'] = torch.cat(logits_list, dim=0)
        
        return results
    
    def predict(self, text: str, events: List[CausalEvent],
                threshold: float = 0.5) -> List[CausalPair]:
        """
        预测文本中的因果事件对
        
        Args:
            text: 文本
            events: 事件列表
            threshold: 置信度阈值
            
        Returns:
            因果事件对列表
        """
        self.eval()
        
        # 生成所有可能的事件对
        event_pairs = [
            (i, j) for i in range(len(events)) 
            for j in range(len(events)) if i != j
        ]
        
        with torch.no_grad():
            outputs = self.forward(text, events, event_pairs)
            logits = outputs['logits']
            probs = F.softmax(logits, dim=1)
            
            # 预测因果关系
            causal_pairs = []
            for idx, (i, j) in enumerate(event_pairs):
                # 类别: 0=原因, 1=结果, 2=无关系
                pred_class = torch.argmax(probs[idx]).item()
                confidence = probs[idx][pred_class].item()
                
                if pred_class < 2 and confidence > threshold:
                    if pred_class == 0:
                        # i是j的原因
                        pair = CausalPair(
                            cause=events[i],
                            effect=events[j],
                            confidence=confidence
                        )
                    else:
                        # j是i的原因
                        pair = CausalPair(
                            cause=events[j],
                            effect=events[i],
                            confidence=confidence
                        )
                    causal_pairs.append(pair)
        
        return causal_pairs


def train_step(model: CausalExtractionModel, text: str, 
               events: List[CausalEvent], labels: torch.Tensor,
               event_pairs: List[Tuple[int, int]], optimizer, 
               alpha: float = 0.25, gamma: float = 2.0) -> float:
    """
    训练一个step
    
    Args:
        model: 模型
        text: 文本
        events: 事件列表
        labels: 标签 [num_pairs]
        event_pairs: 事件对索引
        optimizer: 优化器
        alpha: focal loss参数
        gamma: focal loss参数
        
    Returns:
        loss值
    """
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    outputs = model(text, events, event_pairs)
    logits = outputs['logits']
    
    # 计算focal loss
    loss = model.classifier.compute_focal_loss(logits, labels, alpha, gamma)
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate(model: CausalExtractionModel, test_data: List[Dict]) -> Dict[str, float]:
    """
    评估模型性能
    
    Args:
        model: 模型
        test_data: 测试数据
        
    Returns:
        评估指标(Precision, Recall, F1)
    """
    model.eval()
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for sample in test_data:
        text = sample['text']
        events = sample['events']
        true_pairs = set(sample['causal_pairs'])  # (cause_idx, effect_idx)
        
        # 预测
        predicted_pairs = model.predict(text, events)
        pred_pairs = set([
            (events.index(p.cause), events.index(p.effect))
            for p in predicted_pairs
        ])
        
        # 计算指标
        true_positives += len(pred_pairs & true_pairs)
        false_positives += len(pred_pairs - true_pairs)
        false_negatives += len(true_pairs - pred_pairs)
    
    # 计算P, R, F1
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


if __name__ == '__main__':
    # 示例使用
    print("初始化模型...")
    model = CausalExtractionModel(
        bert_model_name='bert-base-chinese',
        hidden_dim=768,
        num_gnn_layers=3,
        num_heads=4
    )
    
    # 示例文本和事件
    text = "央行加息导致股市下跌,投资者信心受挫。"
    events = [
        CausalEvent(text="央行加息", start_pos=0, end_pos=4, 
                   event_type="政策变动", arguments={}),
        CausalEvent(text="股市下跌", start_pos=6, end_pos=10,
                   event_type="市场变化", arguments={}),
        CausalEvent(text="投资者信心受挫", start_pos=11, end_pos=18,
                   event_type="心理影响", arguments={})
    ]
    
    print("\n预测因果关系...")
    causal_pairs = model.predict(text, events)
    
    print(f"\n发现 {len(causal_pairs)} 个因果关系:")
    for pair in causal_pairs:
        print(f"原因: {pair.cause.text} -> 结果: {pair.effect.text} "
              f"(置信度: {pair.confidence:.3f})")
    
    print("\n模型结构:")
    print(f"- BERT编码器: {model.bert.config.hidden_size}维")
    print(f"- GNN层数: {model.causal_gnn.num_layers}")
    print(f"- 注意力头数: {model.causal_gnn.num_heads}")
    print(f"- 分类类别数: {model.classifier.num_classes}")
