import numpy as np
from collections import Counter


def find_best_split(feature_values, target_labels):
    """Ищет оптимальный порог разбиения для признака"""
    feat_vals = np.asarray(feature_values)
    tgt_labels = np.asarray(target_labels)

    sorted_indices = np.argsort(feat_vals)
    sorted_features = feat_vals[sorted_indices]
    sorted_targets = tgt_labels[sorted_indices]

    unique_values = np.unique(sorted_features)

    if len(unique_values) < 2:
        return np.array([]), np.array([]), None, None

    candidate_thresholds = []
    for i in range(len(unique_values) - 1):
        midpoint = (unique_values[i] + unique_values[i + 1]) / 2.0
        candidate_thresholds.append(midpoint)

    candidate_thresholds = np.array(candidate_thresholds)

    total_samples = len(tgt_labels)
    positive_total = np.sum(tgt_labels == 1)

    impurities = []

    for threshold in candidate_thresholds:
        left_indices = sorted_features < threshold
        right_indices = ~left_indices

        left_labels = sorted_targets[left_indices]
        right_labels = sorted_targets[right_indices]

        left_count = len(left_labels)
        right_count = len(right_labels)

        # Считаем долю положительных примеров в каждой части
        left_positive = np.sum(left_labels == 1)
        right_positive = np.sum(right_labels == 1)

        # Вычисляем вероятности классов
        p_left_pos = left_positive / left_count if left_count > 0 else 0
        p_left_neg = 1 - p_left_pos

        p_right_pos = right_positive / right_count if right_count > 0 else 0
        p_right_neg = 1 - p_right_pos

        # Критерий Джини для каждой части
        gini_left = 1.0 - (p_left_pos ** 2 + p_left_neg ** 2)
        gini_right = 1.0 - (p_right_pos ** 2 + p_right_neg ** 2)

        # Взвешенная неопределённость
        weighted_gini = (left_count / total_samples) * gini_left + \
                        (right_count / total_samples) * gini_right

        impurities.append(weighted_gini)

    impurities = np.array(impurities)

    # Находим индекс минимальной неопределённости
    optimal_index = np.argmin(impurities)
    optimal_threshold = candidate_thresholds[optimal_index]
    optimal_impurity = impurities[optimal_index]

    return candidate_thresholds, impurities, optimal_threshold, optimal_impurity


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        """Инициализация дерева решений"""
        allowed_types = {"real", "categorical"}
        for ftype in feature_types:
            if ftype not in allowed_types:
                raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, node_data, node_labels, current_node, current_depth=0):
        """Рекурсивное построение узла дерева"""
        if self._check_stopping_criteria(node_labels, current_depth):
            self._create_leaf_node(current_node, node_labels)
            return

        # Ищем лучшее разбиение среди всех признаков
        split_info = self._search_best_split(node_data, node_labels)

        if split_info is None:
            self._create_leaf_node(current_node, node_labels)
            return

        # Настраиваем узел как внутренний
        self._setup_internal_node(
            current_node,
            split_info['feature'],
            split_info['threshold'],
            split_info['feature_type']
        )

        # Разделяем данные на левую и правую части
        left_selector, right_selector = self._partition_data(
            node_data[:, split_info['feature']],
            split_info['threshold'],
            split_info['feature_type']
        )

        # Рекурсивно строим левое поддерево
        self._fit_node(
            node_data[left_selector],
            node_labels[left_selector],
            current_node['left_child'],
            current_depth + 1
        )

        # Рекурсивно строим правое поддерево
        self._fit_node(
            node_data[right_selector],
            node_labels[right_selector],
            current_node['right_child'],
            current_depth + 1
        )

    def _check_stopping_criteria(self, labels, depth):
        """Проверяет, нужно ли остановить рост дерева"""
        # Все метки одинаковые - чистый узел
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            return True

        # Достигнута максимальная глубина
        if self._max_depth is not None:
            if depth >= self._max_depth:
                return True

        # Недостаточно объектов для разбиения
        if self._min_samples_split is not None:
            if len(labels) < self._min_samples_split:
                return True

        return False

    def _create_leaf_node(self, node, labels):
        """Создаёт терминальный узел (лист) с предсказанием"""
        node['type'] = 'terminal'
        label_counts = Counter(labels)
        most_common_label = label_counts.most_common(1)[0][0]
        node['class'] = most_common_label

    def _search_best_split(self, data, labels):
        """Перебирает все признаки и находит оптимальное разбиение"""
        optimal_split = {
            'gini': float('inf'),
            'feature': None,
            'threshold': None,
            'feature_type': None
        }

        num_features = data.shape[1]

        # Перебираем все признаки
        for feat_idx in range(num_features):
            ftype = self._feature_types[feat_idx]
            feature_column = data[:, feat_idx]

            # Обрабатываем в зависимости от типа признака
            if ftype == 'categorical':
                # Преобразуем категории в числа
                transformed_data, category_mapping = self._encode_categorical(feature_column, labels)
            elif ftype == 'real':
                # Для числовых признаков просто конвертируем тип
                transformed_data = feature_column.astype(float)
                category_mapping = None
            else:
                raise ValueError(f"Неизвестный тип признака: {ftype}")

            # Ищем лучший порог для этого признака
            _, _, best_threshold, best_gini = find_best_split(transformed_data, labels)

            if best_gini is None:
                continue

            # Проверяем ограничение на размер листьев
            if not self._validate_leaf_sizes(transformed_data, best_threshold):
                continue

            # Обновляем оптимальное разбиение если нашли лучше
            if best_gini < optimal_split['gini']:
                optimal_split['gini'] = best_gini
                optimal_split['feature'] = feat_idx
                optimal_split['threshold'] = self._prepare_threshold(
                    best_threshold, ftype, category_mapping
                )
                optimal_split['feature_type'] = ftype

        if optimal_split['feature'] is None:
            return None

        return optimal_split

    def _encode_categorical(self, feature_data, labels):
        """Кодирует категориальный признак на основе целевой переменной"""
        # Подсчитываем частоты категорий
        category_frequencies = Counter(feature_data)

        # Подсчитываем положительные метки для каждой категории
        positive_per_category = Counter(feature_data[labels == 1])

        # Вычисляем долю положительных для каждой категории
        category_ratios = {}
        for category, total_count in category_frequencies.items():
            positive_count = positive_per_category.get(category, 0)
            category_ratios[category] = positive_count / total_count

        # Сортируем категории по возрастанию доли положительных
        sorted_categories = sorted(category_ratios.keys(),
                                   key=lambda cat: category_ratios[cat])

        # Создаём числовое кодирование
        encoding_dict = {cat: position for position, cat in enumerate(sorted_categories)}

        # Применяем кодирование к данным
        encoded_values = np.array([encoding_dict.get(val, 0) for val in feature_data])

        return encoded_values, encoding_dict

    def _validate_leaf_sizes(self, feature_data, threshold):
        """Проверяет, что размеры обоих листьев удовлетворяют ограничению"""
        if self._min_samples_leaf is None:
            return True

        # Считаем размеры левого и правого листа
        left_count = np.sum(feature_data < threshold)
        right_count = len(feature_data) - left_count

        return (left_count >= self._min_samples_leaf and right_count >= self._min_samples_leaf)


    def _prepare_threshold(self, threshold, feature_type, categories_info):
        """Подготавливает порог в зависимости от типа признака"""
        if feature_type == 'real':
            return threshold

        # Для категориальных собираем список категорий для левого ребёнка
        left_categories = [cat for cat, idx in categories_info.items()
                           if idx < threshold]
        return left_categories

    def _setup_internal_node(self, node, feature, threshold, feature_type):
        """Настраивает узел как внутренний"""
        node['type'] = 'nonterminal'
        node['feature_split'] = feature
        node['left_child'] = {}
        node['right_child'] = {}

        if feature_type == 'real':
            node['threshold'] = threshold
        else:
            node['categories_split'] = threshold

    def _partition_data(self, feature_data, threshold, feature_type):
        """Разделяет данные на левую и правую части"""
        if feature_type == 'real':
            left_mask = feature_data.astype(float) < threshold
        else:
            left_mask = np.isin(feature_data, threshold)

        right_mask = ~left_mask

        return left_mask, right_mask

    def _predict_node(self, x, node):
        """Рекурсивно спускается по дереву для предсказания"""
        if node["type"] == "terminal":
            return node["class"]

        # Получаем информацию о разбиении
        feature_idx = node["feature_split"]
        ftype = self._feature_types[feature_idx]

        if ftype == "real":
            condition = float(x[feature_idx]) < node["threshold"]
            if condition:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        elif ftype == "categorical":
            condition = x[feature_idx] in node["categories_split"]
            if condition:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError("Unknown feature type")

    def fit(self, X, y):
        """Обучает дерево решений"""
        self._tree = {}
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        """Делает предсказания для набора объектов"""
        predictions = [self._predict_node(x, self._tree) for x in X]
        return np.array(predictions)