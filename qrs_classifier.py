import numpy as np

class qrs_classifier:
    """
    QRS complex classifier for MIT-BIH Arrhythmia Database.
    Classifies beats as Normal (N) or Ventricular (V).
    
    """

    def __init__(self, fs: int, pre_ms: int = 60, post_ms: int = 100) -> None:
        self.fs = fs
        self.pre_samp = int(pre_ms * fs / 1000)
        self.post_samp = int(post_ms * fs / 1000)
        self.window_len = self.pre_samp + self.post_samp
        
        self.reference_qrs = None
        self.strategies = ['d2', 'd1', 'dinf', 'corr_distance', 'dr']
        
        self.thresholds = {}
        self.strategy_weights = {}

    # -------------------------------------------------------
    # QRS extraction
    # -------------------------------------------------------
    def extract_qrs(self, signal, ann_samples, ann_symbols):
        qrs_list = []
        labels = []
        positions = []
        search_window = int(0.030 * self.fs) 
        valid_symbols = ['N', 'V']

        for s, sym in zip(ann_samples, ann_symbols):
            if sym not in valid_symbols: continue

            search_start = s - search_window
            search_end = s + search_window
            if search_start < 0 or search_end >= len(signal): continue

            local_slice = signal[search_start:search_end, 0]
            if len(local_slice) == 0: continue
            
            peak_offset = np.argmax(np.abs(local_slice))
            exact_peak = search_start + peak_offset

            start = exact_peak - self.pre_samp
            end = exact_peak + self.post_samp
            if start < 0 or end >= len(signal): continue

            qrs = signal[start:end, 0]
            if len(qrs) != self.window_len: continue

            qrs_list.append(qrs)
            labels.append(sym)
            positions.append(exact_peak)

        return np.array(qrs_list), np.array(labels), np.array(positions)

    # -------------------------------------------------------
    # Normalization
    # -------------------------------------------------------
    @staticmethod
    def normalize_qrs(qrs):
        qrs = qrs - np.mean(qrs)
        max_val = np.max(np.abs(qrs))
        if max_val > 0: qrs = qrs / max_val
        return qrs

    # -------------------------------------------------------
    # Distance Metrics
    # -------------------------------------------------------
    @staticmethod
    def d1(x, y): return np.mean(np.abs(x - y))
    @staticmethod
    def d2(x, y): return np.sqrt(np.mean((x - y) ** 2))
    @staticmethod
    def dinf(x, y): return np.max(np.abs(x - y))
    
    @staticmethod
    def corr_distance(x, y):
        x_m = x - np.mean(x); y_m = y - np.mean(y)
        den = np.sqrt(np.sum(x_m**2) * np.sum(y_m**2))
        if den == 0: return 1.0
        return 1 - (np.sum(x_m * y_m) / den)
    
    @staticmethod
    def dr(x, y):
        x_ave = np.mean(x); y_ave = np.mean(y)
        Sx = np.sum((x - x_ave)**2); Sy = np.sum((y - y_ave)**2)
        denom = np.sqrt(Sx * Sy)
        if denom == 0: return 1.0
        return 1 - (np.sum((x - x_ave) * (y - y_ave)) / denom)

    # -------------------------------------------------------
    # Build Reference
    # -------------------------------------------------------
    def build_reference(self, qrs, labels, minutes=5):
        beats_per_min = 70; max_beats = minutes * beats_per_min
        normal_indices = np.where(labels == 'N')[0]
        if len(normal_indices) == 0: normal_indices = np.arange(len(qrs))
        if len(normal_indices) > max_beats: normal_indices = normal_indices[:max_beats]
        
        normal_qrs = qrs[normal_indices]
        self.reference_qrs = np.median(normal_qrs, axis=0) 
        return self.reference_qrs

    # -------------------------------------------------------
    # Training: BALANCED Grid Search
    # -------------------------------------------------------
    def estimate_threshold(self, qrs, labels):
        normal_qrs = qrs[labels == 'N']
        if len(normal_qrs) == 0: normal_qrs = qrs
        
        training_has_pvcs = 'V' in labels
        
        test_k_values = np.arange(1.0, 8.0, 0.2)
        
        self.strategy_weights = {} 

        for strategy_name in self.strategies:
            dist_func = getattr(self, strategy_name)
            distances_n = np.array([dist_func(self.reference_qrs, q) for q in normal_qrs])
            all_distances = np.array([dist_func(self.reference_qrs, q) for q in qrs])
            
            median_dist = np.median(distances_n)
            mad = np.median(np.abs(distances_n - median_dist))
            sigma_mad = mad * 1.4826
            if sigma_mad == 0: sigma_mad = 1e-6

            best_score = -1.0
            best_k = 4.0
            
            for k in test_k_values:
                temp_thresh = median_dist + (k * sigma_mad)
                pred_is_normal = all_distances <= temp_thresh
                pred_labels = np.where(pred_is_normal, 'N', 'V')
                
                Se, Pp, Sp = self.performance_metrics(labels, pred_labels)
                
                if training_has_pvcs:
                    if Sp < 0.90: 
                        metric = 0 
                    else:
                        metric = 2 * Se * Pp / (Se + Pp) if (Se + Pp) > 0 else 0
                else:
                    metric = Sp

                if metric >= best_score:
                    best_score = metric
                    best_k = k
            
            self.thresholds[strategy_name] = median_dist + (best_k * sigma_mad)
            
            if training_has_pvcs:
                self.strategy_weights[strategy_name] = max(best_score, 0.05)
            else:
                if best_score < 0.90:
                    self.strategy_weights[strategy_name] = 0.1
                else:
                    self.strategy_weights[strategy_name] = 0.5

            # print(f"Strategy {strategy_name}: K={best_k:.1f}, Weight={self.strategy_weights[strategy_name]:.2f}")

    # -------------------------------------------------------
    # Classification: Weighted Voting (RELAXED FILTER)
    # -------------------------------------------------------
    def classify(self, qrs, positions=None):
        final_predictions = []
        if not self.thresholds: return np.array(['N'] * len(qrs))

        rr_intervals = []
        avg_rr = 0
        if positions is not None and len(positions) > 1:
            rr_intervals = np.diff(positions)
            avg_rr = np.median(rr_intervals)
        
        for i, q in enumerate(qrs):
            score_v = 0.0
            score_n = 0.0
            
            is_early = False
            has_pause = False
            if len(rr_intervals) > 0 and i > 0:
                if i-1 < len(rr_intervals) and rr_intervals[i-1] < (0.85 * avg_rr): is_early = True
                if i < len(rr_intervals) and rr_intervals[i] > (1.10 * avg_rr): has_pause = True

            for strategy_name in self.strategies:
                weight = self.strategy_weights.get(strategy_name, 0.0)
                dist_func = getattr(self, strategy_name)
                threshold = self.thresholds[strategy_name]
                d = dist_func(self.reference_qrs, q)
                
                vote = 'N'
                if d > threshold:
                    vote = 'V'
                elif is_early and has_pause:
                    if d > (0.7 * threshold): vote = 'V'
                elif is_early:
                    if d > (0.9 * threshold): vote = 'V'
                
                if weight < 0.15:
                    vote = 'N'

                if vote == 'V': score_v += weight
                else: score_n += weight
            
            if score_v > score_n: final_predictions.append('V')
            else: final_predictions.append('N')

        return np.array(final_predictions)

    @staticmethod
    def performance_metrics(true_labels, predicted_labels):
        TP = np.sum((true_labels == 'V') & (predicted_labels == 'V'))
        TN = np.sum((true_labels == 'N') & (predicted_labels == 'N'))
        FP = np.sum((true_labels == 'N') & (predicted_labels == 'V'))
        FN = np.sum((true_labels == 'V') & (predicted_labels == 'N'))

        Se = TP / (TP + FN + 1e-10)
        Pp = TP / (TP + FP + 1e-10)
        Sp = TN / (TN + FP + 1e-10)
        return Se, Pp, Sp