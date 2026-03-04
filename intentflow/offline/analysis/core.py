import os
import glob
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

class OTTAAnalyzer:
    """
    Core library for loading and analyzing OTTA .npz stats files.
    Consolidates data extraction, handling missing fields, and computing common metrics.
    """
    
    def __init__(self, target_dir: Optional[str] = None):
        if target_dir is None:
            # Default behavior: find latest directory
            base_dir = "intentflow/offline/results"
            self.results_dir = self._find_latest_results_dir(base_dir)
        else:
            if not os.path.exists(target_dir):
                raise ValueError(f"Directory not found: {target_dir}")
            
            # Check if target_dir actually contains .npz files directly
            if glob.glob(os.path.join(target_dir, "otta_stats_s*.npz")):
                self.results_dir = target_dir
            else:
                self.results_dir = self._find_latest_results_dir(target_dir)
                
        if not self.results_dir:
            raise FileNotFoundError(f"No valid results directory found.")
            
        print(f"Initialized OTTAAnalyzer with directory: {self.results_dir}")
        self._df = None # Cached DataFrame of sample-level data
        self._subject_summary = None # Cached DataFrame of subject-level metrics

    def _find_latest_results_dir(self, base_dir: str) -> Optional[str]:
        """Find the most recent results directory containing OTTA stats."""
        dirs = glob.glob(os.path.join(base_dir, "*_bcic2*_seed-*_GPU0_*")) 
        
        valid_dirs = []
        for d in dirs:
            if glob.glob(os.path.join(d, "otta_stats_s*.npz")):
                valid_dirs.append(d)
                
        if not valid_dirs:
            return None
            
        return max(valid_dirs, key=os.path.getmtime)

    def load_data(self) -> pd.DataFrame:
        """
        Loads all NPZ files into a unified Pandas DataFrame where each row is a sample.
        """
        if self._df is not None:
            return self._df
            
        stats_files = sorted(glob.glob(os.path.join(self.results_dir, "otta_stats_s*.npz")))
        if not stats_files:
            raise FileNotFoundError(f"No .npz files found in {self.results_dir}")

        all_records = []
        
        for fpath in stats_files:
            subject_id = os.path.basename(fpath).split('_')[2]
            data = np.load(fpath)
            
            pmax = data['pmax']
            sal = data['sal']
            adapted = data['adapted'].astype(bool)
            pred = data['pred']
            
            total_samples = len(pmax)
            
            label = data['label'] if 'label' in data else np.full(total_samples, np.nan)
            original_pred = data['original_pred'] if 'original_pred' in data else pred
            neuro_score = data['neuro_score'] if 'neuro_score' in data else np.zeros(total_samples)
            
            for i in range(total_samples):
                all_records.append({
                    'subject': subject_id,
                    'sample_idx': i,
                    'pmax': pmax[i],
                    'sal': sal[i],
                    'neuro_score': neuro_score[i],
                    'adapted': adapted[i],
                    'pred': pred[i],
                    'original_pred': original_pred[i],
                    'label': label[i]
                })
                
        self._df = pd.DataFrame(all_records)
        self._add_computed_columns()
        return self._df

    def _add_computed_columns(self):
        """Adds derived metrics to the underlying sample DataFrame."""
        df = self._df
        df['is_correct'] = df['pred'] == df['label']
        df['was_correct_orig'] = df['original_pred'] == df['label']
        df['static_adapt_decision'] = df['sal'] > 0.5
        df['flip_encouraged'] = (~df['static_adapt_decision']) & df['adapted']
        df['flip_suppressed'] = df['static_adapt_decision'] & (~df['adapted'])
        self._df = df

    def get_subject_metrics(self) -> pd.DataFrame:
        """
        Aggregates sample data into subject-level metrics.
        """
        if self._subject_summary is not None:
            return self._subject_summary
            
        df = self.load_data()
        summary = []
        for subject, group in df.groupby('subject'):
            total = len(group)
            adapted = group['adapted'].sum()
            acc = group['is_correct'].mean() * 100 if pd.notnull(group['label'].iloc[0]) else np.nan
            orig_acc = group['was_correct_orig'].mean() * 100 if pd.notnull(group['label'].iloc[0]) else np.nan
            
            enc = group['flip_encouraged'].sum()
            sup = group['flip_suppressed'].sum()
            
            if group['neuro_score'].std() > 1e-6 and pd.notnull(group['label'].iloc[0]):
                corr = np.corrcoef(group['neuro_score'], group['is_correct'])[0, 1]
            else:
                corr = 0.0
                
            summary.append({
                'Subject': subject,
                'Total': total,
                'Acc (%)': acc,
                'Orig Acc (%)': orig_acc,
                'Adapt Rate (%)': (adapted / total) * 100,
                'Flip (%)': ((enc + sup) / total) * 100,
                'Enc (N->A)': enc,
                'Sup (A->N)': sup,
                'Mean NS': group['neuro_score'].mean(),
                'Mean SAL': group['sal'].mean(),
                'NS-Acc Corr': corr
            })
            
        self._subject_summary = pd.DataFrame(summary)
        return self._subject_summary
