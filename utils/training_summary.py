"""
Training Summary Generator

학습 완료 후 epoch별 진행도를 테이블로 정리
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path


class TrainingSummary:
    """학습 진행도를 정리해서 보여주는 클래스"""
    
    def __init__(self, log_dir='logs', history_file='history.json'):
        """
        Args:
            log_dir: 로그 디렉토리
            history_file: history.json 파일명
        """
        self.log_dir = log_dir
        self.history_path = os.path.join(log_dir, history_file)
        self.history = None
        
    def load_history(self):
        """history.json 로드"""
        if not os.path.exists(self.history_path):
            print(f"❌ History file not found: {self.history_path}")
            return False
        
        with open(self.history_path, 'r') as f:
            self.history = json.load(f)
        
        print(f"✅ History loaded: {len(self.history['epoch'])} epochs")
        return True
    
    def get_checkpoint_summary(self, interval=10):
        """
        Epoch별 진행도 테이블 생성 (10 epoch마다)
        
        ✅ 수정: evaluation metrics는 실제 평가된 epoch과 **가장 가까운** 값을 매칭
        예: epoch 0 → epoch 9의 평가 결과 사용
        
        Args:
            interval: 몇 epoch마다 보여줄 것인가 (기본: 10)
        
        Returns:
            DataFrame
        """
        if self.history is None:
            self.load_history()
        
        epochs = self.history.get('epoch', [])
        train_losses = self.history.get('train_loss', [])
        val_losses = self.history.get('val_loss', [])
        feat_stds = self.history.get('feature_std', [])
        avg_cos_sims = self.history.get('avg_cos_sim', [])

        # Clustering metrics (있으면)
        n_clusters = self.history.get('n_clusters', [])
        noise_ratio = self.history.get('noise_ratio', [])
        silhouette = self.history.get('silhouette', [])

        # Rotation invariance (있으면)
        rotation_inv = self.history.get('rotation_invariance', [])

        # 🆕 새 지표
        knn_consistency = self.history.get('knn_consistency', [])
        calinski_harabasz = self.history.get('calinski_harabasz', [])
        davies_bouldin = self.history.get('davies_bouldin', [])
        cluster_consistency_d4 = self.history.get('cluster_consistency_d4', [])
        avg_distance_top_k = self.history.get('avg_distance_top_k', [])
        composite_score = self.history.get('composite_score', [])
        
        # 🔹 Helper function: 가장 가까운 evaluation epoch의 value 찾기
        def find_closest_value(metric_list, target_epoch):
            """
            (epoch, value) 튜플 리스트에서 target_epoch에 **가장 가까운** evaluation epoch의 value 찾기
            예: target=0, metric_list=[(9, 0.6), (19, 0.5)] → 0.6 (epoch 9가 가장 가까움)
            """
            if not metric_list or len(metric_list) == 0:
                return None
            
            # 모든 evaluation epoch 추출
            eval_epochs = []
            eval_values = []
            for item in metric_list:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    eval_epochs.append(int(item[0]))
                    eval_values.append(item[1])
            
            if len(eval_epochs) == 0:
                return None
            
            # 가장 가까운 evaluation epoch 찾기
            closest_idx = min(range(len(eval_epochs)), 
                            key=lambda i: abs(eval_epochs[i] - target_epoch))
            
            return eval_values[closest_idx]
        
        # Interval로 샘플링
        summary_data = []
        
        for idx, epoch_idx in enumerate(range(0, len(epochs), interval)):
            if epoch_idx >= len(epochs):
                break
            
            epoch = epochs[epoch_idx]
            
            # 데이터 추출
            row = {
                'Epoch': int(epoch),
                'Train Loss': f"{train_losses[epoch_idx]:.4f}" if epoch_idx < len(train_losses) else "N/A",
                'Val Loss': f"{val_losses[epoch_idx]:.4f}" if epoch_idx < len(val_losses) else "N/A",
                'Feat Std': f"{feat_stds[epoch_idx]:.4f}" if epoch_idx < len(feat_stds) else "N/A",
                'Avg Cos Sim': f"{avg_cos_sims[epoch_idx]:.4f}" if epoch_idx < len(avg_cos_sims) else "N/A",
            }
            
            # Evaluation metrics: 범위 내 인덱스만 접근
            sil_value = silhouette[idx] if idx < len(silhouette) else None
            row['Silhouette'] = f"{sil_value:.4f}" if sil_value is not None else "N/A"

            ncl_value = n_clusters[idx] if idx < len(n_clusters) else None
            row['n_clusters'] = int(ncl_value) if ncl_value is not None else "N/A"

            noise_value = noise_ratio[idx] if idx < len(noise_ratio) else None
            row['Noise %'] = f"{noise_value*100:.1f}%" if noise_value is not None else "N/A"

            rot_value = rotation_inv[idx] if idx < len(rotation_inv) else None
            row['Rotation Inv'] = f"{rot_value:.4f}" if rot_value is not None else "N/A"

            knn_value = knn_consistency[idx] if idx < len(knn_consistency) else None
            row['kNN Consist'] = f"{knn_value:.4f}" if knn_value is not None else "N/A"

            ch_value = calinski_harabasz[idx] if idx < len(calinski_harabasz) else None
            row['CH Score'] = f"{ch_value:.1f}" if ch_value is not None else "N/A"

            db_value = davies_bouldin[idx] if idx < len(davies_bouldin) else None
            row['DB Score'] = f"{db_value:.4f}" if db_value is not None else "N/A"

            d4_value = cluster_consistency_d4[idx] if idx < len(cluster_consistency_d4) else None
            row['D4 Consist'] = f"{d4_value:.4f}" if d4_value is not None else "N/A"

            dist_value = avg_distance_top_k[idx] if idx < len(avg_distance_top_k) else None
            row['Avg Dist@k'] = f"{dist_value:.4f}" if dist_value is not None else "N/A"

            comp_value = composite_score[idx] if idx < len(composite_score) else None
            row['Composite'] = f"{comp_value:.4f}" if comp_value is not None else "N/A"

            summary_data.append(row)

        # 마지막 epoch도 추가 (완료 상태)
        if len(epochs) > 0 and (len(epochs) - 1) % interval != 0:
            epoch_idx = len(epochs) - 1
            epoch = epochs[epoch_idx]
            last_eval_idx = len(silhouette) - 1  # 마지막 평가 인덱스

            row = {
                'Epoch': int(epoch),
                'Train Loss': f"{train_losses[epoch_idx]:.4f}" if epoch_idx < len(train_losses) else "N/A",
                'Val Loss': f"{val_losses[epoch_idx]:.4f}" if epoch_idx < len(val_losses) else "N/A",
                'Feat Std': f"{feat_stds[epoch_idx]:.4f}" if epoch_idx < len(feat_stds) else "N/A",
                'Avg Cos Sim': f"{avg_cos_sims[epoch_idx]:.4f}" if epoch_idx < len(avg_cos_sims) else "N/A",
            }

            sil_value = silhouette[last_eval_idx] if last_eval_idx >= 0 else None
            row['Silhouette'] = f"{sil_value:.4f}" if sil_value is not None else "N/A"

            ncl_value = n_clusters[last_eval_idx] if last_eval_idx >= 0 and last_eval_idx < len(n_clusters) else None
            row['n_clusters'] = int(ncl_value) if ncl_value is not None else "N/A"

            noise_value = noise_ratio[last_eval_idx] if last_eval_idx >= 0 and last_eval_idx < len(noise_ratio) else None
            row['Noise %'] = f"{noise_value*100:.1f}%" if noise_value is not None else "N/A"

            rot_value = rotation_inv[last_eval_idx] if last_eval_idx >= 0 and last_eval_idx < len(rotation_inv) else None
            row['Rotation Inv'] = f"{rot_value:.4f}" if rot_value is not None else "N/A"

            knn_value = knn_consistency[last_eval_idx] if last_eval_idx >= 0 and last_eval_idx < len(knn_consistency) else None
            row['kNN Consist'] = f"{knn_value:.4f}" if knn_value is not None else "N/A"

            ch_value = calinski_harabasz[last_eval_idx] if last_eval_idx >= 0 and last_eval_idx < len(calinski_harabasz) else None
            row['CH Score'] = f"{ch_value:.1f}" if ch_value is not None else "N/A"

            db_value = davies_bouldin[last_eval_idx] if last_eval_idx >= 0 and last_eval_idx < len(davies_bouldin) else None
            row['DB Score'] = f"{db_value:.4f}" if db_value is not None else "N/A"

            d4_value = cluster_consistency_d4[last_eval_idx] if last_eval_idx >= 0 and last_eval_idx < len(cluster_consistency_d4) else None
            row['D4 Consist'] = f"{d4_value:.4f}" if d4_value is not None else "N/A"

            dist_value = avg_distance_top_k[last_eval_idx] if last_eval_idx >= 0 and last_eval_idx < len(avg_distance_top_k) else None
            row['Avg Dist@k'] = f"{dist_value:.4f}" if dist_value is not None else "N/A"

            comp_value = composite_score[last_eval_idx] if last_eval_idx >= 0 and last_eval_idx < len(composite_score) else None
            row['Composite'] = f"{comp_value:.4f}" if comp_value is not None else "N/A"

            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def get_best_epoch_summary(self):
        """
        Best epoch 정보 추출
        
        Returns:
            dict
        """
        if self.history is None:
            self.load_history()
        
        val_losses = self.history.get('val_loss', [])
        epochs = self.history.get('epoch', [])
        
        if len(val_losses) == 0:
            return None
        
        # Best val loss epoch 찾기
        best_idx = np.argmin(val_losses)
        best_epoch = epochs[best_idx]
        best_val_loss = val_losses[best_idx]
        
        # 해당 epoch의 다른 지표들
        train_losses = self.history.get('train_loss', [])
        feat_stds = self.history.get('feature_std', [])
        avg_cos_sims = self.history.get('avg_cos_sim', [])
        silhouette = self.history.get('silhouette', [])
        
        # 🔹 Helper function: 가장 가까운 evaluation epoch 찾기
        def find_closest_value(metric_list, target_epoch):
            if not metric_list or len(metric_list) == 0:
                return None
            
            eval_epochs = []
            eval_values = []
            for item in metric_list:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    eval_epochs.append(int(item[0]))
                    eval_values.append(item[1])
            
            if len(eval_epochs) == 0:
                return None
            
            closest_idx = min(range(len(eval_epochs)), 
                            key=lambda i: abs(eval_epochs[i] - target_epoch))
            
            return eval_values[closest_idx]
        
        best_info = {
            'Best Epoch': int(best_epoch),
            'Train Loss': f"{train_losses[best_idx]:.4f}" if best_idx < len(train_losses) else "N/A",
            'Val Loss': f"{best_val_loss:.4f}",
            'Feat Std': f"{feat_stds[best_idx]:.4f}" if best_idx < len(feat_stds) else "N/A",
            'Avg Cos Sim': f"{avg_cos_sims[best_idx]:.4f}" if best_idx < len(avg_cos_sims) else "N/A",
        }
        
        # 🔹 Silhouette: 가장 가까운 평가 결과
        sil_value = find_closest_value(silhouette, best_epoch)
        best_info['Silhouette'] = f"{sil_value:.4f}" if sil_value is not None else "N/A"

        # 🆕 추가 지표
        knn_consistency = self.history.get('knn_consistency', [])
        knn_value = find_closest_value(knn_consistency, best_epoch)
        best_info['kNN Consist'] = f"{knn_value:.4f}" if knn_value is not None else "N/A"

        calinski = self.history.get('calinski_harabasz', [])
        ch_value = find_closest_value(calinski, best_epoch)
        best_info['CH Score'] = f"{ch_value:.1f}" if ch_value is not None else "N/A"

        davies = self.history.get('davies_bouldin', [])
        db_value = find_closest_value(davies, best_epoch)
        best_info['DB Score'] = f"{db_value:.4f}" if db_value is not None else "N/A"

        d4_consist = self.history.get('cluster_consistency_d4', [])
        d4_value = find_closest_value(d4_consist, best_epoch)
        best_info['D4 Consist'] = f"{d4_value:.4f}" if d4_value is not None else "N/A"

        avg_dist = self.history.get('avg_distance_top_k', [])
        dist_value = find_closest_value(avg_dist, best_epoch)
        best_info['Avg Dist@k'] = f"{dist_value:.4f}" if dist_value is not None else "N/A"

        composite = self.history.get('composite_score', [])
        comp_value = find_closest_value(composite, best_epoch)
        best_info['Composite'] = f"{comp_value:.4f}" if comp_value is not None else "N/A"

        return best_info
    
    def print_summary(self, interval=10, save_csv=True):
        """
        전체 요약을 콘솔에 출력하고 CSV 저장
        
        Args:
            interval: 몇 epoch마다 보여줄 것인가
            save_csv: CSV로 저장할 것인가
        """
        if self.history is None:
            self.load_history()
        
        print("\n" + "="*120)
        print("📊 TRAINING PROGRESS SUMMARY (Every {} Epochs)".format(interval))
        print("="*120)
        
        # Checkpoint summary
        df_checkpoint = self.get_checkpoint_summary(interval=interval)
        print("\n" + df_checkpoint.to_string(index=False))
        
        # Best epoch
        print("\n" + "-"*120)
        print("🏆 BEST EPOCH (by Val Loss)")
        print("-"*120)
        
        best_info = self.get_best_epoch_summary()
        if best_info:
            for key, value in best_info.items():
                print(f"{key:20s}: {value}")
        
        # 최종 상태
        print("\n" + "-"*120)
        print("✅ FINAL STATUS")
        print("-"*120)
        
        epochs = self.history.get('epoch', [])
        train_losses = self.history.get('train_loss', [])
        val_losses = self.history.get('val_loss', [])
        feat_stds = self.history.get('feature_std', [])
        silhouette = self.history.get('silhouette', [])
        
        # 🔹 Helper function: 가장 가까운 evaluation epoch 찾기
        def find_closest_value(metric_list, target_epoch):
            if not metric_list or len(metric_list) == 0:
                return None
            
            eval_epochs = []
            eval_values = []
            for item in metric_list:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    eval_epochs.append(int(item[0]))
                    eval_values.append(item[1])
            
            if len(eval_epochs) == 0:
                return None
            
            closest_idx = min(range(len(eval_epochs)), 
                            key=lambda i: abs(eval_epochs[i] - target_epoch))
            
            return eval_values[closest_idx]
        
        if len(epochs) > 0:
            final_idx = len(epochs) - 1
            final_epoch = epochs[final_idx]
            
            print(f"Total Epochs          : {int(final_epoch) + 1}")
            print(f"Final Train Loss      : {train_losses[final_idx]:.4f}")
            print(f"Final Val Loss        : {val_losses[final_idx]:.4f}")
            if final_idx < len(feat_stds):
                print(f"Feature Std           : {feat_stds[final_idx]:.4f}")
            
            # 🔹 Silhouette: 가장 가까운 평가 결과
            sil_value = find_closest_value(silhouette, final_epoch)
            if sil_value is not None:
                print(f"Silhouette Score      : {sil_value:.4f}", end="")
                if sil_value >= 0.5:
                    print(" ✅ (목표 달성!)")
                elif sil_value >= 0.3:
                    print(" ⭐ (양호)")
                else:
                    print(" ⚠️  (개선 필요)")

            # 🆕 추가 지표 출력
            knn_consistency = self.history.get('knn_consistency', [])
            knn_value = find_closest_value(knn_consistency, final_epoch)
            if knn_value is not None:
                print(f"kNN Consistency       : {knn_value:.4f}")

            calinski = self.history.get('calinski_harabasz', [])
            ch_value = find_closest_value(calinski, final_epoch)
            if ch_value is not None:
                print(f"Calinski-Harabasz     : {ch_value:.1f}")

            davies = self.history.get('davies_bouldin', [])
            db_value = find_closest_value(davies, final_epoch)
            if db_value is not None:
                print(f"Davies-Bouldin        : {db_value:.4f}")

            d4_consist = self.history.get('cluster_consistency_d4', [])
            d4_value = find_closest_value(d4_consist, final_epoch)
            if d4_value is not None:
                print(f"D4 Consistency        : {d4_value:.4f}")

            avg_dist = self.history.get('avg_distance_top_k', [])
            dist_value = find_closest_value(avg_dist, final_epoch)
            if dist_value is not None:
                print(f"Avg Distance@k        : {dist_value:.4f}")

            composite = self.history.get('composite_score', [])
            comp_value = find_closest_value(composite, final_epoch)
            if comp_value is not None:
                print(f"Composite Score       : {comp_value:.4f}")
        
        print("\n" + "="*120)
        
        # CSV 저장
        if save_csv:
            csv_path = os.path.join(self.log_dir, 'training_summary.csv')
            df_checkpoint.to_csv(csv_path, index=False)
            print(f"\n💾 Summary saved to: {csv_path}")


def generate_training_summary(log_dir='logs', interval=10):
    """
    학습 완료 후 요약 생성 (편의 함수)
    
    Args:
        log_dir: 로그 디렉토리
        interval: 몇 epoch마다 보여줄 것인가
    """
    summary = TrainingSummary(log_dir=log_dir)
    summary.print_summary(interval=interval, save_csv=True)
    
    return summary


if __name__ == "__main__":
    # 테스트
    generate_training_summary(log_dir='logs', interval=10)