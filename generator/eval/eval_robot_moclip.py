"""
Evaluation functions for robot motion generation with MoCLIP metrics.
"""
from datetime import datetime
import numpy as np
import torch
from utils.metrics import *
from collections import OrderedDict
from eval.motion_safety_score import evaluate_mss_batch, average_mss_results
from eval.root_trajectory_consistency import (
    extract_root_trajectory_from_38d, extract_root_trajectory_raw,
    calculate_rtc_single
)


def evaluate_matching_score_moclip(eval_wrapper, motion_loaders, file):
    """
    Evaluate matching score using MoCLIP.
    
    Args:
        eval_wrapper: MoCLIPEvaluatorWrapper instance
        motion_loaders: Dict of motion loaders
        file: Log file handle
    
    Returns:
        match_score_dict: Matching scores
        R_precision_dict: R-precision scores
        activation_dict: Motion embeddings
    """
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    
    print('========== Evaluating Matching Score (MoCLIP) ==========')
    
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                # MoCLIP expects: captions (List[str]), motions, m_lens
                captions, motions, m_lens = batch
                
                # Get embeddings from MoCLIP
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    captions=captions,
                    motions=motions,
                    m_lens=m_lens
                )
                
                # Compute distance matrix
                dist_mat = euclidean_distance_matrix(
                    text_embeddings.cpu().numpy(),
                    motion_embeddings.cpu().numpy()
                )
                matching_score_sum += dist_mat.trace()
                
                # Compute R-precision
                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)
                
                all_size += text_embeddings.shape[0]
                all_motion_embeddings.append(motion_embeddings.cpu().numpy())
        
        # Aggregate results
        all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
        matching_score = matching_score_sum / all_size
        R_precision = top_k_count / all_size
        
        match_score_dict[motion_loader_name] = matching_score
        R_precision_dict[motion_loader_name] = R_precision
        activation_dict[motion_loader_name] = all_motion_embeddings
        
        # Log results
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', 
              file=file, flush=True)
        
        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)
    
    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid_moclip(eval_wrapper, groundtruth_loader, activation_dict, file):
    """
    Evaluate FID using MoCLIP.
    
    Args:
        eval_wrapper: MoCLIPEvaluatorWrapper instance
        groundtruth_loader: Ground truth data loader
        activation_dict: Dict of motion embeddings from generated data
        file: Log file handle
    
    Returns:
        eval_dict: FID scores
    """
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    
    print('========== Evaluating FID (MoCLIP) ==========')
    
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            captions, motions, m_lens = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(motions, m_lens)
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)
    
    # Compute FID for each model
    for model_name, motion_embeddings in activation_dict.items():
        if model_name == 'ground truth':
            continue
        
        mu, cov = calculate_activation_statistics(motion_embeddings)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        
        eval_dict[model_name] = fid
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
    
    return eval_dict


def evaluate_diversity_moclip(activation_dict, file, diversity_times):
    """
    Evaluate diversity using motion embeddings.
    
    Args:
        activation_dict: Dict of motion embeddings
        file: Log file handle
        diversity_times: Number of samples for diversity calculation
    
    Returns:
        eval_dict: Diversity scores
    """
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity (MoCLIP) ==========')
    
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    
    return eval_dict


def evaluate_multimodality_moclip(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    """
    Evaluate multimodality using MoCLIP.
    
    Args:
        eval_wrapper: MoCLIPEvaluatorWrapper instance
        mm_motion_loaders: Dict of multimodality motion loaders
        file: Log file handle
        mm_num_times: Number of samples for multimodality calculation
    
    Returns:
        eval_dict: Multimodality scores
    """
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality (MoCLIP) ==========')
    
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, T, 38)
                motions, m_lens = batch
                motion_embeddings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embeddings.unsqueeze(0))
        
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        
        eval_dict[model_name] = multimodality
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
    
    return eval_dict


def evaluate_mss_moclip(motion_loaders, dataset_mean, dataset_std, file, fps=50):
    """
    Evaluate Motion Safety Score using MSS.
    
    Args:
        motion_loaders: Dict of motion loaders
        dataset_mean: (38,) - Dataset mean for denormalization
        dataset_std: (38,) - Dataset std for denormalization
        file: Log file handle
        fps: int - Frame rate for velocity/acceleration calculation
    
    Returns:
        eval_dict: MSS scores with sub-scores
    """
    eval_dict = OrderedDict({})
    print('========== Evaluating Motion Safety Score (MSS) ==========')
    print('========== Evaluating Motion Safety Score (MSS) ==========', file=file, flush=True)
    
    for model_name, motion_loader in motion_loaders.items():
        mss_scores = []
        
        for idx, batch in enumerate(motion_loader):
            # MoCLIP loader returns: captions, motions, m_lens
            captions, motions, m_lens = batch
            
            # Convert to numpy if needed
            if torch.is_tensor(motions):
                motions = motions.cpu().numpy()
            if torch.is_tensor(m_lens):
                m_lens = m_lens.cpu().numpy()
            
            # Evaluate MSS for this batch
            batch_mss = evaluate_mss_batch(motions, m_lens, dataset_mean, dataset_std, fps)
            mss_scores.append(batch_mss)
        
        # Aggregate all batches
        avg_mss = average_mss_results(mss_scores)
        eval_dict[model_name] = avg_mss
        
        # Log results
        print(f'---> [{model_name}] MSS: {avg_mss["mss"]:.4f} '
              f'(Pos: {avg_mss["pos_score"]:.4f}, '
              f'Vel: {avg_mss["vel_score"]:.4f}, '
              f'Acc: {avg_mss["acc_score"]:.4f})\n'
              f'      Violation Rate - Pos: {avg_mss["pos_rate"]:.4%}, '
              f'Vel: {avg_mss["vel_rate"]:.4%}, '
              f'Acc: {avg_mss["acc_rate"]:.4%}')
        
        print(f'---> [{model_name}] MSS: {avg_mss["mss"]:.4f} '
              f'(Pos: {avg_mss["pos_score"]:.4f}, '
              f'Vel: {avg_mss["vel_score"]:.4f}, '
              f'Acc: {avg_mss["acc_score"]:.4f})\n'
              f'      Violation Rate - Pos: {avg_mss["pos_rate"]:.4%}, '
              f'Vel: {avg_mss["vel_rate"]:.4%}, '
              f'Acc: {avg_mss["acc_rate"]:.4%}', 
              file=file, flush=True)
    
    return eval_dict


def evaluate_rtc_moclip(motion_loaders, dataset, dataset_mean, dataset_std, file):
    """
    Evaluate Root Trajectory Consistency (RTC).

    Matches generated motions to ground truth by caption and computes trajectory consistency.
    """
    eval_dict = OrderedDict({})
    print('========== Evaluating Root Trajectory Consistency (RTC) ==========')
    print('========== Evaluating Root Trajectory Consistency (RTC) ==========', file=file, flush=True)

    # Build caption -> GT raw motion mapping from the underlying RobotMotionDataset
    gt_ds = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    caption_to_gt = {}
    for name in gt_ds.name_list:
        data = gt_ds.data_dict[name]
        raw_motion = data['motion']  # un-normalized (T, 38)
        length = data['length']
        for text_entry in data['text']:
            cap = text_entry['caption']
            if cap not in caption_to_gt:
                caption_to_gt[cap] = (raw_motion, length)

    print(f'Built caption-to-GT mapping: {len(caption_to_gt)} unique captions')

    # Compute RTC for each generated model
    for model_name, motion_loader in motion_loaders.items():
        if model_name == 'ground truth':
            continue
        
        all_rtc = []
        all_shape = []
        all_extent = []
        matched = 0
        total = 0
        
        for idx, batch in enumerate(motion_loader):
            captions, motions, m_lens = batch
            
            if torch.is_tensor(motions):
                motions = motions.cpu().numpy()
            if torch.is_tensor(m_lens):
                m_lens = m_lens.cpu().numpy()
            
            batch_size = motions.shape[0]
            
            for i in range(batch_size):
                total += 1
                caption = captions[i]
                
                # 通过 caption 查找对应的 GT
                gt_info = caption_to_gt.get(caption)
                if gt_info is None:
                    continue
                
                gt_raw_motion, gt_length = gt_info
                
                # 生成动作：已归一化 → 用 extract_root_trajectory_from_38d 反归一化后提取
                gen_motion = motions[i, :int(m_lens[i]), :]  # (T_gen, 38) 归一化
                traj_gen = extract_root_trajectory_from_38d(gen_motion, dataset_mean, dataset_std)
                
                # GT 动作：未归一化 → 直接用 extract_root_trajectory_raw 提取
                gt_motion = gt_raw_motion[:gt_length, :]  # (T_gt, 38) 未归一化
                traj_gt = extract_root_trajectory_raw(gt_motion)
                
                # 计算 RTC
                scores = calculate_rtc_single(traj_gen, traj_gt)
                all_rtc.append(scores['rtc'])
                all_shape.append(scores['shape_score'])
                all_extent.append(scores['extent_score'])
                matched += 1
        
        # 聚合结果
        if matched == 0:
            avg_rtc = {'rtc': 0.0, 'shape_score': 0.0, 'extent_score': 0.0}
        else:
            avg_rtc = {
                'rtc': np.mean(all_rtc),
                'shape_score': np.mean(all_shape),
                'extent_score': np.mean(all_extent)
            }
        
        eval_dict[model_name] = avg_rtc
        
        # 输出结果
        print(f'---> [{model_name}] RTC: {avg_rtc["rtc"]:.4f} '
              f'(Shape: {avg_rtc["shape_score"]:.4f}, '
              f'Extent: {avg_rtc["extent_score"]:.4f}) '
              f'[matched {matched}/{total}]')
        
        print(f'---> [{model_name}] RTC: {avg_rtc["rtc"]:.4f} '
              f'(Shape: {avg_rtc["shape_score"]:.4f}, '
              f'Extent: {avg_rtc["extent_score"]:.4f}) '
              f'[matched {matched}/{total}]',
              file=file, flush=True)
    
    return eval_dict


def get_metric_statistics(values, replication_times):
    """Calculate mean and confidence interval."""
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation_moclip(eval_wrapper, gt_loader, eval_motion_loaders, log_file, 
                      replication_times, diversity_times, mm_num_times, 
                      dataset=None, run_mm=False):
    """
    Main evaluation function for MoCLIP.
    
    Args:
        eval_wrapper: MoCLIPEvaluatorWrapper instance
        gt_loader: Ground truth data loader
        eval_motion_loaders: Dict of motion loader getters
        log_file: Path to log file
        replication_times: Number of evaluation replications
        diversity_times: Number of samples for diversity
        mm_num_times: Number of samples for multimodality
        dataset: Dataset object (for MSS mean/std)
        run_mm: Whether to run multimodality evaluation
    
    Returns:
        all_metrics: Dict of all evaluation metrics
    """
    with open(log_file, 'a') as f:
        all_metrics = OrderedDict({
            'Matching Score': OrderedDict({}),
            'R_precision': OrderedDict({}),
            'FID': OrderedDict({}),
            'Diversity': OrderedDict({}),
            'MultiModality': OrderedDict({}),
            'MSS': OrderedDict({}),
            'RTC': OrderedDict({})
        })
        
        for replication in range(replication_times):
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            
            motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            mm_motion_loaders = {}
            
            # Get generated motion loaders
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader, eval_generate_time = motion_loader_getter()
                print(f'---> [{motion_loader_name}] batch_generate_time: {eval_generate_time}s', 
                      file=f, flush=True)
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader
            
            if replication_times > 1:
                print(f'==================== Replication {replication} ====================')
                print(f'==================== Replication {replication} ====================', 
                      file=f, flush=True)
            
            # Evaluate matching score
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score_moclip(
                eval_wrapper, motion_loaders, f
            )
            
            # Evaluate FID
            fid_score_dict = evaluate_fid_moclip(eval_wrapper, gt_loader, acti_dict, f)
            
            # Evaluate diversity
            div_score_dict = evaluate_diversity_moclip(acti_dict, f, diversity_times)
            
            # Evaluate multimodality
            if run_mm:
                mm_score_dict = evaluate_multimodality_moclip(
                    eval_wrapper, mm_motion_loaders, f, mm_num_times
                )
            
            # Evaluate Motion Safety Score
            # RobotEvalDataset wraps RobotMotionDataset: use dataset.dataset.mean/std
            if dataset is not None:
                ds = dataset.dataset if hasattr(dataset, 'dataset') else dataset
                mss_score_dict = evaluate_mss_moclip(
                    motion_loaders, ds.mean, ds.std, f, fps=50
                )
            
            # Evaluate Root Trajectory Consistency
            if dataset is not None:
                ds = dataset.dataset if hasattr(dataset, 'dataset') else dataset
                rtc_score_dict = evaluate_rtc_moclip(
                    motion_loaders, dataset, ds.mean, ds.std, f
                )
            
            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)
            
            # Aggregate metrics
            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key].append(item)
            
            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key].append(item)
            
            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key].append(item)
            
            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key].append(item)
            
            if run_mm:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics['MultiModality']:
                        all_metrics['MultiModality'][key] = [item]
                    else:
                        all_metrics['MultiModality'][key].append(item)
            
            if dataset is not None:
                for key, item in mss_score_dict.items():
                    if key not in all_metrics['MSS']:
                        all_metrics['MSS'][key] = [item]
                    else:
                        all_metrics['MSS'][key].append(item)
            
            if dataset is not None:
                for key, item in rtc_score_dict.items():
                    if key not in all_metrics['RTC']:
                        all_metrics['RTC'][key] = [item]
                    else:
                        all_metrics['RTC'][key].append(item)
        
        # Print final statistics (to stdout and log file)
        print('=' * 80)
        print(f'Final Results (averaged over {replication_times} replications):')
        print('=' * 80)
        print('=' * 80, file=f, flush=True)
        print(f'Final Results (averaged over {replication_times} replications):', file=f, flush=True)
        print('=' * 80, file=f, flush=True)
        
        for metric_name, metric_dict in all_metrics.items():
            print(f'\n{metric_name}:')
            print(f'\n{metric_name}:', file=f, flush=True)
            for model_name, values in metric_dict.items():
                # Special handling for MSS (dict with sub-scores)
                if metric_name == 'MSS' and len(values) > 0 and isinstance(values[0], dict):
                    # Extract each sub-score
                    mss_vals = [v['mss'] for v in values]
                    pos_vals = [v['pos_score'] for v in values]
                    vel_vals = [v['vel_score'] for v in values]
                    acc_vals = [v['acc_score'] for v in values]
                    r_pos_vals = [v.get('pos_rate', 0.0) for v in values]
                    r_vel_vals = [v.get('vel_rate', 0.0) for v in values]
                    r_acc_vals = [v.get('acc_rate', 0.0) for v in values]
                    
                    mss_mean, mss_conf = get_metric_statistics(np.array(mss_vals), replication_times)
                    pos_mean, pos_conf = get_metric_statistics(np.array(pos_vals), replication_times)
                    vel_mean, vel_conf = get_metric_statistics(np.array(vel_vals), replication_times)
                    acc_mean, acc_conf = get_metric_statistics(np.array(acc_vals), replication_times)
                    r_pos_mean, r_pos_conf = get_metric_statistics(np.array(r_pos_vals), replication_times)
                    r_vel_mean, r_vel_conf = get_metric_statistics(np.array(r_vel_vals), replication_times)
                    r_acc_mean, r_acc_conf = get_metric_statistics(np.array(r_acc_vals), replication_times)
                    
                    line = (f'  [{model_name}] MSS: {mss_mean:.4f}±{mss_conf:.4f} '
                            f'(Pos: {pos_mean:.4f}±{pos_conf:.4f}, '
                            f'Vel: {vel_mean:.4f}±{vel_conf:.4f}, '
                            f'Acc: {acc_mean:.4f}±{acc_conf:.4f})\n'
                            f'      Violation Rate - Pos: {r_pos_mean:.4%}±{r_pos_conf:.4%}, '
                            f'Vel: {r_vel_mean:.4%}±{r_vel_conf:.4%}, '
                            f'Acc: {r_acc_mean:.4%}±{r_acc_conf:.4%}')
                    print(line)
                    print(line, file=f, flush=True)
                # Special handling for RTC (dict with sub-scores)
                elif metric_name == 'RTC' and len(values) > 0 and isinstance(values[0], dict):
                    # Extract each sub-score
                    rtc_vals = [v['rtc'] for v in values]
                    shape_vals = [v['shape_score'] for v in values]
                    extent_vals = [v['extent_score'] for v in values]
                    
                    rtc_mean, rtc_conf = get_metric_statistics(np.array(rtc_vals), replication_times)
                    shape_mean, shape_conf = get_metric_statistics(np.array(shape_vals), replication_times)
                    extent_mean, extent_conf = get_metric_statistics(np.array(extent_vals), replication_times)
                    
                    line = (f'  [{model_name}] RTC: {rtc_mean:.4f}±{rtc_conf:.4f} '
                            f'(Shape: {shape_mean:.4f}±{shape_conf:.4f}, '
                            f'Extent: {extent_mean:.4f}±{extent_conf:.4f})')
                    print(line)
                    print(line, file=f, flush=True)
                else:
                    mean, conf = get_metric_statistics(np.array(values), replication_times)
                    if isinstance(mean, np.ndarray):
                        line = f'  [{model_name}] mean: {mean}, conf: {conf}'
                        print(line)
                        print(line, file=f, flush=True)
                    else:
                        line = f'  [{model_name}] mean: {mean:.4f}, conf: {conf:.4f}'
                        print(line)
                        print(line, file=f, flush=True)
        
        return all_metrics

