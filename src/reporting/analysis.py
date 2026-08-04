import numpy as np
from scipy.stats import wilcoxon

from src.reporting.plotting import create_multidim_visualizations

def analyze_multi_objective_results(df, k_values=None):
    """
    Analyze results comparing single (independent) vs multi (joint) encoding

    Args:
        df:
        k_values: List of k values to analyze separately. If None, analyzes all k together.
    """

    if k_values is None:
        k_values = sorted(df['k'].unique())

    # Analyze for each k separately and combined
    analysis_results = {}

    # 1. Analyze each k separately
    for k in k_values:
        print('\n' + '='*70)
        print('Analysis for {} k features'.format(k))
        print('='*70)
        k_df = df[df['k'] == k]
        analysis_results[f'k_{k}'] = analyze_single_k(k_df)

    # 2. Analyze all k combined
    print('\n' + '='*70)
    print('Analysis for all features')
    print('='*70)
    analysis_results['all_k'] = analyze_single_k(df)

    # 4. Create comprehensive visualizations
    create_multidim_visualizations(df, analysis_results, k_values)

    return analysis_results

def analyze_single_k(df):
    """
    Perform complete analysis for a single k value or combined data
    """
    results = {
        'pareto_fronts': {},
        'coverage_ratio': {},
        'statistical_tests': {},
        'hypervolume': {},
        'accuracy_differences': {}
    }

    # Extract data for both approaches
    single_data = extract_approach_data(df, 'single')
    multi_data = extract_approach_data(df, 'multi')

    # 1. Compute Pareto fronts
    results['pareto_fronts']['single'] = compute_pareto_front(single_data)
    results['pareto_fronts']['multi'] = compute_pareto_front(multi_data)

    # 2. Coverage Ratio: What fraction of approach 2's solutions are dominated by approach 1
    results['coverage_ratio'] = {
        'single_covers_multi': calculate_coverage_ratio(results['pareto_fronts']['single'], results['pareto_fronts']['multi']),
        'multi_covers_single': calculate_coverage_ratio(results['pareto_fronts']['multi'], results['pareto_fronts']['single'])
    }

    # 3. Statistical tests at different memory levels
    results['statistical_tests'] = perform_statistical_analysis(single_data, multi_data)

    # 4. Hypervolume comparison
    results['hypervolume'] = {
        'single': calculate_hypervolume_2d(results['pareto_fronts']['single']),
        'multi': calculate_hypervolume_2d(results['pareto_fronts']['multi'])
    }

    # 5. Accuracy differences between DDoS and classification tasks
    results['accuracy_differences'] = {
        'single': calculate_accuracy_differences(results['pareto_fronts']['single'], single_data),
        'multi': calculate_accuracy_differences(results['pareto_fronts']['multi'], multi_data)
    }

    return results


def extract_approach_data(df, approach='single'):
    """
    Extract accuracy and memory data for a specific approach
    """
    # Filter by method column
    if approach == 'single':
        approach_df = df[df['method'] == 'single']
    else:  # multi
        approach_df = df[df['method'] == 'multi']

    data = []

    for _, row in approach_df.iterrows():

        # Average accuracy across both datasets
        acc_app = row['acc_app']
        acc_ddos = row['acc_ddos']
        avg_accuracy = (acc_app + acc_ddos) / 2

        # Total memory blocks
        memory = row['blocks']

        data.append({
            'accuracy': avg_accuracy,
            'memory': memory,
            'acc_app': acc_app,
            'acc_ddos': acc_ddos,
            'f1_app': row['f1_app'],
            'f1_ddos': row['f1_ddos'],
            'split': row['split'],
            'k': row.get('k', None)
        })

    print('Extracted {} data points for {} from the df with {} rows'.format(len(data), approach, approach_df.shape[0]))

    return data


def compute_pareto_front(data):
    """
    Compute Pareto front from a list of solutions
    """
    # Extract unique points (accuracy, memory)
    points = [(d['accuracy'], d['memory']) for d in data]
    unique_points = list(set(points))

    pareto_front = []

    for i, (acc1, mem1) in enumerate(unique_points):
        dominated = False
        for j, (acc2, mem2) in enumerate(unique_points):
            if i != j:
                # Check if point j dominates point i
                if acc2 >= acc1 and mem2 <= mem1 and (acc2 > acc1 or mem2 < mem1):
                    dominated = True
                    break

        if not dominated:
            pareto_front.append((acc1, mem1))

    # Sort by memory for easier visualization
    return sorted(pareto_front, key=lambda x: x[1])


def calculate_accuracy_differences(pareto_front, data):
    """
    Calculate statistics on |Acc_DDoS - Acc_classification| for Pareto front points

    Args:
        pareto_front: List of (accuracy, memory) tuples representing the Pareto front
        data: List of data dictionaries with 'acc_app', 'acc_ddos', 'accuracy', 'memory'

    Returns:
        Dictionary with 'avg', 'max', and 'n_points' statistics
    """
    if not pareto_front:
        return {'avg': 0.0, 'max': 0.0, 'n_points': 0}

    # Create a mapping from (accuracy, memory) to data points
    point_to_data = {}
    for d in data:
        key = (d['accuracy'], d['memory'])
        if key not in point_to_data:
            point_to_data[key] = []
        point_to_data[key].append(d)

    # Calculate accuracy differences for Pareto front points
    acc_diffs = []
    for acc, mem in pareto_front:
        key = (acc, mem)
        if key in point_to_data:
            # If multiple data points map to same (accuracy, memory), use all of them
            for d in point_to_data[key]:
                diff = abs(d['acc_ddos'] - d['acc_app'])
                acc_diffs.append(diff)

    if not acc_diffs:
        return {'avg': 0.0, 'max': 0.0, 'n_points': 0}

    return {
        'avg': np.mean(acc_diffs),
        'max': np.max(acc_diffs),
        'n_points': len(pareto_front)
    }


def calculate_coverage_ratio(data1, data2):
    """
    Calculate what fraction of data2 solutions are dominated by at least one solution in data1
    """
    dominated_count = 0
    
    for acc2, mem2 in data2:
        
        # Check if any solution in data1 dominates this solution
        for acc1, mem1 in data1:
            
            if acc1 >= acc2 and mem1 <= mem2 and (acc1 > acc2 or mem1 < mem2):
                dominated_count += 1
                break
    
    return dominated_count / len(data2) if data2 else 0


def perform_statistical_analysis(data1, data2):
    """
    Perform statistical tests comparing the two approaches
    """
    results = {}

    # 1. Overall comparison - match by (split, k) for paired test
    # Create lookup dictionaries
    data1_lookup = {(d['split'], d['k']): d['accuracy'] for d in data1}
    data2_lookup = {(d['split'], d['k']): d['accuracy'] for d in data2}

    # Find common (split, k) pairs
    common_keys = set(data1_lookup.keys()) & set(data2_lookup.keys())

    if len(common_keys) > 0:
        acc1 = [data1_lookup[key] for key in sorted(common_keys)]
        acc2 = [data2_lookup[key] for key in sorted(common_keys)]

        differences = np.array(acc2) - np.array(acc1)
        print(f"Paired sample size (common split/k pairs): {len(differences)}")
        print(f"Median difference: {np.median(differences)}")
        print(f"Percentage positive: {np.mean(differences > 0) * 100:.1f}%")
        print(f"Percentage negative: {np.mean(differences < 0) * 100:.1f}%")
        print(f"Percentage zero: {np.mean(differences == 0) * 100:.1f}%")
        print(f"Range of differences: {np.min(differences):.6f} to {np.max(differences):.6f}")

        stat, p_value = wilcoxon(acc1, acc2)
        results['overall'] = {
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'median_diff': np.median(differences),
            'mean_diff': np.mean(differences),
            'n_paired': len(common_keys)
        }

        print("p-value: {}".format(p_value))
        print(stat)
    else:
        print("No common (split, k) pairs found for paired comparison")
        results['overall'] = None
    
    # 2. Comparison at specific memory levels
    memory_levels = [5, 10, 25, 50]
    results['by_memory'] = {}
    
    for budget in memory_levels:
        valid1 = [d['accuracy'] for d in data1 if d['memory'] <= budget]
        valid2 = [d['accuracy'] for d in data2 if d['memory'] <= budget]
        
        if valid1 and valid2:
            results['by_memory'][budget] = {
                'n_single': len(valid1),
                'n_multi': len(valid2),
                'median_single': np.median(valid1),
                'median_multi': np.median(valid2),
                'max_single': np.max(valid1),
                'max_multi': np.max(valid2)
            }
    
    return results


def calculate_hypervolume_2d(pareto_front, ref_point=(0.5, 100)):
    if not pareto_front:
        return 0
    
    # Filter to points that dominate the reference point
    # (higher accuracy, lower memory)
    valid = [(acc, mem) for acc, mem in pareto_front 
             if acc > ref_point[0] and mem < ref_point[1]]
    
    if not valid:
        return 0
    
    # Sort by memory ascending
    sorted_front = sorted(valid, key=lambda x: x[1])
    
    hv = 0
    prev_mem = ref_point[1]
    
    for acc, mem in reversed(sorted_front):
        width = prev_mem - mem
        height = acc - ref_point[0]
        hv += width * height
        prev_mem = mem
    
    return hv