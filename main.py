#!/usr/bin/env python3
"""
Minimized Optimization Caller

This module runs both genetic algorithm and integer programming optimization
and creates shared comparison plots.

Usage:
    python main.py [genetic|ip|both]
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import threading

# Clean imports using wrapper modules
from genetic.genetic_algorithm import run_genetic_algorithm
from integer.integer_programming import run_integer_programming


def create_shared_comparison_plots(ip_results, ga_results):
    """Create comprehensive comparison plots for both algorithms."""
    # Create main comparison plot (2x2)
    fig1 = create_basic_comparison_plots(ip_results, ga_results)

    # Create additional detailed analysis plots
    fig2 = create_detailed_analysis_plots(ip_results, ga_results)

    # Create convergence and performance analysis plots
    fig3 = create_performance_analysis_plots(ip_results, ga_results)

    return fig1, fig2, fig3


def create_basic_comparison_plots(ip_results, ga_results):
    """Create basic 2x2 comparison plots."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Cost Comparison
    ip_details = ip_results['solution_details']
    ga_details = ga_results['solution_details']

    categories = ['Infrastructure', 'Penalty', 'Total']
    ip_costs = [
        ip_details['total_cost'],
        ip_details.get('total_penalty', 0),
        ip_details['total_cost'] + ip_details.get('total_penalty', 0)
    ]
    ga_costs = [
        ga_details['total_cost'],
        ga_details['total_penalty'],
        ga_details['total_cost'] + ga_details['total_penalty']
    ]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.bar(x - width/2, ip_costs, width,
                    label='Integer Programming', alpha=0.8, color='#2E86AB')
    bars2 = ax1.bar(x + width/2, ga_costs, width,
                    label='Genetic Algorithm', alpha=0.8, color='#A23B72')

    ax1.set_title('Cost Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cost ($)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                     f'${height:,.0f}', ha='center', va='bottom', fontsize=8, rotation=90)

    # 2. GA Evolution vs IP Solution
    if ga_results.get('fitness_history'):
        generations = list(range(1, len(ga_results['fitness_history']) + 1))
        ax2.plot(generations, ga_results['fitness_history'],
                 'o-', linewidth=2, color='#A23B72', markersize=4, label='GA Evolution')
        ax2.axhline(y=ip_results['best_fitness'], color='#2E86AB',
                    linestyle='--', linewidth=3, label='IP Optimal')
        ax2.fill_between(generations, ga_results['fitness_history'],
                         alpha=0.3, color='#A23B72')
        ax2.set_title('Solution Quality Convergence',
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness (Lower is Better)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. Performance Metrics
    metrics = ['Utilization %', 'Violations', 'Time (s)']
    ip_metrics = [
        ip_details['utilization_stats']['network_overall'],
        ip_details['capacity_violations'],
        ip_results['total_time']
    ]
    ga_metrics = [
        ga_details['utilization_stats']['network_overall'],
        ga_details['capacity_violations'],
        ga_results['total_time']
    ]

    x = np.arange(len(metrics))
    bars1 = ax3.bar(x - width/2, ip_metrics, width,
                    label='IP', alpha=0.8, color='#2E86AB')
    bars2 = ax3.bar(x + width/2, ga_metrics, width,
                    label='GA', alpha=0.8, color='#A23B72')

    ax3.set_title('Performance Metrics', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Value')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    # 4. Module Selection Comparison
    ip_modules = {}
    ga_modules = {}

    for link in ip_details['link_details']:
        choice = link['module_info']['choice_description']
        ip_modules[choice] = ip_modules.get(choice, 0) + 1

    for link in ga_details['link_details']:
        choice = link['module_info']['choice_description']
        ga_modules[choice] = ga_modules.get(choice, 0) + 1

    all_modules = set(list(ip_modules.keys()) + list(ga_modules.keys()))
    module_names = list(all_modules)

    ip_counts = [ip_modules.get(name, 0) for name in module_names]
    ga_counts = [ga_modules.get(name, 0) for name in module_names]

    x = np.arange(len(module_names))
    bars1 = ax4.bar(x - width/2, ip_counts, width,
                    label='IP', alpha=0.8, color='#2E86AB')
    bars2 = ax4.bar(x + width/2, ga_counts, width,
                    label='GA', alpha=0.8, color='#A23B72')

    ax4.set_title('Module Selection Distribution',
                  fontsize=14, fontweight='bold')
    ax4.set_ylabel('Number of Links')
    ax4.set_xticks(x)
    ax4.set_xticklabels(module_names, rotation=45, ha='right')
    ax4.legend()

    plt.tight_layout()
    plt.suptitle(
        'Algorithm Comparison: Integer Programming vs Genetic Algorithm', fontsize=16, y=0.98)
    plt.savefig('optimization_comparison_basic.png',
                dpi=300, bbox_inches='tight')
    print("📊 Basic comparison plot saved as 'optimization_comparison_basic.png'")

    return fig


def create_detailed_analysis_plots(ip_results, ga_results):
    """Create detailed analysis plots with capacity utilization, cost breakdown, and network topology insights."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    ip_details = ip_results['solution_details']
    ga_details = ga_results['solution_details']

    # 1. Link Capacity Utilization Distribution
    ip_utilizations = [link['utilization_percent']
                       for link in ip_details['link_details']]
    ga_utilizations = [link['utilization_percent']
                       for link in ga_details['link_details']]

    bins = np.linspace(0, 100, 21)  # 0-100% in 5% increments
    ax1.hist(ip_utilizations, bins=bins, alpha=0.7,
             label='IP', color='#2E86AB', density=True)
    ax1.hist(ga_utilizations, bins=bins, alpha=0.7,
             label='GA', color='#A23B72', density=True)
    ax1.axvline(np.mean(ip_utilizations), color='#2E86AB', linestyle='--',
                label=f'IP Mean: {np.mean(ip_utilizations):.1f}%')
    ax1.axvline(np.mean(ga_utilizations), color='#A23B72', linestyle='--',
                label=f'GA Mean: {np.mean(ga_utilizations):.1f}%')
    ax1.set_title('Link Utilization Distribution',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Utilization (%)')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Cost Efficiency Analysis (Cost per Unit Capacity)
    ip_cost_efficiency = []
    ga_cost_efficiency = []

    for link in ip_details['link_details']:
        if link['total_capacity'] > 0:
            # For IP, use module_info total_cost if available, otherwise use link total_cost
            cost = link['module_info'].get('total_cost', link['total_cost'])
            efficiency = cost / link['total_capacity']
            ip_cost_efficiency.append(efficiency)

    for link in ga_details['link_details']:
        if link['total_capacity'] > 0:
            # For GA, use link total_cost since module_info doesn't have total_cost
            cost = link['total_cost']
            efficiency = cost / link['total_capacity']
            ga_cost_efficiency.append(efficiency)

    ax2.scatter(range(len(ip_cost_efficiency)), sorted(ip_cost_efficiency),
                alpha=0.7, color='#2E86AB', label='IP', s=30)
    ax2.scatter(range(len(ga_cost_efficiency)), sorted(ga_cost_efficiency),
                alpha=0.7, color='#A23B72', label='GA', s=30)
    ax2.set_title('Cost Efficiency per Link\n(Cost per Unit Capacity)',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Link Index (sorted by efficiency)')
    ax2.set_ylabel('Cost per Unit Capacity ($/Mbps)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Capacity vs Demand Analysis
    ip_capacities = [link['total_capacity']
                     for link in ip_details['link_details']]
    ip_demands = [link['demand'] for link in ip_details['link_details']]
    ga_capacities = [link['total_capacity']
                     for link in ga_details['link_details']]
    ga_demands = [link.get('traffic_load', link.get('demand', 0))
                  for link in ga_details['link_details']]

    ax3.scatter(ip_demands, ip_capacities, alpha=0.6, color='#2E86AB',
                label='IP', s=40, edgecolors='white', linewidth=0.5)
    ax3.scatter(ga_demands, ga_capacities, alpha=0.6, color='#A23B72',
                label='GA', s=40, edgecolors='white', linewidth=0.5)

    # Add diagonal line showing perfect capacity matching
    max_demand = max(max(ip_demands), max(ga_demands))
    ax3.plot([0, max_demand], [0, max_demand], 'k--', alpha=0.5,
             label='Perfect Match Line')

    ax3.set_title('Capacity vs Demand Allocation',
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Traffic Demand (Mbps)')
    ax3.set_ylabel('Allocated Capacity (Mbps)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Module Cost vs Capacity Trade-off
    module_costs_ip = {}
    module_capacities_ip = {}
    module_costs_ga = {}
    module_capacities_ga = {}

    for link in ip_details['link_details']:
        module_type = link['module_info']['choice_description']
        cost = link['module_info'].get('total_cost', link['total_cost'])
        capacity = link['total_capacity']

        if module_type not in module_costs_ip:
            module_costs_ip[module_type] = []
            module_capacities_ip[module_type] = []
        module_costs_ip[module_type].append(cost)
        module_capacities_ip[module_type].append(capacity)

    for link in ga_details['link_details']:
        module_type = link['module_info']['choice_description']
        cost = link['total_cost']
        capacity = link['total_capacity']

        if module_type not in module_costs_ga:
            module_costs_ga[module_type] = []
            module_capacities_ga[module_type] = []
        module_costs_ga[module_type].append(cost)
        module_capacities_ga[module_type].append(capacity)

    colors = plt.cm.Set3(np.linspace(
        0, 1, len(set(list(module_costs_ip.keys()) + list(module_costs_ga.keys())))))
    color_map = {module: colors[i] for i, module in enumerate(
        set(list(module_costs_ip.keys()) + list(module_costs_ga.keys())))}

    for module_type in module_costs_ip:
        ax4.scatter(module_capacities_ip[module_type], module_costs_ip[module_type],
                    color=color_map[module_type], alpha=0.8, s=60,
                    marker='^', label=f'IP-{module_type}', edgecolors='black', linewidth=0.5)

    for module_type in module_costs_ga:
        ax4.scatter(module_capacities_ga[module_type], module_costs_ga[module_type],
                    color=color_map[module_type], alpha=0.8, s=60,
                    marker='o', label=f'GA-{module_type}', edgecolors='black', linewidth=0.5)

    ax4.set_title('Module Cost vs Capacity Trade-off',
                  fontsize=14, fontweight='bold')
    ax4.set_xlabel('Total Capacity (Mbps)')
    ax4.set_ylabel('Total Cost ($)')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Detailed Network Analysis: Utilization, Efficiency & Trade-offs',
                 fontsize=16, y=0.98)
    plt.savefig('optimization_detailed_analysis.png',
                dpi=300, bbox_inches='tight')
    print("📊 Detailed analysis plot saved as 'optimization_detailed_analysis.png'")
    plt.show()

    return fig


def create_performance_analysis_plots(ip_results, ga_results):
    """Create performance and convergence analysis plots."""
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)
          ) = plt.subplots(2, 3, figsize=(24, 12))

    ip_details = ip_results['solution_details']
    ga_details = ga_results['solution_details']

    # 1. Solution Quality Gap Analysis
    if ga_results.get('fitness_history'):
        generations = list(range(1, len(ga_results['fitness_history']) + 1))
        ip_optimal = ip_results['best_fitness']
        ga_fitness = ga_results['fitness_history']

        # Calculate gap percentage
        gap_percentage = [(fitness - ip_optimal) /
                          ip_optimal * 100 for fitness in ga_fitness]

        ax1.plot(generations, gap_percentage, 'o-', color='#A23B72',
                 linewidth=2, markersize=4, label='Optimality Gap')
        ax1.axhline(y=0, color='#2E86AB', linestyle='--',
                    linewidth=2, label='IP Optimal (0% gap)')
        ax1.fill_between(generations, gap_percentage,
                         0, alpha=0.3, color='#A23B72')

        ax1.set_title('GA Convergence: Optimality Gap Over Time',
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Gap from IP Optimal (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. Algorithm Robustness Analysis (Violation Distribution)
    violation_types = ['Over-utilized Links',
                       'Under-utilized Links (<10%)', 'Well-utilized Links (10-80%)']

    ip_violations = [
        sum(1 for link in ip_details['link_details']
            if link['utilization_percent'] > 100),
        sum(1 for link in ip_details['link_details']
            if link['utilization_percent'] < 10),
        sum(1 for link in ip_details['link_details']
            if 10 <= link['utilization_percent'] <= 80)
    ]

    ga_violations = [
        sum(1 for link in ga_details['link_details']
            if link['utilization_percent'] > 100),
        sum(1 for link in ga_details['link_details']
            if link['utilization_percent'] < 10),
        sum(1 for link in ga_details['link_details']
            if 10 <= link['utilization_percent'] <= 80)
    ]

    x = np.arange(len(violation_types))
    width = 0.35

    bars1 = ax2.bar(x - width/2, ip_violations, width, label='IP',
                    alpha=0.8, color='#2E86AB')
    bars2 = ax2.bar(x + width/2, ga_violations, width, label='GA',
                    alpha=0.8, color='#A23B72')

    ax2.set_title('Network Utilization Quality',
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Links')
    ax2.set_xticks(x)
    ax2.set_xticklabels(violation_types, rotation=15)
    ax2.legend()

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=10)

    # 3. Resource Allocation Efficiency
    ip_total_capacity = sum(link['total_capacity']
                            for link in ip_details['link_details'])
    ip_total_demand = sum(link['demand']
                          for link in ip_details['link_details'])
    ga_total_capacity = sum(link['total_capacity']
                            for link in ga_details['link_details'])
    ga_total_demand = sum(link.get('traffic_load', link.get(
        'demand', 0)) for link in ga_details['link_details'])

    efficiency_metrics = ['Total Capacity', 'Total Demand', 'Efficiency Ratio']
    ip_efficiency = [ip_total_capacity, ip_total_demand,
                     ip_total_demand/ip_total_capacity if ip_total_capacity > 0 else 0]
    ga_efficiency = [ga_total_capacity, ga_total_demand,
                     ga_total_demand/ga_total_capacity if ga_total_capacity > 0 else 0]

    # Normalize for better visualization
    max_capacity = max(ip_total_capacity, ga_total_capacity)
    ip_efficiency_norm = [ip_efficiency[0]/max_capacity,
                          ip_efficiency[1]/max_capacity, ip_efficiency[2]]
    ga_efficiency_norm = [ga_efficiency[0]/max_capacity,
                          ga_efficiency[1]/max_capacity, ga_efficiency[2]]

    x = np.arange(len(efficiency_metrics))
    bars1 = ax3.bar(x - width/2, ip_efficiency_norm, width, label='IP',
                    alpha=0.8, color='#2E86AB')
    bars2 = ax3.bar(x + width/2, ga_efficiency_norm, width, label='GA',
                    alpha=0.8, color='#A23B72')

    ax3.set_title('Resource Allocation Efficiency',
                  fontsize=14, fontweight='bold')
    ax3.set_ylabel('Normalized Value')
    ax3.set_xticks(x)
    ax3.set_xticklabels(efficiency_metrics)
    ax3.legend()
    ax3.set_ylim(0, 1.1)

    # Add actual value labels
    labels = [f'{val:.0f}' if i < 2 else f'{val:.3f}' for i,
              val in enumerate(ip_efficiency)]
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 labels[i], ha='center', va='bottom', fontsize=9)

    labels = [f'{val:.0f}' if i < 2 else f'{val:.3f}' for i,
              val in enumerate(ga_efficiency)]
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 labels[i], ha='center', va='bottom', fontsize=9)

    # 4. Average Optimal Norm over Generation/Iteration
    if ga_results.get('average_fitness_history'):
        generations = list(
            range(1, len(ga_results['average_fitness_history']) + 1))
        ga_avg_fitness = ga_results['average_fitness_history']

        ax4.plot(generations, ga_avg_fitness, 'o-', color='#A23B72',
                 linewidth=2, markersize=4, label='GA Average Fitness')

        # Add IP optimal line as reference (single point since IP doesn't iterate)
        ax4.axhline(y=ip_results['best_fitness'], color='#2E86AB',
                    linestyle='--', linewidth=2, alpha=0.8, label='IP Optimal Reference')

        ax4.set_title('Average Optimal Norm Over Generations',
                      fontsize=14, fontweight='bold')
        ax4.set_xlabel('Generation/Iteration')
        ax4.set_ylabel('Average Fitness (Higher is Better)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Fill area under curve for better visualization
        ax4.fill_between(generations, ga_avg_fitness,
                         alpha=0.3, color='#A23B72')
    else:
        # If no GA data, just show IP point
        ax4.scatter([1], [ip_results['best_fitness']], color='#2E86AB',
                    s=100, label='IP Optimal', alpha=0.8)
        ax4.set_title('Average Optimal Norm Over Generations',
                      fontsize=14, fontweight='bold')
        ax4.set_xlabel('Generation/Iteration')
        ax4.set_ylabel('Average Fitness (Higher is Better)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # 5. Time per Generation/Iteration
    if ga_results.get('plotting_data', {}).get('execution_timeline', {}).get('time_per_generation'):
        generations = list(range(1, len(
            ga_results['plotting_data']['execution_timeline']['time_per_generation']) + 1))
        time_per_gen = ga_results['plotting_data']['execution_timeline']['time_per_generation']

        ax5.plot(generations, time_per_gen, 'o-', color='#28A745',
                 linewidth=2, markersize=4, label='GA Time per Generation')

        # Add IP time as reference (single point)
        ax5.scatter([1], [ip_results['total_time']], color='#2E86AB',
                    s=100, label='IP Total Time', alpha=0.8)

        ax5.set_title('Computation Time per Generation/Iteration',
                      fontsize=14, fontweight='bold')
        ax5.set_xlabel('Generation/Iteration')
        ax5.set_ylabel('Time (seconds)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Fill area under curve
        ax5.fill_between(generations, time_per_gen, alpha=0.3, color='#28A745')
    else:
        # If no GA timing data, just show IP point
        ax5.scatter([1], [ip_results['total_time']], color='#2E86AB',
                    s=100, label='IP Total Time', alpha=0.8)
        ax5.set_title('Computation Time per Generation/Iteration',
                      fontsize=14, fontweight='bold')
        ax5.set_xlabel('Generation/Iteration')
        ax5.set_ylabel('Time (seconds)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # 6. Algorithm Trade-off Analysis (Pareto Front Style)
    algorithms = ['Integer Programming', 'Genetic Algorithm']
    total_costs = [ip_details['total_cost'] + ip_details.get('total_penalty', 0),
                   ga_details['total_cost'] + ga_details['total_penalty']]
    solution_times = [ip_results['total_time'], ga_results['total_time']]
    solution_quality = [100, ga_results['best_fitness'] /
                        ip_results['best_fitness']*100]  # IP as 100% reference

    # Create bubble chart
    colors = ['#2E86AB', '#A23B72']
    sizes = [300, 300]  # Base size for bubbles

    for i, (alg, cost, time, quality) in enumerate(zip(algorithms, total_costs, solution_times, solution_quality)):
        ax6.scatter(time, cost, s=sizes[i], alpha=0.7, color=colors[i],
                    edgecolors='black', linewidth=2, label=f'{alg}\n(Quality: {quality:.1f}%)')

        # Add algorithm name as annotation
        ax6.annotate(alg.split()[0], (time, cost),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=10, fontweight='bold')

    ax6.set_title('Algorithm Trade-off Analysis\n(Cost vs Time vs Quality)',
                  fontsize=14, fontweight='bold')
    ax6.set_xlabel('Solution Time (seconds)')
    ax6.set_ylabel('Total Cost ($)')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Performance Analysis: Convergence, Robustness & Trade-offs',
                 fontsize=16, y=0.98)
    plt.savefig('optimization_performance_analysis.png',
                dpi=300, bbox_inches='tight')
    print("📊 Performance analysis plot saved as 'optimization_performance_analysis.png'")
    plt.show()

    return fig


def print_summary(algorithm_name, results):
    """Print optimization summary."""
    details = results['solution_details']
    print(f"\n{'='*60}")
    print(f"{algorithm_name.upper()} OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"Status: {results.get('status', 'Completed')}")
    print(f"Time: {results['total_time']:.1f}s")
    print(f"Best Fitness: {results['best_fitness']:.2f}")
    print(f"Total Cost: ${details['total_cost']:,.2f}")
    print(
        f"Network Utilization: {details['utilization_stats']['network_overall']:.1f}%")
    print(f"Capacity Violations: {details['capacity_violations']}")

    modules_selected = sum(1 for detail in details['link_details']
                           if detail['module_info']['choice_index'] > 0)
    total_links = len(details['link_details'])
    print(
        f"Links with Modules: {modules_selected}/{total_links} ({modules_selected/total_links*100:.1f}%)")


def genetic_algorithm():
    return run_genetic_algorithm(
        num_generations=50,
        sol_per_pop=50,
        num_parents_mating=20,
        time_limit=20*60,
        msg=False
    )


def integer_programming():
    return run_integer_programming(
        solver_name="GUROBI",
        time_limit=20*60,
        msg=False
    )


def main(algorithm="both"):
    """Main function to run optimization algorithms."""
    if algorithm == "genetic":
        print("Running Genetic Algorithm...")
        results = genetic_algorithm()
        print_summary("Genetic Algorithm", results)

    elif algorithm == "ip":
        print("Running Integer Programming...")
        results = integer_programming()
        print_summary("Integer Programming", results)

    else:  # both
        print("Running Both Algorithms in Parallel...")

        # Use ThreadPoolExecutor to run both algorithms concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            print("\n🧬 Starting Genetic Algorithm...")
            print("🔢 Starting Integer Programming...")

            # Submit both tasks
            ga_future = executor.submit(
                genetic_algorithm,
            )
            ip_future = executor.submit(
                integer_programming(),
            )

            # Wait for results
            print("\n⏳ Waiting for algorithms to complete...")
            ga_results = ga_future.result()
            ip_results = ip_future.result()

        print_summary("Genetic Algorithm", ga_results)
        print_summary("Integer Programming", ip_results)

        # Compare results
        ip_cost = ip_results['solution_details']['total_cost']
        ga_cost = ga_results['solution_details']['total_cost'] + \
            ga_results['solution_details']['total_penalty']
        cost_difference = ip_cost - ga_cost

        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Cost Difference (IP - GA): ${cost_difference:,.2f}")
        if cost_difference < 0:
            print(
                f"✅ IP provides {abs(cost_difference/ga_cost)*100:.1f}% cost improvement")
        elif cost_difference > 0:
            print(
                f"✅ GA provides {cost_difference/ip_cost*100:.1f}% cost improvement")
        else:
            print("Both algorithms achieved similar costs")

        print(f"IP Optimal: {ip_results.get('status') == 'Optimal'}")
        print(f"IP Status: {ip_results.get('status')}")
        print(f"GA Timeout: {ga_results.get('timeout_occurred', False)}")

        # Create comprehensive comparison plots
        print("\n📊 Creating comprehensive comparison plots...")
        fig1, fig2, fig3 = create_shared_comparison_plots(
            ip_results, ga_results)

        return {"ip_results": ip_results, "ga_results": ga_results}

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        algorithm = sys.argv[1].lower()
        if algorithm in ["genetic", "ga", "g"]:
            main("genetic")
        elif algorithm in ["integer", "ip", "i"]:
            main("ip")
        elif algorithm in ["both", "compare", "all"]:
            main("both")
        else:
            print("Usage: python main.py [genetic|ip|both]")
            sys.exit(1)
    else:
        main("both")  # Default to both algorithms
