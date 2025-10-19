import json
import matplotlib.pyplot as plt
import numpy as np


# B = 16, H = 64, HK8, D = 128.
mi355x_gqa_baselines_causal = {
    "triton": {
        "1024": 456,
        "2048": 569,
        "4096": 614,
        "8192": 764,
        "16384": 848,
    },
    "ck": {
        "1024": 596.53,
        "2048": 695.71,
        "4096": 799.97,
        "8192": 861.79,
        "16384": 878.86,
    },
    "torch": {
        "1024": 14,
        "2048": 15,
        "4096": 15,
        "8192": "OOM",
        "16384": "OOM",
    }
}

mi355x_gqa_baselines_non_causal = {
    "triton": {
        "1024": 844,
        "2048": 945,
        "4096": 996,
        "8192": 1005,
        "16384":1011,
    },
    "ck": {
        "1024": 799,
        "2048": 847,
        "4096": 884,
        "8192": 904,
        "16384": 901,
    },
    "torch": {
        "1024": 29,
        "2048": 31,
        "4096": 34,
        "8192": "OOM",
        "16384": "OOM",
    }
}

# B = 16, H = 16, D = 128.
mi355x_mha_baselines_causal = {
    "triton": {
        "1024": 371,
        "2048": 508,
        "4096": 573,
        "8192": 733,
        "16384": 845,
    },
    "ck": {
        "1024": 485,
        "2048": 601,
        "4096": 745,
        "8192": 834,
        "16384": 893,
    },
    "torch": {
        "1024": 13,
        "2048": 14,
        "4096": 15,
        "8192": 15,
        "16384": "OOM",
    }
}

# B = 16, H = 16, D = 128.
mi355x_mha_baselines_non_causal = {
    "triton": {
        "1024": 694,
        "2048": 855,
        "4096": 944,
        "8192": 1001,
        "16384": 1011,
    },
    "ck": {
        "1024": 761,
        "2048": 733,
        "4096": 816,
        "8192": 896,
        "16384": 914,
    },
    "torch": {
        "1024": 29,
        "2048": 32,
        "4096": 34,
        "8192": 33,
        "16384": "OOM",
    }
}

colors = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC"]


for device in ['mi300x', 'mi325x', 'mi350x', 'mi355x']:

    for setting in ['mha_causal_fwd', 'mha_non_causal_fwd', 'gqa_causal_fwd', 'gqa_non_causal_fwd']:

        # Read data
        try:
            with open(f'mi350x/{device}_{setting}.json', 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {device}_{setting}.json: {e}")
            continue

        # Extract data for plotting
        matrix_sizes = sorted([int(size) for size in data.keys()])
        aiter_tflops = [data[str(size)]['tflops_ref'] for size in matrix_sizes]
        tk_tflops = [data[str(size)]['tflops'] for size in matrix_sizes]

        triton_tflops = []
        torch_tflops = []
        ck_tflops = []
        if setting == 'mha_causal_fwd' and device == 'mi355x':
            triton_tflops = [mi355x_mha_baselines_causal['triton'][str(size)] for size in matrix_sizes]
            torch_tflops = [mi355x_mha_baselines_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_mha_baselines_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'mha_non_causal_fwd' and device == 'mi355x':
            triton_tflops = [mi355x_mha_baselines_non_causal['triton'][str(size)] for size in matrix_sizes]
            torch_tflops = [mi355x_mha_baselines_non_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_mha_baselines_non_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'gqa_causal_fwd' and device == 'mi355x':
            triton_tflops = [mi355x_gqa_baselines_causal['triton'][str(size)] for size in matrix_sizes]
            torch_tflops = [mi355x_gqa_baselines_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_gqa_baselines_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'gqa_non_causal_fwd' and device == 'mi355x':
            triton_tflops = [mi355x_gqa_baselines_non_causal['triton'][str(size)] for size in matrix_sizes]
            torch_tflops = [mi355x_gqa_baselines_non_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_gqa_baselines_non_causal['ck'][str(size)] for size in matrix_sizes]

        # Create bar chart
        x = np.arange(len(matrix_sizes))
        width = 0.3

        fig, ax = plt.subplots(figsize=(10, 6))
        bars0 = ax.bar(x - width, aiter_tflops, width, label='AITER (AMD)', color=colors[1])
        bars1 = ax.bar(x, tk_tflops, width, label='ThunderKittens', color=colors[3])
        if len(triton_tflops) > 0:
            bars2 = ax.bar(x + width, triton_tflops, width, label='Triton', color=colors[0])
        if len(torch_tflops) > 0:
            bars3 = ax.bar(x + width * 2, torch_tflops, width, label='PyTorch', color=colors[2])
        if len(ck_tflops) > 0:
            bars4 = ax.bar(x + width * 3, ck_tflops, width, label='Composable Kernel', color=colors[4])

        max_tflops = max(max(aiter_tflops), max(tk_tflops), max(triton_tflops), max(torch_tflops), max(ck_tflops))

        # Add value labels on bars
        for bar, value in zip(bars0, aiter_tflops):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=14)

        for bar, value in zip(bars1, tk_tflops):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=14)

        if len(triton_tflops) > 0:
            for bar, value in zip(bars2, triton_tflops):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                        f'{value:.0f}', ha='center', va='bottom', fontsize=14)
        if len(torch_tflops) > 0:
            for bar, value in zip(bars3, torch_tflops):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                        f'{value:.0f}', ha='center', va='bottom', fontsize=14)
                        
        if len(ck_tflops) > 0:
            for bar, value in zip(bars4, ck_tflops):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                        f'{value:.0f}', ha='center', va='bottom', fontsize=14)

        # add some padding to the top of the y-axis to prevent label overlap
        ax.set_ylim(0, max_tflops * 1.15)
        ax.set_xlabel('Sequence Length (N)', fontsize=16)
        ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
        ax.set_title(f'Attention Performance Comparison {device.upper()}', fontsize=16)
        ax.set_xticks(x, fontsize=16)
        ax.set_yticks(fontsize=16)
        ax.set_xticklabels(matrix_sizes, fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.legend(fontsize=16)
        # ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        output_file = f'{device}_{setting}_attn_plot.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")

        # Print summary
        print(f"Matrix sizes tested: {matrix_sizes}")
        print(f"AITER (AMD) TFLOPS: {[f'{t:.2f}' for t in aiter_tflops]}")
        print(f"TK TFLOPS: {[f'{t:.2f}' for t in tk_tflops]}")
        if len(triton_tflops) > 0:
            print(f"Triton TFLOPS: {[f'{t:.2f}' for t in triton_tflops]}")
        if len(torch_tflops) > 0:
            print(f"PyTorch TFLOPS: {[f'{t:.2f}' for t in torch_tflops]}")
        if len(ck_tflops) > 0:
            print(f"Composable Kernel TFLOPS: {[f'{t:.2f}' for t in ck_tflops]}")

