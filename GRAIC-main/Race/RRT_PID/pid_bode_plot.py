#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PID控制器波特图可视化工具
用于分析和调节PID参数，帮助优化控制性能
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

class PIDBodeAnalyzer:
    """PID控制器波特图分析器"""
    
    def __init__(self, dt=0.05):
        """
        初始化分析器
        Args:
            dt: 采样时间（秒），默认0.05（20Hz）
        """
        self.dt = dt
        self.frequencies = np.logspace(-2, 2, 1000)  # 频率范围：0.01 到 100 Hz
        
    def pid_transfer_function(self, kp, ki, kd):
        """
        计算PID控制器的传递函数（连续时间）
        G(s) = Kp + Ki/s + Kd*s
        
        Args:
            kp: 比例系数
            ki: 积分系数
            kd: 微分系数
        Returns:
            TransferFunction对象
        """
        # PID传递函数: G(s) = Kp + Ki/s + Kd*s
        # 转换为标准形式: (Kd*s^2 + Kp*s + Ki) / s
        num = [kd, kp, ki]
        den = [1, 0]  # s
        return signal.TransferFunction(num, den)
    
    def pid_discrete_transfer_function(self, kp, ki, kd):
        """
        计算PID控制器的离散时间传递函数（考虑采样时间）
        
        Args:
            kp: 比例系数
            ki: 积分系数
            kd: 微分系数
        Returns:
            TransferFunction对象（离散时间）
        """
        # 离散PID: u[k] = Kp*e[k] + Ki*sum(e[i])*dt + Kd*(e[k]-e[k-1])/dt
        # 使用双线性变换或直接计算z变换
        # 简化：使用连续时间传递函数，然后转换为离散时间
        tf_continuous = self.pid_transfer_function(kp, ki, kd)
        tf_discrete = signal.cont2discrete(
            (tf_continuous.num, tf_continuous.den), 
            self.dt, 
            method='bilinear'
        )
        return signal.TransferFunction(tf_discrete[0], tf_discrete[1], dt=self.dt)
    
    def plot_bode(self, kp, ki, kd, label="PID", color=None, linestyle='-'):
        """
        绘制PID控制器的波特图
        
        Args:
            kp: 比例系数
            ki: 积分系数
            kd: 微分系数
            label: 图例标签
            color: 线条颜色
            linestyle: 线条样式
        Returns:
            (magnitude, phase, frequency) 元组
        """
        # 计算传递函数
        tf = self.pid_transfer_function(kp, ki, kd)
        
        # 计算频率响应
        w, mag, phase = signal.bode(tf, self.frequencies * 2 * np.pi)
        
        # 转换为Hz
        freq_hz = w / (2 * np.pi)
        
        return freq_hz, mag, phase, label, color, linestyle
    
    def plot_comparison(self, pid_params_list, title="PID控制器波特图对比"):
        """
        对比多个PID参数设置的波特图
        
        Args:
            pid_params_list: PID参数列表，格式为 [(kp, ki, kd, label, color), ...]
            title: 图表标题
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
        
        for idx, params in enumerate(pid_params_list):
            if len(params) >= 4:
                kp, ki, kd, label = params[0], params[1], params[2], params[3]
                color = params[4] if len(params) > 4 else colors[idx % len(colors)]
                linestyle = params[5] if len(params) > 5 else '-'
            else:
                kp, ki, kd = params[0], params[1], params[2]
                label = f"Kp={kp:.2f}, Ki={ki:.2f}, Kd={kd:.2f}"
                color = colors[idx % len(colors)]
                linestyle = '-'
            
            freq, mag, phase, _, _, _ = self.plot_bode(kp, ki, kd, label, color, linestyle)
            
            # 绘制幅频特性
            ax1.semilogx(freq, mag, label=label, color=color, linestyle=linestyle, linewidth=2)
            
            # 绘制相频特性
            ax2.semilogx(freq, phase, label=label, color=color, linestyle=linestyle, linewidth=2)
        
        # 设置幅频特性图
        ax1.set_xlabel('频率 (Hz)', fontsize=12)
        ax1.set_ylabel('幅值 (dB)', fontsize=12)
        ax1.set_title('幅频特性', fontsize=14, fontweight='bold')
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.legend(loc='best', fontsize=10)
        ax1.set_xlim([0.01, 100])
        
        # 设置相频特性图
        ax2.set_xlabel('频率 (Hz)', fontsize=12)
        ax2.set_ylabel('相位 (度)', fontsize=12)
        ax2.set_title('相频特性', fontsize=14, fontweight='bold')
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.legend(loc='best', fontsize=10)
        ax2.set_xlim([0.01, 100])
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def save_plot(self, fig, filename='pid_bode_plot.png', dpi=300):
        """保存波特图"""
        try:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"✓ 图表已保存为: {filename}")
        except Exception as e:
            print(f"✗ 保存失败: {e}")
    
    def analyze_stability_margins(self, kp, ki, kd):
        """
        分析PID控制器的稳定裕度
        
        Args:
            kp, ki, kd: PID参数
        Returns:
            包含稳定裕度信息的字典
        """
        try:
            tf = self.pid_transfer_function(kp, ki, kd)
            gm, pm, wcg, wcp = signal.margin(tf)
            
            # 处理特殊情况
            if gm <= 0 or np.isinf(gm):
                gain_margin_db = np.inf
            else:
                gain_margin_db = 20 * np.log10(gm)
            
            if wcg > 0:
                gain_crossover_freq = wcg / (2 * np.pi)
            else:
                gain_crossover_freq = np.inf
                
            if wcp > 0:
                phase_crossover_freq = wcp / (2 * np.pi)
            else:
                phase_crossover_freq = np.inf
            
            return {
                'gain_margin_db': gain_margin_db,
                'phase_margin_deg': pm if not np.isnan(pm) else np.inf,
                'gain_crossover_freq': gain_crossover_freq,
                'phase_crossover_freq': phase_crossover_freq
            }
        except Exception as e:
            print(f"警告：稳定性分析失败 - {e}")
            return {
                'gain_margin_db': np.nan,
                'phase_margin_deg': np.nan,
                'gain_crossover_freq': np.nan,
                'phase_crossover_freq': np.nan
            }
    
    def print_stability_info(self, kp, ki, kd):
        """打印稳定性信息"""
        margins = self.analyze_stability_margins(kp, ki, kd)
        print(f"\nPID参数: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")
        print(f"增益裕度: {margins['gain_margin_db']:.2f} dB")
        print(f"相位裕度: {margins['phase_margin_deg']:.2f} 度")
        print(f"增益穿越频率: {margins['gain_crossover_freq']:.2f} Hz")
        print(f"相位穿越频率: {margins['phase_crossover_freq']:.2f} Hz")


def plot_agent_pid_analysis(save_plot=False):
    """分析agent.py中定义的PID控制器"""
    analyzer = PIDBodeAnalyzer(dt=0.05)
    
    # 从agent.py中提取的PID参数
    pid_configs = [
        # (kp, ki, kd, label, color)
        (1.2, 0.05, 0.3, "转向PID (默认)", 'b'),
        (0.15, 0.01, 0.1, "横向误差PID", 'r'),
        (0.8, 0.1, 0.2, "速度PID", 'g'),
        # 自适应PID参数（不同道路类型）
        (1.5, 0.03, 0.2, "直道PID", 'm'),
        (0.9, 0.08, 0.5, "急转弯PID", 'c'),
    ]
    
    # 绘制对比图
    fig = analyzer.plot_comparison(pid_configs, "Agent PID控制器波特图分析")
    
    # 打印稳定性信息
    print("=" * 60)
    print("PID控制器稳定性分析")
    print("=" * 60)
    for config in pid_configs:
        kp, ki, kd = config[0], config[1], config[2]
        analyzer.print_stability_info(kp, ki, kd)
        print("-" * 60)
    
    # 保存图片
    if save_plot:
        analyzer.save_plot(fig, 'agent_pid_bode_plot.png')
    
    plt.show()
    return fig


def interactive_pid_tuning():
    """交互式PID参数调节工具"""
    analyzer = PIDBodeAnalyzer(dt=0.05)
    
    print("=" * 60)
    print("交互式PID参数调节工具")
    print("=" * 60)
    print("输入PID参数，查看波特图和稳定性分析")
    print("输入 'q' 退出")
    print("-" * 60)
    
    pid_list = []
    
    while True:
        try:
            user_input = input("\n请输入PID参数 (格式: kp,ki,kd 或 'q'退出): ").strip()
            
            if user_input.lower() == 'q':
                break
            
            parts = user_input.split(',')
            if len(parts) != 3:
                print("错误：请输入三个参数，用逗号分隔")
                continue
            
            kp = float(parts[0])
            ki = float(parts[1])
            kd = float(parts[2])
            
            # 添加到列表
            label = f"Kp={kp:.2f}, Ki={ki:.2f}, Kd={kd:.2f}"
            pid_list.append((kp, ki, kd, label))
            
            # 显示稳定性信息
            analyzer.print_stability_info(kp, ki, kd)
            
            # 询问是否继续
            continue_input = input("\n继续添加参数？(y/n，输入'p'绘制当前所有参数): ").strip().lower()
            if continue_input == 'p':
                if pid_list:
                    fig = analyzer.plot_comparison(pid_list, "PID参数对比分析")
                    plt.show()
                else:
                    print("没有参数可绘制")
            elif continue_input != 'y':
                break
                
        except ValueError:
            print("错误：请输入有效的数字")
        except KeyboardInterrupt:
            break
    
    # 绘制所有输入的参数
    if pid_list:
        fig = analyzer.plot_comparison(pid_list, "PID参数对比分析")
        plt.show()


def plot_sensitivity_analysis(kp_base=1.2, ki_base=0.05, kd_base=0.3, variation=0.3):
    """
    绘制PID参数敏感性分析
    显示参数变化对频率响应的影响
    
    Args:
        kp_base, ki_base, kd_base: 基准PID参数
        variation: 参数变化范围（比例，如0.3表示±30%）
    """
    analyzer = PIDBodeAnalyzer(dt=0.05)
    
    # Kp敏感性分析
    kp_variations = [
        (kp_base * (1 - variation), ki_base, kd_base, f"Kp-{variation*100:.0f}%", 'r', '--'),
        (kp_base, ki_base, kd_base, "Kp基准", 'b', '-'),
        (kp_base * (1 + variation), ki_base, kd_base, f"Kp+{variation*100:.0f}%", 'g', '--'),
    ]
    
    # Ki敏感性分析
    ki_variations = [
        (kp_base, ki_base * (1 - variation), kd_base, f"Ki-{variation*100:.0f}%", 'r', '--'),
        (kp_base, ki_base, kd_base, "Ki基准", 'b', '-'),
        (kp_base, ki_base * (1 + variation), kd_base, f"Ki+{variation*100:.0f}%", 'g', '--'),
    ]
    
    # Kd敏感性分析
    kd_variations = [
        (kp_base, ki_base, kd_base * (1 - variation), f"Kd-{variation*100:.0f}%", 'r', '--'),
        (kp_base, ki_base, kd_base, "Kd基准", 'b', '-'),
        (kp_base, ki_base, kd_base * (1 + variation), f"Kd+{variation*100:.0f}%", 'g', '--'),
    ]
    
    # 创建子图
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # Kp敏感性
    for params in kp_variations:
        freq, mag, phase, label, color, linestyle = analyzer.plot_bode(*params[:3], params[3], params[4], params[5])
        axes[0, 0].semilogx(freq, mag, label=label, color=color, linestyle=linestyle, linewidth=2)
        axes[0, 1].semilogx(freq, phase, label=label, color=color, linestyle=linestyle, linewidth=2)
    
    # Ki敏感性
    for params in ki_variations:
        freq, mag, phase, label, color, linestyle = analyzer.plot_bode(*params[:3], params[3], params[4], params[5])
        axes[1, 0].semilogx(freq, mag, label=label, color=color, linestyle=linestyle, linewidth=2)
        axes[1, 1].semilogx(freq, phase, label=label, color=color, linestyle=linestyle, linewidth=2)
    
    # Kd敏感性
    for params in kd_variations:
        freq, mag, phase, label, color, linestyle = analyzer.plot_bode(*params[:3], params[3], params[4], params[5])
        axes[2, 0].semilogx(freq, mag, label=label, color=color, linestyle=linestyle, linewidth=2)
        axes[2, 1].semilogx(freq, phase, label=label, color=color, linestyle=linestyle, linewidth=2)
    
    # 设置标签和标题
    titles = ['Kp参数敏感性 - 幅频特性', 'Kp参数敏感性 - 相频特性',
              'Ki参数敏感性 - 幅频特性', 'Ki参数敏感性 - 相频特性',
              'Kd参数敏感性 - 幅频特性', 'Kd参数敏感性 - 相频特性']
    
    for i, ax in enumerate(axes.flat):
        ax.set_xlabel('频率 (Hz)', fontsize=10)
        if i % 2 == 0:
            ax.set_ylabel('幅值 (dB)', fontsize=10)
        else:
            ax.set_ylabel('相位 (度)', fontsize=10)
        ax.set_title(titles[i], fontsize=11, fontweight='bold')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim([0.01, 100])
    
    plt.suptitle(f'PID参数敏感性分析 (基准: Kp={kp_base}, Ki={ki_base}, Kd={kd_base})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return fig


if __name__ == '__main__':
    import sys
    
    print("PID控制器波特图分析工具")
    print("=" * 60)
    print("1. 分析agent.py中的PID参数")
    print("2. 交互式PID参数调节")
    print("3. 参数敏感性分析")
    print("=" * 60)
    
    # 检查是否有保存选项
    save_plot = '--save' in sys.argv or '-s' in sys.argv
    
    if len(sys.argv) > 1 and sys.argv[1] not in ['--save', '-s']:
        mode = sys.argv[1]
    else:
        mode = input("请选择模式 (1/2/3，直接回车默认选择1): ").strip() or "1"
    
    if mode == "1":
        plot_agent_pid_analysis(save_plot=save_plot)
    elif mode == "2":
        interactive_pid_tuning()
    elif mode == "3":
        # 可以修改基准参数
        fig = plot_sensitivity_analysis(kp_base=1.2, ki_base=0.05, kd_base=0.3, variation=0.3)
        if save_plot:
            analyzer = PIDBodeAnalyzer(dt=0.05)
            analyzer.save_plot(fig, 'pid_sensitivity_analysis.png')
    else:
        print("无效的选择，使用默认模式1")
        plot_agent_pid_analysis(save_plot=save_plot)

