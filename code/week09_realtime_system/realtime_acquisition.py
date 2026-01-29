#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时EMG数据采集模块

支持多种数据源：
- 串口设备（Arduino、MCU）
- 模拟数据生成（用于测试）
- CSV文件播放（用于演示）

作者: EMG Learning Project
日期: 2026-01-29
"""

import numpy as np
import time
import threading
from collections import deque
from typing import Optional, Callable
import queue


class RealtimeEMGAcquisition:
    """实时EMG数据采集类"""

    def __init__(self,
                 source='simulator',
                 n_channels=4,
                 fs=1000,
                 buffer_size=5000,
                 port=None):
        """
        初始化采集器

        参数:
            source: 数据源类型 ('simulator', 'serial', 'file')
            n_channels: 通道数
            fs: 采样率 (Hz)
            buffer_size: 缓冲区大小（样本数）
            port: 串口端口（如 'COM3' 或 '/dev/ttyUSB0'）
        """
        self.source = source
        self.n_channels = n_channels
        self.fs = fs
        self.buffer_size = buffer_size
        self.port = port

        # 数据缓冲区（每个通道一个deque）
        self.buffers = [deque(maxlen=buffer_size) for _ in range(n_channels)]

        # 控制标志
        self.is_running = False
        self.is_paused = False

        # 采集线程
        self.acquisition_thread = None

        # 回调函数
        self.data_callbacks = []

        # 统计信息
        self.sample_count = 0
        self.start_time = None

        # 初始化数据源
        self._init_source()

    def _init_source(self):
        """初始化数据源"""
        if self.source == 'serial':
            try:
                import serial
                self.serial_port = serial.Serial(self.port, 115200, timeout=0.1)
                print(f"✓ 串口已打开: {self.port}")
            except Exception as e:
                print(f"✗ 串口打开失败: {e}")
                print("  切换到模拟器模式")
                self.source = 'simulator'

        elif self.source == 'simulator':
            print("✓ 使用模拟器生成数据")
            self._init_simulator()

        elif self.source == 'file':
            print("✓ 从文件读取数据")
            # 文件播放模式的初始化
            pass

    def _init_simulator(self):
        """初始化模拟器"""
        self.sim_time = 0
        self.sim_state = 'rest'  # 'rest' 或 'active'
        self.sim_state_duration = 0

    def start(self):
        """开始采集"""
        if self.is_running:
            print("采集已经在运行中")
            return

        self.is_running = True
        self.is_paused = False
        self.start_time = time.time()
        self.sample_count = 0

        # 启动采集线程
        self.acquisition_thread = threading.Thread(target=self._acquisition_loop)
        self.acquisition_thread.daemon = True
        self.acquisition_thread.start()

        print("✓ 数据采集已启动")

    def stop(self):
        """停止采集"""
        if not self.is_running:
            return

        self.is_running = False

        # 等待线程结束
        if self.acquisition_thread:
            self.acquisition_thread.join(timeout=2)

        # 关闭串口
        if self.source == 'serial' and hasattr(self, 'serial_port'):
            self.serial_port.close()

        print("✓ 数据采集已停止")
        print(f"  总采样数: {self.sample_count}")
        print(f"  运行时长: {time.time() - self.start_time:.2f}秒")

    def pause(self):
        """暂停采集"""
        self.is_paused = True
        print("⏸ 采集已暂停")

    def resume(self):
        """恢复采集"""
        self.is_paused = False
        print("▶ 采集已恢复")

    def _acquisition_loop(self):
        """采集循环（在独立线程中运行）"""
        while self.is_running:
            if self.is_paused:
                time.sleep(0.01)
                continue

            # 根据数据源读取数据
            if self.source == 'simulator':
                data = self._read_from_simulator()
            elif self.source == 'serial':
                data = self._read_from_serial()
            elif self.source == 'file':
                data = self._read_from_file()
            else:
                data = None

            if data is not None:
                # 添加到缓冲区
                for ch in range(self.n_channels):
                    self.buffers[ch].append(data[ch])

                self.sample_count += 1

                # 调用回调函数
                for callback in self.data_callbacks:
                    try:
                        callback(data)
                    except Exception as e:
                        print(f"回调函数错误: {e}")

            # 控制采样率
            time.sleep(1.0 / self.fs)

    def _read_from_simulator(self):
        """从模拟器读取数据"""
        # 模拟EMG信号
        self.sim_time += 1.0 / self.fs
        self.sim_state_duration += 1.0 / self.fs

        # 每3-5秒切换状态
        if self.sim_state_duration > np.random.uniform(3, 5):
            self.sim_state = 'active' if self.sim_state == 'rest' else 'rest'
            self.sim_state_duration = 0

        # 生成数据
        data = []
        for ch in range(self.n_channels):
            if self.sim_state == 'rest':
                # 静息：低幅度噪声
                signal = np.random.normal(0, 0.02)
            else:
                # 激活：高幅度信号
                signal = 0
                for freq in range(60, 150, 20):
                    signal += 0.2 * np.sin(2 * np.pi * freq * self.sim_time)
                signal += np.random.normal(0, 0.1)

            data.append(signal)

        return np.array(data)

    def _read_from_serial(self):
        """从串口读取数据"""
        try:
            line = self.serial_port.readline().decode('utf-8').strip()
            if line:
                # 假设格式: "ch0,ch1,ch2,ch3"
                values = [float(x) for x in line.split(',')]
                if len(values) == self.n_channels:
                    return np.array(values)
        except Exception as e:
            print(f"串口读取错误: {e}")
        return None

    def _read_from_file(self):
        """从文件读取数据"""
        # TODO: 实现文件播放
        return None

    def get_latest_data(self, n_samples=None):
        """
        获取最新的数据

        参数:
            n_samples: 获取的样本数，None则返回所有缓冲区数据

        返回:
            data: (n_samples, n_channels) 数组
        """
        if n_samples is None:
            n_samples = len(self.buffers[0])

        data = np.zeros((n_samples, self.n_channels))
        for ch in range(self.n_channels):
            buffer_list = list(self.buffers[ch])
            if len(buffer_list) >= n_samples:
                data[:, ch] = buffer_list[-n_samples:]
            else:
                # 不足时用0填充
                data[:len(buffer_list), ch] = buffer_list

        return data

    def get_buffer_size(self):
        """获取当前缓冲区大小"""
        return len(self.buffers[0])

    def register_callback(self, callback: Callable):
        """
        注册数据回调函数

        参数:
            callback: 回调函数，接收一个参数（新数据点）
        """
        self.data_callbacks.append(callback)

    def clear_buffers(self):
        """清空缓冲区"""
        for buffer in self.buffers:
            buffer.clear()
        print("✓ 缓冲区已清空")

    def get_statistics(self):
        """获取统计信息"""
        if self.start_time is None:
            return None

        elapsed = time.time() - self.start_time
        actual_fs = self.sample_count / elapsed if elapsed > 0 else 0

        return {
            'sample_count': self.sample_count,
            'elapsed_time': elapsed,
            'actual_sampling_rate': actual_fs,
            'buffer_fill': self.get_buffer_size() / self.buffer_size * 100,
        }


def demo():
    """演示实时采集"""
    print("="*60)
    print("实时EMG数据采集演示")
    print("="*60)

    # 创建采集器
    acq = RealtimeEMGAcquisition(
        source='simulator',
        n_channels=4,
        fs=100,  # 降低采样率以便观察
        buffer_size=1000
    )

    # 定义回调函数
    sample_counter = [0]
    def on_new_data(data):
        sample_counter[0] += 1
        if sample_counter[0] % 50 == 0:
            print(f"采样 {sample_counter[0]}: {data}")

    acq.register_callback(on_new_data)

    # 开始采集
    acq.start()

    try:
        # 运行5秒
        print("\n运行5秒...")
        time.sleep(5)

        # 获取最新数据
        print("\n获取最新100个样本:")
        data = acq.get_latest_data(100)
        print(f"  数据形状: {data.shape}")
        print(f"  通道0均值: {np.mean(data[:, 0]):.4f}")
        print(f"  通道0标准差: {np.std(data[:, 0]):.4f}")

        # 统计信息
        stats = acq.get_statistics()
        print("\n统计信息:")
        print(f"  总采样数: {stats['sample_count']}")
        print(f"  运行时长: {stats['elapsed_time']:.2f}秒")
        print(f"  实际采样率: {stats['actual_sampling_rate']:.1f} Hz")
        print(f"  缓冲区填充: {stats['buffer_fill']:.1f}%")

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        acq.stop()

    print("\n演示完成！")


if __name__ == '__main__':
    demo()
