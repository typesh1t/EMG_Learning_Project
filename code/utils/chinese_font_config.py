#!/usr/bin/env python3
"""
中文字体配置模块
解决matplotlib中文显示问题
"""

import matplotlib.pyplot as plt
import platform
import warnings

def setup_chinese_font():
    """
    配置matplotlib使用中文字体
    根据不同操作系统自动选择合适的字体
    """
    system = platform.system()

    # 根据操作系统选择字体
    if system == 'Windows':
        # Windows系统字体
        fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
    elif system == 'Darwin':  # macOS
        # macOS系统字体
        fonts = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
    else:  # Linux
        # Linux系统字体
        fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback', 'AR PL UMing CN']

    # 尝试设置字体
    font_set = False
    for font in fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

            # 测试字体是否可用
            fig, ax = plt.subplots(1, 1, figsize=(1, 1))
            ax.text(0.5, 0.5, '测试', fontsize=12)
            plt.close(fig)

            font_set = True
            print(f"✓ 中文字体配置成功: {font}")
            break
        except:
            continue

    if not font_set:
        # 如果所有字体都失败，使用备用方案
        warnings.warn("未找到合适的中文字体，图表中的中文可能无法正常显示")
        print("\n⚠️  解决方案:")
        print("1. Windows: 确保安装了SimHei或微软雅黑字体")
        print("2. macOS: 系统自带PingFang SC字体")
        print("3. Linux: 安装文泉驿字体 (sudo apt-get install fonts-wqy-microhei)")
        print("4. 或者使用英文替代图表标题和标签\n")

    return font_set


def get_available_chinese_fonts():
    """
    获取系统中可用的中文字体列表

    返回:
        list: 可用的中文字体名称列表
    """
    from matplotlib.font_manager import FontManager

    fm = FontManager()
    chinese_fonts = []

    # 常见中文字体关键词
    keywords = ['Hei', 'Song', 'Kai', 'Ming', 'SimHei', 'SimSun',
                'Microsoft', 'YaHei', 'PingFang', 'WenQuanYi', 'AR PL']

    for font in fm.ttflist:
        for keyword in keywords:
            if keyword in font.name:
                if font.name not in chinese_fonts:
                    chinese_fonts.append(font.name)
                break

    return chinese_fonts


def test_chinese_display():
    """
    测试中文显示效果
    """
    import numpy as np

    # 配置字体
    setup_chinese_font()

    # 创建测试图表
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # 测试1：文本显示
    axes[0].text(0.5, 0.5, '中文显示测试\nChinese Font Test\n字体配置成功！',
                ha='center', va='center', fontsize=20)
    axes[0].set_title('中文字体测试')
    axes[0].axis('off')

    # 测试2：图表标签
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    axes[1].plot(x, y)
    axes[1].set_xlabel('时间 (秒)')
    axes[1].set_ylabel('幅度 (mV)')
    axes[1].set_title('正弦波示例')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('code/utils/chinese_font_test.png', dpi=100)
    print("\n✓ 测试图表已保存到: code/utils/chinese_font_test.png")
    plt.show()

    print("\n如果图表中的中文正常显示，说明配置成功！")


# 自动配置（导入时执行）
_auto_config = True
if _auto_config:
    setup_chinese_font()


if __name__ == '__main__':
    print("\n" + "="*60)
    print("中文字体配置工具".center(60))
    print("="*60 + "\n")

    # 查找可用字体
    print("正在查找系统中的中文字体...")
    available_fonts = get_available_chinese_fonts()

    if available_fonts:
        print(f"\n找到 {len(available_fonts)} 个中文字体:")
        for i, font in enumerate(available_fonts[:10], 1):
            print(f"  {i}. {font}")
        if len(available_fonts) > 10:
            print(f"  ... 还有 {len(available_fonts) - 10} 个字体")
    else:
        print("\n⚠️  未找到中文字体")

    # 配置字体
    print("\n正在配置matplotlib...")
    success = setup_chinese_font()

    # 运行测试
    if success:
        print("\n正在生成测试图表...")
        test_chinese_display()

    print("\n配置完成！")
