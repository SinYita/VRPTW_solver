#!/bin/bash

echo "演示交互模式的使用方法："
echo ""
echo "运行程序："
echo "./terminal_viz"
echo ""
echo "选择 2 进入交互模式"
echo ""
echo "在交互模式中，您可以使用以下命令："
echo "- n 或 N: 下一个时间窗口"
echo "- p 或 P: 上一个时间窗口" 
echo "- a 或 A: 开始自动播放"
echo "- q 或 Q: 退出程序"
echo ""
echo "程序特点："
echo "✓ 彩色终端输出，每个时间窗口使用不同颜色和符号"
echo "✓ 随机生成点位置数据"
echo "✓ 实时动画播放"
echo "✓ 支持手动控制浏览"
echo "✓ 显示详细的统计信息"
echo ""

# 让用户选择是否运行交互模式
echo "是否现在运行交互模式演示？(y/n)"
read -r answer

if [[ $answer == "y" || $answer == "Y" ]]; then
    echo "启动交互模式..."
    cd /workspaces/CVRTM-solver
    echo "2" | ./terminal_viz
fi