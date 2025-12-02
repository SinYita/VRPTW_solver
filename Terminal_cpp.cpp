#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <set>
#include <string>
#include <unistd.h>

// ANSI color codes for terminal output
namespace Colors {
    const std::string RESET = "\033[0m";
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string BLUE = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN = "\033[36m";
    const std::string WHITE = "\033[37m";
    const std::string BRIGHT_RED = "\033[91m";
    const std::string BRIGHT_GREEN = "\033[92m";
    const std::string BRIGHT_YELLOW = "\033[93m";
    const std::string BRIGHT_BLUE = "\033[94m";
    const std::string BRIGHT_MAGENTA = "\033[95m";
    const std::string BRIGHT_CYAN = "\033[96m";
    const std::string BRIGHT_WHITE = "\033[97m";
    
    // Background colors
    const std::string BG_BLACK = "\033[40m";
    const std::string BG_RED = "\033[41m";
    const std::string BG_GREEN = "\033[42m";
    const std::string BG_YELLOW = "\033[43m";
    const std::string BG_BLUE = "\033[44m";
    const std::string BG_MAGENTA = "\033[45m";
    const std::string BG_CYAN = "\033[46m";
    const std::string BG_WHITE = "\033[47m";
    const std::string BG_BRIGHT_RED = "\033[101m";
    const std::string BG_BRIGHT_GREEN = "\033[102m";
    const std::string BG_BRIGHT_YELLOW = "\033[103m";
    const std::string BG_BRIGHT_BLUE = "\033[104m";
    const std::string BG_BRIGHT_MAGENTA = "\033[105m";
    const std::string BG_BRIGHT_CYAN = "\033[106m";
    const std::string BG_GRAY = "\033[100m";
}

// Structure to represent a point with coordinates and time window
struct Point {
    int x, y;
    int timeWindow;
    std::string bgColor;
    
    Point(int x, int y, int timeWindow) : x(x), y(y), timeWindow(timeWindow) {
        // Assign different background colors based on time window
        std::vector<std::string> bgColors = {
            Colors::BG_RED, Colors::BG_GREEN, Colors::BG_YELLOW, Colors::BG_BLUE,
            Colors::BG_MAGENTA, Colors::BG_CYAN, Colors::BG_BRIGHT_RED, Colors::BG_BRIGHT_GREEN,
            Colors::BG_BRIGHT_YELLOW, Colors::BG_BRIGHT_BLUE, Colors::BG_BRIGHT_MAGENTA, Colors::BG_BRIGHT_CYAN
        };
        
        this->bgColor = bgColors[timeWindow % bgColors.size()];
    }
};

class TerminalVisualizer {
private:
    int width, height;
    int maxTimeWindows;
    std::vector<Point> allPoints;
    std::mt19937 rng;
    
public:
    TerminalVisualizer(int w, int h, int timeWindows) 
        : width(w), height(h), maxTimeWindows(timeWindows), rng(std::random_device{}()) {
        generatePoints();
    }
    
    void generatePoints() {
        std::uniform_int_distribution<int> xDist(0, width - 1);
        std::uniform_int_distribution<int> yDist(0, height - 1);
        std::uniform_int_distribution<int> pointCountDist(3, 8); // 3-8 points per time window
        
        std::cout << Colors::BRIGHT_CYAN << "生成点数据..." << Colors::RESET << std::endl;
        
        for (int timeWindow = 0; timeWindow < maxTimeWindows; timeWindow++) {
            int pointCount = pointCountDist(rng);
            std::cout << "时间窗口 " << timeWindow + 1 << ": ";
            
            for (int i = 0; i < pointCount; i++) {
                int x = xDist(rng);
                int y = yDist(rng);
                allPoints.emplace_back(x, y, timeWindow);
                std::cout << "(" << x << "," << y << ") ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
    void clearScreen() {
        std::cout << "\033[2J\033[H"; // Clear screen and move cursor to top-left
    }
    
    void drawBorder() {
        // Top border without Y-axis separator
        std::cout << Colors::WHITE << "   ┌";
        for (int x = 0; x < width; x++) {
            std::cout << "───";
            if (x < width - 1) std::cout << "┬";
        }
        std::cout << "┐" << Colors::RESET << std::endl;
        
        // Side borders will be drawn with each row
        
        // Bottom border is drawn after all rows
    }
    
    void drawBottomBorder() {
        // Bottom border without Y-axis separator
        std::cout << Colors::WHITE << "   └";
        for (int x = 0; x < width; x++) {
            std::cout << "───";
            if (x < width - 1) std::cout << "┴";
        }
        std::cout << "┘" << Colors::RESET << std::endl;
        

    }
    
    void drawTimeWindow(int currentTimeWindow) {
        clearScreen();
        
        // Title
        std::cout << Colors::BRIGHT_WHITE << Colors::BG_BLUE 
                  << "  2D 分层模型可视化 - 时间窗口 " << (currentTimeWindow + 1) 
                  << "/" << maxTimeWindows << "  " << Colors::RESET << std::endl << std::endl;
        
        // Create 2D grid
        std::vector<std::vector<Point*>> grid(height, std::vector<Point*>(width, nullptr));
        
        // Place points for current time window
        for (auto& point : allPoints) {
            if (point.timeWindow == currentTimeWindow) {
                grid[point.y][point.x] = &point;
            }
        }
        
        // Add X-axis labels at the top
        std::cout << Colors::BRIGHT_WHITE << "   ";
        for (int x = 0; x < width; x++) {
            if (x < 10) {
                std::cout << " " << x << " ";  // Single digit: space + digit + space
            } else {
                std::cout << x << " ";   // Double digit: 2 digits + space
            }
            if (x < width - 1) {
                std::cout << " ";  // Additional space between columns
            }
        }
        std::cout << Colors::RESET << std::endl;
        
        // Add horizontal separator after top border
        std::cout << Colors::WHITE << "   ├";
        for (int x = 0; x < width; x++) {
            std::cout << "───";
            if (x < width - 1) std::cout << "┼";
        }
        std::cout << "┤" << Colors::RESET << std::endl;
        
        // Draw each row
        for (int y = 0; y < height; y++) {
            // Y-axis label with separator line
            if (y < 10) {
                std::cout << Colors::BRIGHT_WHITE << " " << y << " │" << Colors::RESET;
            } else {
                std::cout << Colors::BRIGHT_WHITE << y << " │" << Colors::RESET;
            }
            
            for (int x = 0; x < width; x++) {
                if (grid[y][x] != nullptr) {
                    Point* p = grid[y][x];
                    std::cout << p->bgColor << "   " << Colors::RESET; // Colored cell
                } else {
                    std::cout << "   "; // Empty cell with default background
                }
                
                // Vertical grid line
                if (x < width - 1) {
                    std::cout << Colors::BRIGHT_WHITE << "│" << Colors::RESET;
                }
            }
            std::cout << Colors::BRIGHT_WHITE << "│" << Colors::RESET << std::endl;
            
            // Horizontal grid line (except for last row)
            if (y < height - 1) {
                std::cout << Colors::BRIGHT_WHITE << "   ├";
                for (int x = 0; x < width; x++) {
                    std::cout << "───";
                    if (x < width - 1) std::cout << "┼";
                }
                std::cout << "┤" << Colors::RESET << std::endl;
            }
        }
        

        
        // Legend
        std::cout << std::endl << Colors::BRIGHT_YELLOW << "图例:" << Colors::RESET;
        std::set<int> currentTimeWindows;
        for (auto& point : allPoints) {
            if (point.timeWindow == currentTimeWindow) {
                if (currentTimeWindows.find(point.timeWindow) == currentTimeWindows.end()) {
                    currentTimeWindows.insert(point.timeWindow);
                    std::cout << " " << point.bgColor << "   " << Colors::RESET 
                              << " = 时间窗口 " << (point.timeWindow + 1);
                }
            }
        }
        std::cout << std::endl;
        
        // Statistics
        int pointCount = 0;
        for (auto& point : allPoints) {
            if (point.timeWindow == currentTimeWindow) {
                pointCount++;
            }
        }
        std::cout << Colors::BRIGHT_GREEN << "当前时间窗口点数: " << pointCount << Colors::RESET << std::endl;
        std::cout << Colors::CYAN << "网格尺寸: " << width << "x" << height << Colors::RESET << std::endl;
    }
    
    void animate() {
        std::cout << Colors::BRIGHT_MAGENTA 
                  << "开始动画演示...(按 Ctrl+C 停止)" << Colors::RESET << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        for (int timeWindow = 0; timeWindow < maxTimeWindows; timeWindow++) {
            drawTimeWindow(timeWindow);
            std::this_thread::sleep_for(std::chrono::milliseconds(2000)); // 2 seconds per frame
        }
        
        // Show all layers summary
        showSummary();
    }
    
    void showSummary() {
        clearScreen();
        std::cout << Colors::BRIGHT_WHITE << Colors::BG_MAGENTA 
                  << "  所有时间窗口总结  " << Colors::RESET << std::endl << std::endl;
        
        for (int timeWindow = 0; timeWindow < maxTimeWindows; timeWindow++) {
            std::cout << Colors::BRIGHT_CYAN << "时间窗口 " << (timeWindow + 1) << ": " << Colors::RESET;
            
            for (auto& point : allPoints) {
                if (point.timeWindow == timeWindow) {
                    std::cout << point.bgColor << "   " << Colors::RESET 
                              << "(" << point.x << "," << point.y << ") ";
                }
            }
            std::cout << std::endl;
        }
        
        std::cout << std::endl << Colors::BRIGHT_GREEN 
                  << "总点数: " << allPoints.size() << Colors::RESET << std::endl;
        std::cout << Colors::BRIGHT_YELLOW 
                  << "动画演示完成!" << Colors::RESET << std::endl;
    }
    
    void interactiveMode() {
        std::string input;
        int currentWindow = 0;
        
        while (true) {
            drawTimeWindow(currentWindow);
            std::cout << std::endl << Colors::BRIGHT_WHITE 
                      << "交互模式: [n]下一个 [p]上一个 [q]退出 [a]自动播放: " << Colors::RESET;
            std::cin >> input;
            
            if (input == "n" || input == "N") {
                currentWindow = (currentWindow + 1) % maxTimeWindows;
            } else if (input == "p" || input == "P") {
                currentWindow = (currentWindow - 1 + maxTimeWindows) % maxTimeWindows;
            } else if (input == "q" || input == "Q") {
                break;
            } else if (input == "a" || input == "A") {
                animate();
                break;
            }
        }
    }
};

int main() {
    // Check if terminal supports color
    if (!isatty(STDOUT_FILENO)) {
        std::cout << "警告: 终端可能不支持颜色显示" << std::endl;
    }
    
    std::cout << Colors::BRIGHT_CYAN << Colors::BG_BLACK
              << "  欢迎使用彩色终端2D分层模型可视化工具  " << Colors::RESET << std::endl << std::endl;
    
    // Configuration
    const int GRID_WIDTH = 20;
    const int GRID_HEIGHT = 12;
    const int TIME_WINDOWS = 5;
    
    std::cout << "配置参数:" << std::endl;
    std::cout << "- 网格尺寸: " << GRID_WIDTH << "x" << GRID_HEIGHT << std::endl;
    std::cout << "- 时间窗口数: " << TIME_WINDOWS << std::endl << std::endl;
    
    TerminalVisualizer viz(GRID_WIDTH, GRID_HEIGHT, TIME_WINDOWS);
    
    std::cout << "选择运行模式:" << std::endl;
    std::cout << "1. 自动播放动画" << std::endl;
    std::cout << "2. 交互模式" << std::endl;
    std::cout << "请输入选择 (1 或 2): ";
    
    int choice;
    std::cin >> choice;
    
    if (choice == 1) {
        viz.animate();
    } else {
        viz.interactiveMode();
    }
    
    std::cout << Colors::BRIGHT_GREEN << "感谢使用!" << Colors::RESET << std::endl;
    return 0;
}